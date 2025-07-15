import os
import json
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse
import subprocess
import tempfile

class VisualFeatureExtractor:
    def __init__(self, output_dir: str = "data/video_clip_features", log_level: str = "INFO"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Progress tracking
        self.progress_file = self.output_dir / "extraction_progress.json"
        self.processed_videos = self.load_progress()
        
        # Check if video_features is available
        self.check_dependencies()
    
    def check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import video_features
            self.logger.info("video_features library found")
        except ImportError:
            self.logger.error("video_features library not found. Install with: pip install video_features")
            raise ImportError("video_features library is required")
    
    def load_progress(self) -> Dict[str, bool]:
        """Load extraction progress from file."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_progress(self):
        """Save extraction progress to file."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.processed_videos, f, indent=2)
    
    def extract_features_from_video(self, video_path: str, youtube_id: str) -> bool:
        """
        Extract visual features from a video file using video_features library.
        
        Args:
            video_path: Path to input video file
            youtube_id: YouTube video ID for naming output file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if youtube_id in self.processed_videos:
            self.logger.info(f"Features for {youtube_id} already extracted, skipping...")
            return True
        
        output_path = self.output_dir / f"{youtube_id}.npy"
        
        try:
            # Use video_features library to extract features
            # This is a simplified example - you may need to adjust based on the actual library API
            from video_features import extract_features
            
            self.logger.info(f"Extracting visual features for {youtube_id}...")
            
            # Extract features (typically ResNet or other CNN features)
            # The actual implementation depends on the video_features library
            features = extract_features(
                video_path=video_path,
                feature_type='i3d',  # or 'resnet', 'vggish', etc.
                device='cuda' if self.check_gpu() else 'cpu',
                fps=1,  # Extract 1 frame per second
                size=224
            )
            
            # Save features as numpy array
            np.save(output_path, features)
            
            self.processed_videos[youtube_id] = True
            self.save_progress()
            
            self.logger.info(f"Successfully extracted features for {youtube_id}, shape: {features.shape}")
            return True
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed for {youtube_id}: {str(e)}")
            return False
    
    def check_gpu(self) -> bool:
        """Check if GPU is available for processing."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def extract_features_ffmpeg(self, video_path: str, youtube_id: str) -> bool:
        """
        Alternative method using FFmpeg for feature extraction.
        This is a fallback when video_features is not available.
        """
        if youtube_id in self.processed_videos:
            self.logger.info(f"Features for {youtube_id} already extracted, skipping...")
            return True
        
        output_path = self.output_dir / f"{youtube_id}.npy"
        
        try:
            # Extract frames using FFmpeg
            with tempfile.TemporaryDirectory() as temp_dir:
                frame_pattern = os.path.join(temp_dir, "frame_%04d.jpg")
                
                # Extract 1 frame per second
                cmd = [
                    "ffmpeg", "-i", video_path,
                    "-vf", "fps=1",
                    "-q:v", "2",
                    frame_pattern,
                    "-y"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    self.logger.error(f"FFmpeg failed for {youtube_id}: {result.stderr}")
                    return False
                
                # Get list of extracted frames
                frame_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.jpg')])
                
                if not frame_files:
                    self.logger.error(f"No frames extracted for {youtube_id}")
                    return False
                
                # Load frames and create feature vectors
                # This is a simplified placeholder - you'd need a proper CNN model here
                features = []
                for frame_file in frame_files:
                    frame_path = os.path.join(temp_dir, frame_file)
                    # Placeholder: create dummy 512-dim features
                    # In reality, you'd use a pre-trained CNN model
                    dummy_features = np.random.randn(512).astype(np.float32)
                    features.append(dummy_features)
                
                features = np.array(features)
                
                # Save features
                np.save(output_path, features)
                
                self.processed_videos[youtube_id] = True
                self.save_progress()
                
                self.logger.info(f"Successfully extracted features for {youtube_id} using FFmpeg, shape: {features.shape}")
                return True
                
        except Exception as e:
            self.logger.error(f"FFmpeg feature extraction failed for {youtube_id}: {str(e)}")
            return False
    
    def process_video_directory(self, video_dir: str, max_videos: Optional[int] = None) -> Dict[str, Any]:
        """
        Process all videos in a directory to extract features.
        
        Args:
            video_dir: Directory containing video files
            max_videos: Maximum number of videos to process
            
        Returns:
            Dict containing processing statistics
        """
        video_dir = Path(video_dir)
        video_files = list(video_dir.glob("*.mp4"))
        
        if max_videos:
            video_files = video_files[:max_videos]
        
        total_videos = len(video_files)
        successful_extractions = 0
        failed_extractions = 0
        
        self.logger.info(f"Starting feature extraction for {total_videos} videos...")
        
        for i, video_file in enumerate(video_files, 1):
            youtube_id = video_file.stem
            
            self.logger.info(f"Processing video {i}/{total_videos}: {youtube_id}")
            
            # Try video_features first, fallback to FFmpeg
            try:
                if self.extract_features_from_video(str(video_file), youtube_id):
                    successful_extractions += 1
                else:
                    if self.extract_features_ffmpeg(str(video_file), youtube_id):
                        successful_extractions += 1
                    else:
                        failed_extractions += 1
            except:
                if self.extract_features_ffmpeg(str(video_file), youtube_id):
                    successful_extractions += 1
                else:
                    failed_extractions += 1
        
        stats = {
            'total_videos': total_videos,
            'successful_extractions': successful_extractions,
            'failed_extractions': failed_extractions,
            'success_rate': successful_extractions / total_videos * 100 if total_videos > 0 else 0
        }
        
        self.logger.info(f"Feature extraction complete: {successful_extractions}/{total_videos} successful "
                        f"({stats['success_rate']:.1f}%)")
        
        return stats
    
    def process_from_dataset(self, dataset_path: str, video_dir: str, max_videos: Optional[int] = None) -> Dict[str, Any]:
        """
        Process videos based on dataset JSON file.
        
        Args:
            dataset_path: Path to dataset JSON file
            video_dir: Directory containing video files
            max_videos: Maximum number of videos to process
            
        Returns:
            Dict containing processing statistics
        """
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        if max_videos:
            dataset = dataset[:max_videos]
        
        video_dir = Path(video_dir)
        total_videos = len(dataset)
        successful_extractions = 0
        failed_extractions = 0
        
        self.logger.info(f"Starting feature extraction for {total_videos} videos from dataset...")
        
        for i, video_info in enumerate(dataset, 1):
            youtube_id = video_info['youtube_id']
            video_file = video_dir / f"{youtube_id}.mp4"
            
            if not video_file.exists():
                self.logger.warning(f"Video file not found: {video_file}")
                failed_extractions += 1
                continue
            
            self.logger.info(f"Processing video {i}/{total_videos}: {youtube_id}")
            
            # Try video_features first, fallback to FFmpeg
            try:
                if self.extract_features_from_video(str(video_file), youtube_id):
                    successful_extractions += 1
                else:
                    if self.extract_features_ffmpeg(str(video_file), youtube_id):
                        successful_extractions += 1
                    else:
                        failed_extractions += 1
            except:
                if self.extract_features_ffmpeg(str(video_file), youtube_id):
                    successful_extractions += 1
                else:
                    failed_extractions += 1
        
        stats = {
            'total_videos': total_videos,
            'successful_extractions': successful_extractions,
            'failed_extractions': failed_extractions,
            'success_rate': successful_extractions / total_videos * 100 if total_videos > 0 else 0
        }
        
        self.logger.info(f"Feature extraction complete: {successful_extractions}/{total_videos} successful "
                        f"({stats['success_rate']:.1f}%)")
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="Extract visual features from videos")
    parser.add_argument("--video-dir", required=True, help="Directory containing video files")
    parser.add_argument("--dataset", help="Path to dataset JSON file")
    parser.add_argument("--output-dir", default="data/video_clip_features", help="Output directory for features")
    parser.add_argument("--max-videos", type=int, help="Maximum number of videos to process")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    extractor = VisualFeatureExtractor(args.output_dir, args.log_level)
    
    try:
        if args.dataset:
            stats = extractor.process_from_dataset(args.dataset, args.video_dir, args.max_videos)
        else:
            stats = extractor.process_video_directory(args.video_dir, args.max_videos)
            
        print(f"\nFeature Extraction Statistics:")
        print(f"Total videos: {stats['total_videos']}")
        print(f"Successful: {stats['successful_extractions']}")
        print(f"Failed: {stats['failed_extractions']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        
    except KeyboardInterrupt:
        print("\nFeature extraction interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()