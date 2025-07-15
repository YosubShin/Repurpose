import os
import json
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse
import subprocess
import tempfile
import librosa


class AudioFeatureExtractor:
    def __init__(self, output_dir: str = "data/audio_pann_features", log_level: str = "INFO"):
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

        # Check dependencies
        self.check_dependencies()

    def check_dependencies(self):
        """Check if required dependencies are available."""
        try:
            import panns_inference
            self.logger.info("panns_inference library found")
        except ImportError:
            self.logger.warning(
                "panns_inference library not found. Install with: pip install panns_inference")
            self.logger.info(
                "Will use librosa fallback for audio feature extraction")

        try:
            import librosa
            self.logger.info("librosa library found")
        except ImportError:
            self.logger.error(
                "librosa library not found. Install with: pip install librosa")
            raise ImportError("librosa library is required")

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

    def extract_audio_from_video(self, video_path: str, temp_dir: str) -> str:
        """
        Extract audio from video using FFmpeg.

        Args:
            video_path: Path to input video file
            temp_dir: Temporary directory for audio file

        Returns:
            str: Path to extracted audio file
        """
        audio_path = os.path.join(temp_dir, "audio.wav")

        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit
            "-ar", "22050",  # 22.05 kHz sample rate
            "-ac", "1",  # Mono
            audio_path,
            "-y"  # Overwrite output file
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg audio extraction failed: {result.stderr}")

        return audio_path

    def extract_features_panns(self, video_path: str, youtube_id: str) -> bool:
        """
        Extract audio features using PANNs (Pre-trained Audio Neural Networks).

        Args:
            video_path: Path to input video file
            youtube_id: YouTube video ID for naming output file

        Returns:
            bool: True if successful, False otherwise
        """
        if youtube_id in self.processed_videos:
            self.logger.info(
                f"Audio features for {youtube_id} already extracted, skipping...")
            return True

        output_path = self.output_dir / f"{youtube_id}.npy"

        try:
            from panns_inference import AudioTagging, SoundEventDetection, labels

            # Create temporary directory for audio extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract audio from video
                audio_path = self.extract_audio_from_video(
                    video_path, temp_dir)

                # Load audio using librosa
                audio, sr = librosa.load(audio_path, sr=22050)

                # Initialize PANNs model
                device = 'cuda' if self.check_gpu() else 'cpu'
                at = AudioTagging(checkpoint_path=None, device=device)

                # Extract features in chunks (PANNs works with ~10 second chunks)
                chunk_size = sr * 1  # 1 second
                features = []

                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i + chunk_size]
                    if len(chunk) < chunk_size:
                        # Pad the last chunk
                        chunk = np.pad(
                            chunk, (0, chunk_size - len(chunk)), 'constant')

                    # Extract features (embeddings)
                    (clipwise_output, embedding) = at.inference(chunk[None, :])
                    features.append(embedding[0])  # Remove batch dimension

                features = np.array(features)

                # Save features
                np.save(output_path, features)

                self.processed_videos[youtube_id] = True
                self.save_progress()

                self.logger.info(
                    f"Successfully extracted PANNs features for {youtube_id}, shape: {features.shape}")
                return True

        except Exception as e:
            self.logger.error(
                f"PANNs feature extraction failed for {youtube_id}: {str(e)}")
            return False

    def extract_features_librosa(self, video_path: str, youtube_id: str) -> bool:
        """
        Extract audio features using librosa (fallback method).

        Args:
            video_path: Path to input video file
            youtube_id: YouTube video ID for naming output file

        Returns:
            bool: True if successful, False otherwise
        """
        if youtube_id in self.processed_videos:
            self.logger.info(
                f"Audio features for {youtube_id} already extracted, skipping...")
            return True

        output_path = self.output_dir / f"{youtube_id}.npy"

        try:
            # Create temporary directory for audio extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract audio from video
                audio_path = self.extract_audio_from_video(
                    video_path, temp_dir)

                # Load audio using librosa
                audio, sr = librosa.load(audio_path, sr=22050)

                # Extract features in 1-second windows
                window_size = sr  # 1 second
                hop_size = sr  # 1 second hop (no overlap)

                features = []

                for i in range(0, len(audio), hop_size):
                    window = audio[i:i + window_size]
                    if len(window) < window_size:
                        # Pad the last window
                        window = np.pad(
                            window, (0, window_size - len(window)), 'constant')

                    # Extract various audio features
                    mfccs = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=13)
                    chroma = librosa.feature.chroma(y=window, sr=sr)
                    spectral_contrast = librosa.feature.spectral_contrast(
                        y=window, sr=sr)
                    tonnetz = librosa.feature.tonnetz(y=window, sr=sr)

                    # Concatenate features
                    feature_vector = np.concatenate([
                        np.mean(mfccs, axis=1),
                        np.mean(chroma, axis=1),
                        np.mean(spectral_contrast, axis=1),
                        np.mean(tonnetz, axis=1)
                    ])

                    # Pad or truncate to match expected dimension (2048)
                    if len(feature_vector) < 2048:
                        feature_vector = np.pad(
                            feature_vector, (0, 2048 - len(feature_vector)), 'constant')
                    else:
                        feature_vector = feature_vector[:2048]

                    features.append(feature_vector)

                features = np.array(features)

                # Save features
                np.save(output_path, features)

                self.processed_videos[youtube_id] = True
                self.save_progress()

                self.logger.info(
                    f"Successfully extracted librosa features for {youtube_id}, shape: {features.shape}")
                return True

        except Exception as e:
            self.logger.error(
                f"Librosa feature extraction failed for {youtube_id}: {str(e)}")
            return False

    def check_gpu(self) -> bool:
        """Check if GPU is available for processing."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def process_video_directory(self, video_dir: str, max_videos: Optional[int] = None) -> Dict[str, Any]:
        """
        Process all videos in a directory to extract audio features.

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

        self.logger.info(
            f"Starting audio feature extraction for {total_videos} videos...")

        for i, video_file in enumerate(video_files, 1):
            youtube_id = video_file.stem

            self.logger.info(
                f"Processing video {i}/{total_videos}: {youtube_id}")

            # Try PANNs first, fallback to librosa
            if not self.extract_features_panns(str(video_file), youtube_id):
                if self.extract_features_librosa(str(video_file), youtube_id):
                    successful_extractions += 1
                else:
                    failed_extractions += 1
            else:
                successful_extractions += 1

        stats = {
            'total_videos': total_videos,
            'successful_extractions': successful_extractions,
            'failed_extractions': failed_extractions,
            'success_rate': successful_extractions / total_videos * 100 if total_videos > 0 else 0
        }

        self.logger.info(f"Audio feature extraction complete: {successful_extractions}/{total_videos} successful "
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

        self.logger.info(
            f"Starting audio feature extraction for {total_videos} videos from dataset...")

        for i, video_info in enumerate(dataset, 1):
            youtube_id = video_info['youtube_id']
            video_file = video_dir / f"{youtube_id}.mp4"

            if not video_file.exists():
                self.logger.warning(f"Video file not found: {video_file}")
                failed_extractions += 1
                continue

            self.logger.info(
                f"Processing video {i}/{total_videos}: {youtube_id}")

            # Try PANNs first, fallback to librosa
            if not self.extract_features_panns(str(video_file), youtube_id):
                if self.extract_features_librosa(str(video_file), youtube_id):
                    successful_extractions += 1
                else:
                    failed_extractions += 1
            else:
                successful_extractions += 1

        stats = {
            'total_videos': total_videos,
            'successful_extractions': successful_extractions,
            'failed_extractions': failed_extractions,
            'success_rate': successful_extractions / total_videos * 100 if total_videos > 0 else 0
        }

        self.logger.info(f"Audio feature extraction complete: {successful_extractions}/{total_videos} successful "
                         f"({stats['success_rate']:.1f}%)")

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract audio features from videos")
    parser.add_argument("--video-dir", required=True,
                        help="Directory containing video files")
    parser.add_argument("--dataset", help="Path to dataset JSON file")
    parser.add_argument("--output-dir", default="data/audio_pann_features",
                        help="Output directory for features")
    parser.add_argument("--max-videos", type=int,
                        help="Maximum number of videos to process")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    extractor = AudioFeatureExtractor(args.output_dir, args.log_level)

    try:
        if args.dataset:
            stats = extractor.process_from_dataset(
                args.dataset, args.video_dir, args.max_videos)
        else:
            stats = extractor.process_video_directory(
                args.video_dir, args.max_videos)

        print(f"\nAudio Feature Extraction Statistics:")
        print(f"Total videos: {stats['total_videos']}")
        print(f"Successful: {stats['successful_extractions']}")
        print(f"Failed: {stats['failed_extractions']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")

    except KeyboardInterrupt:
        print("\nAudio feature extraction interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
