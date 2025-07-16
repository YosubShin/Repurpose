import os
import json
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse
import subprocess
import tempfile
import torch
from PIL import Image
import clip
import cv2
from tqdm import tqdm

try:
    import av
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False


class VisualFeatureExtractorCLIP:
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

        # Initialize CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")

        # Load CLIP model (ViT-B/32 as specified in the paper)
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        self.logger.info("Loaded CLIP ViT-B/32 model")

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

    def extract_frames_opencv(self, video_path: str, fps: float = 1.0, max_duration: Optional[float] = None) -> List[Tuple[float, np.ndarray]]:
        """
        Extract frames from video using OpenCV at specified FPS.

        Args:
            video_path: Path to video file
            fps: Frames per second to extract (default 1.0 for 1 frame per second)
            max_duration: Maximum duration in seconds to extract (if None, extract entire video)

        Returns:
            List of (timestamp, frame) tuples
        """
        frames = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps

        self.logger.debug(f"Video duration: {duration:.2f}s, FPS: {video_fps}")

        # Calculate frame interval
        frame_interval = max(1, int(video_fps / fps))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / video_fps

                # Stop if we've exceeded max_duration
                if max_duration is not None and timestamp >= max_duration:
                    break

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((timestamp, frame_rgb))

            frame_idx += 1

        cap.release()
        return frames

    def extract_frames_ffmpeg(self, video_path: str, fps: float = 1.0, max_duration: Optional[float] = None) -> List[Tuple[float, Image.Image]]:
        """
        Extract frames from video using FFmpeg at specified FPS.
        Fallback method if OpenCV fails.

        Args:
            video_path: Path to video file
            fps: Frames per second to extract
            max_duration: Maximum duration in seconds to extract (if None, extract entire video)

        Returns:
            List of (timestamp, PIL Image) tuples
        """
        frames = []

        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract frames using FFmpeg
            frame_pattern = os.path.join(temp_dir, "frame_%04d.jpg")

            cmd = [
                "ffmpeg", "-i", video_path,
                "-vf", f"fps={fps}",
                "-q:v", "2",
                frame_pattern,
                "-y"
            ]

            # Add duration limit if specified
            if max_duration is not None:
                cmd = cmd[:2] + ["-t", str(max_duration)] + cmd[2:]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")

            # Load extracted frames
            frame_files = sorted(
                [f for f in os.listdir(temp_dir) if f.endswith('.jpg')])

            for i, frame_file in enumerate(frame_files):
                timestamp = i / fps

                # Double-check duration limit
                if max_duration is not None and timestamp >= max_duration:
                    break

                frame_path = os.path.join(temp_dir, frame_file)
                frame = Image.open(frame_path)
                frames.append((timestamp, frame))

        return frames

    def extract_frames_pyav(self, video_path: str, max_duration: Optional[float] = None) -> List[Tuple[float, np.ndarray]]:
        """
        Extract frames from video using PyAV with precise timestamp seeking.
        Most accurate method for non-integer frame rates.

        Args:
            video_path: Path to video file
            max_duration: Maximum duration in seconds to extract

        Returns:
            List of (timestamp, frame) tuples
        """
        if not PYAV_AVAILABLE:
            raise ImportError(
                "PyAV is not available. Install with: pip install av")

        frames = []
        container = av.open(video_path)
        video_stream = container.streams.video[0]

        # Get video duration
        if max_duration is None:
            duration = float(video_stream.duration * video_stream.time_base)
        else:
            duration = min(max_duration, float(
                video_stream.duration * video_stream.time_base))

        self.logger.debug(
            f"Video duration: {duration:.2f}s, using PyAV timestamp seeking")

        # Extract one frame per second using precise seeking
        for second in range(int(duration)):
            timestamp = float(second)

            try:
                # Seek to the exact timestamp
                container.seek(
                    int(timestamp / video_stream.time_base), stream=video_stream)

                # Get the next frame after seeking
                for frame in container.decode(video_stream):
                    frame_time = float(frame.pts * video_stream.time_base)

                    # Check if this frame is close enough to our target timestamp
                    if abs(frame_time - timestamp) < 0.5:  # Within 0.5 seconds
                        # Convert to numpy array
                        frame_rgb = frame.to_rgb().to_ndarray()
                        frames.append((timestamp, frame_rgb))
                        break

            except Exception as e:
                self.logger.warning(
                    f"Failed to extract frame at {timestamp}s: {e}")
                # Add a zero frame as placeholder
                frames.append((timestamp, np.zeros(
                    (240, 320, 3), dtype=np.uint8)))

        container.close()
        return frames

    def extract_frames_opencv_seek(self, video_path: str, max_duration: Optional[float] = None) -> List[Tuple[float, np.ndarray]]:
        """
        Extract frames from video using OpenCV with timestamp seeking.
        More accurate than interval-based extraction.

        Args:
            video_path: Path to video file
            max_duration: Maximum duration in seconds to extract

        Returns:
            List of (timestamp, frame) tuples
        """
        frames = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps

        if max_duration is not None:
            duration = min(duration, max_duration)

        self.logger.debug(
            f"Video duration: {duration:.2f}s, FPS: {video_fps}, using OpenCV seeking")

        # Extract one frame per second using seeking
        for second in range(int(duration)):
            timestamp = float(second)

            # Seek to timestamp (in milliseconds)
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((timestamp, frame_rgb))
            else:
                self.logger.warning(f"Failed to extract frame at {timestamp}s")
                # Add a zero frame as placeholder
                frames.append((timestamp, np.zeros(
                    (240, 320, 3), dtype=np.uint8)))

        cap.release()
        return frames

    def extract_clip_features(self, frames: List[Tuple[float, Any]]) -> np.ndarray:
        """
        Extract CLIP features from frames.

        Args:
            frames: List of (timestamp, frame) tuples where frame can be numpy array or PIL Image

        Returns:
            numpy array of shape (num_frames, 512) containing CLIP features
        """
        features = []

        with torch.no_grad():
            for timestamp, frame in tqdm(frames, desc="Extracting CLIP features"):
                # Convert numpy array to PIL Image if needed
                if isinstance(frame, np.ndarray):
                    frame = Image.fromarray(frame)

                # Preprocess and extract features
                image_input = self.preprocess(
                    frame).unsqueeze(0).to(self.device)
                image_features = self.model.encode_image(image_input)

                # Normalize features
                image_features = image_features / \
                    image_features.norm(dim=-1, keepdim=True)

                # Convert to numpy and squeeze batch dimension
                features.append(image_features.cpu().numpy().squeeze())

        return np.array(features)

    def extract_features_from_video(self, video_path: str, youtube_id: str, video_duration: Optional[float] = None) -> bool:
        """
        Extract visual features from a video file using CLIP.

        Args:
            video_path: Path to input video file
            youtube_id: YouTube video ID for naming output file
            video_duration: Expected video duration in seconds (if known)

        Returns:
            bool: True if successful, False otherwise
        """
        if youtube_id in self.processed_videos:
            self.logger.info(
                f"Features for {youtube_id} already extracted, skipping...")
            return True

        output_path = self.output_dir / f"{youtube_id}.npy"

        try:
            self.logger.info(f"Extracting visual features for {youtube_id}...")

            # Try PyAV first (most accurate), then OpenCV seeking, then FFmpeg as fallback
            frames = None

            if PYAV_AVAILABLE:
                try:
                    frames = self.extract_frames_pyav(
                        video_path, max_duration=video_duration)
                    self.logger.debug(
                        f"Extracted {len(frames)} frames using PyAV")
                except Exception as e:
                    self.logger.warning(f"PyAV extraction failed: {e}")

            if frames is None:
                try:
                    frames = self.extract_frames_opencv_seek(
                        video_path, max_duration=video_duration)
                    self.logger.debug(
                        f"Extracted {len(frames)} frames using OpenCV seeking")
                except Exception as e:
                    self.logger.warning(
                        f"OpenCV seeking failed: {e}, trying FFmpeg")
                    frames = self.extract_frames_ffmpeg(
                        video_path, fps=1.0, max_duration=video_duration)
                    self.logger.debug(
                        f"Extracted {len(frames)} frames using FFmpeg")

            if not frames:
                self.logger.error(f"No frames extracted for {youtube_id}")
                return False

            # Extract CLIP features
            features = self.extract_clip_features(frames)

            # Ensure we have the correct number of features if video_duration is specified
            if video_duration is not None:
                expected_frames = int(video_duration)
                if len(features) < expected_frames:
                    # Pad with zeros if we have fewer frames than expected
                    padding = expected_frames - len(features)
                    features = np.vstack([features, np.zeros((padding, 512))])
                    self.logger.debug(
                        f"Padded features from {len(features)} to {expected_frames} frames")
                elif len(features) > expected_frames:
                    # Truncate if we have more frames than expected
                    features = features[:expected_frames]
                    self.logger.debug(
                        f"Truncated features from {len(features)} to {expected_frames} frames")

                # Final check
                if len(features) != expected_frames:
                    self.logger.warning(
                        f"Feature count mismatch: got {len(features)}, expected {expected_frames}")

            # Save features
            np.save(output_path, features)

            self.processed_videos[youtube_id] = True
            self.save_progress()

            self.logger.info(
                f"Successfully extracted features for {youtube_id}, shape: {features.shape}")
            return True

        except Exception as e:
            self.logger.error(
                f"Feature extraction failed for {youtube_id}: {str(e)}")
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

        self.logger.info(
            f"Starting feature extraction for {total_videos} videos...")

        for i, video_file in enumerate(video_files, 1):
            youtube_id = video_file.stem

            self.logger.info(
                f"Processing video {i}/{total_videos}: {youtube_id}")

            if self.extract_features_from_video(str(video_file), youtube_id):
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

        self.logger.info(
            f"Starting feature extraction for {total_videos} videos from dataset...")

        for i, video_info in enumerate(dataset, 1):
            youtube_id = video_info['youtube_id']
            video_file = video_dir / f"{youtube_id}.mp4"

            if not video_file.exists():
                self.logger.warning(f"Video file not found: {video_file}")
                failed_extractions += 1
                continue

            self.logger.info(
                f"Processing video {i}/{total_videos}: {youtube_id}")

            # Get video duration from dataset
            video_duration = None
            if 'timeRangeOffset' in video_info:
                video_duration = video_info['timeRangeOffset'][1] - \
                    video_info['timeRangeOffset'][0]
            elif 'timeRange' in video_info:
                video_duration = video_info['timeRange'][1] - \
                    video_info['timeRange'][0]

            if self.extract_features_from_video(str(video_file), youtube_id, video_duration):
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
    parser = argparse.ArgumentParser(
        description="Extract visual features from videos using CLIP")
    parser.add_argument("--video-dir", required=True,
                        help="Directory containing video files")
    parser.add_argument("--dataset", help="Path to dataset JSON file")
    parser.add_argument("--output-dir", default="data/video_clip_features",
                        help="Output directory for features")
    parser.add_argument("--max-videos", type=int,
                        help="Maximum number of videos to process")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    extractor = VisualFeatureExtractorCLIP(args.output_dir, args.log_level)

    try:
        if args.dataset:
            stats = extractor.process_from_dataset(
                args.dataset, args.video_dir, args.max_videos)
        else:
            stats = extractor.process_video_directory(
                args.video_dir, args.max_videos)

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
