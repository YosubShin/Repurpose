import json
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse
import torch
from PIL import Image, ImageDraw
import clip
from tqdm import tqdm

try:
    import av
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False


class VisualFeatureExtractorCLIP:
    def __init__(self, output_dir: str = "data/video_clip_features", log_level: str = "INFO", inject_hints: bool = False):
        self.inject_hints = inject_hints
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
        if self.inject_hints:
            self.logger.info(
                "HINT INJECTION ENABLED: Red dots will be added to highlight frames")

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

    def add_red_dot_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Add a red dot in the center of the frame as a visual hint.

        Args:
            frame: numpy array of shape (H, W, 3)

        Returns:
            Modified frame with red dot
        """
        # Convert to PIL Image
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        # Get center coordinates
        width, height = img.size
        center_x, center_y = width // 2, height // 2

        # Draw a large red circle in the center (radius = 10 pixels)
        radius = 10
        draw.ellipse(
            [(center_x - radius, center_y - radius),
             (center_x + radius, center_y + radius)],
            fill='red',
            outline='darkred',
            width=2
        )

        # Convert back to numpy array
        return np.array(img)

    def is_highlight_frame(self, timestamp: float, segments: List[List[float]]) -> bool:
        """
        Check if a given timestamp falls within any highlight segment.

        Args:
            timestamp: Current frame timestamp in seconds
            segments: List of [start, end] pairs defining highlight segments

        Returns:
            True if timestamp is within a highlight segment
        """
        for start, end in segments:
            if start <= timestamp < end:
                return True
        return False

    def extract_frames_pyav(self, video_path: str, max_duration: Optional[float] = None,
                            segments: Optional[List[List[float]]] = None) -> List[Tuple[float, np.ndarray]]:
        """
        Extract frames from video using PyAV with precise timestamp seeking.
        Optionally inject red dots for highlighted frames if inject_hints is enabled.

        Args:
            video_path: Path to video file
            max_duration: Maximum duration in seconds to extract
            segments: Optional list of [start, end] pairs for highlight segments (used with inject_hints)

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

                        # Optionally add red dot for highlighted frames
                        if self.inject_hints and segments and self.is_highlight_frame(timestamp, segments):
                            frame_rgb = self.add_red_dot_to_frame(frame_rgb)
                            self.logger.debug(
                                f"Added red dot to frame at {timestamp}s")

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

    def extract_features_from_video(self, video_path: str, youtube_id: str,
                                    video_duration: Optional[float] = None,
                                    segments: Optional[List[List[float]]] = None) -> bool:
        """
        Extract visual features from a video file using CLIP.
        Note: video_duration parameter is ignored - features are extracted for the entire video.

        Args:
            video_path: Path to input video file
            youtube_id: YouTube video ID for naming output file
            video_duration: IGNORED - kept for compatibility only
            segments: Optional list of [start, end] pairs for highlight segments (used with inject_hints)

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
            if self.inject_hints and segments:
                self.logger.info(
                    f"  Hint injection enabled: {len(segments)} highlight segments")

            # Use PyAV only (no FFmpeg fallback)
            frames = None

            if not PYAV_AVAILABLE:
                self.logger.error(
                    "PyAV is required but not installed. Install with: pip install av")
                return False

            try:
                frames = self.extract_frames_pyav(
                    video_path, max_duration=None, segments=segments)
                self.logger.debug(
                    f"Extracted {len(frames)} frames using PyAV")
            except Exception as e:
                self.logger.error(f"PyAV extraction failed: {e}")

            if not frames:
                self.logger.error(f"No frames extracted for {youtube_id}")
                return False

            # Extract CLIP features
            features = self.extract_clip_features(frames)

            # Note: No longer truncating or padding based on dataset duration
            # The dataset loader will handle slicing based on timeRange

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
        Uses two-pass approach to aggregate segments for videos split into multiple pieces.

        Args:
            dataset_path: Path to dataset JSON file
            video_dir: Directory containing video files
            max_videos: Maximum number of videos to process

        Returns:
            Dict containing processing statistics
        """
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        # First pass: Aggregate segments by youtube_id
        # Videos longer than 1800 seconds are split into multiple entries
        video_segments = {}  # youtube_id -> list of all segments for that video

        self.logger.info("First pass: Aggregating segments by video ID...")
        for video_info in dataset:
            youtube_id = video_info['youtube_id']

            # Use 'segments' (absolute timestamps) not 'segmentsOffset' (relative to timeRange)
            segments = video_info.get('segments', [])

            if youtube_id not in video_segments:
                video_segments[youtube_id] = []

            # Add all segments for this video
            video_segments[youtube_id].extend(segments)

        # Sort segments for each video (no merging needed as splits don't overlap)
        for youtube_id in video_segments:
            segments = video_segments[youtube_id]
            if segments:
                # Sort by start time for consistency
                segments.sort(key=lambda x: x[0])
                video_segments[youtube_id] = segments
                self.logger.info(
                    f"  {youtube_id}: {len(segments)} highlight segments")

        # Apply max_videos limit if specified
        if max_videos:
            video_ids = list(video_segments.keys())[:max_videos]
            video_segments = {k: video_segments[k] for k in video_ids}

        video_dir = Path(video_dir)
        total_videos = len(video_segments)
        successful_extractions = 0
        failed_extractions = 0

        self.logger.info(
            f"Second pass: Extracting features for {total_videos} unique videos...")

        for i, (youtube_id, segments) in enumerate(video_segments.items(), 1):
            video_file = video_dir / f"{youtube_id}.mp4"

            if not video_file.exists():
                self.logger.warning(f"Video file not found: {video_file}")
                failed_extractions += 1
                continue

            self.logger.info(
                f"Processing video {i}/{total_videos}: {youtube_id}")

            # Get segments for hint injection if enabled
            segments_for_hints = None
            if self.inject_hints and segments:
                segments_for_hints = segments
                # Calculate total highlight duration
                total_highlight_duration = sum(
                    end - start for start, end in segments)
                self.logger.info(f"  Will inject hints for {len(segments_for_hints)} segments, "
                                 f"total highlight duration: {total_highlight_duration:.1f}s")
                # Log first few segments for debugging
                if segments_for_hints:
                    preview = segments_for_hints[:3]
                    self.logger.debug(f"  First segments: {preview}")

            # Extract features for the entire video
            if self.extract_features_from_video(str(video_file), youtube_id, segments=segments_for_hints):
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
    parser.add_argument("--inject-hints", action="store_true",
                        help="Inject red dots into highlight frames (for debugging)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    extractor = VisualFeatureExtractorCLIP(
        args.output_dir, args.log_level, inject_hints=args.inject_hints)

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
