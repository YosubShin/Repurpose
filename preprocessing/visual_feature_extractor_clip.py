import json
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse
import torch
from PIL import Image, ImageDraw
import clip
import threading
import queue
import time
import signal
import sys
import traceback

try:
    import av

    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False


class VisualFeatureExtractorCLIP:
    def __init__(
        self,
        output_dir: str = "data/video_clip_features",
        log_level: str = "INFO",
        inject_hints: bool = False,
        use_black_white: bool = False,
        num_workers: int = 4,  # Reduced to prevent OOM
        batch_size: int = 32,  # Reduced to prevent OOM
        queue_size: int = 1000,  # Reduced to prevent OOM
    ):
        self.inject_hints = inject_hints
        self.use_black_white = use_black_white
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Parallel processing parameters
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.queue_size = min(queue_size, 1000)  # Limit queue size to prevent OOM

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s - %(levelname)s - %(message)s",
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
                "HINT INJECTION ENABLED: Red dots will be added to highlight frames"
            )
        if self.use_black_white:
            self.logger.info(
                "BLACK/WHITE MODE ENABLED: Using synthetic frames (white for highlights, black for non-highlights)"
            )

    def load_progress(self) -> Dict[str, bool]:
        """Load extraction progress from file."""
        if self.progress_file.exists():
            with open(self.progress_file, "r") as f:
                return json.load(f)
        return {}

    def save_progress(self):
        """Save extraction progress to file."""
        with open(self.progress_file, "w") as f:
            json.dump(self.processed_videos, f, indent=2)

    def create_black_white_frame(
        self, is_highlight: bool, width: int = 224, height: int = 224
    ) -> np.ndarray:
        """
        Create a black or white frame for testing.

        Args:
            is_highlight: True for white frame (highlight), False for black frame
            width: Frame width
            height: Frame height

        Returns:
            numpy array of shape (H, W, 3) with black or white pixels
        """
        if is_highlight:
            # White frame for highlights
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        else:
            # Black frame for non-highlights
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        return frame

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
            [
                (center_x - radius, center_y - radius),
                (center_x + radius, center_y + radius),
            ],
            fill="red",
            outline="darkred",
            width=2,
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

    def extract_frames_pyav(
        self,
        video_path: str,
        max_duration: Optional[float] = None,
        segments: Optional[List[List[float]]] = None,
    ) -> List[Tuple[float, np.ndarray]]:
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
            raise ImportError("PyAV is not available. Install with: pip install av")

        frames = []
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        video_stream.thread_type = "AUTO"

        # Get video duration
        if max_duration is None:
            duration = float(video_stream.duration * video_stream.time_base)
        else:
            duration = min(
                max_duration, float(video_stream.duration * video_stream.time_base)
            )

        self.logger.debug(
            f"Video duration: {duration:.2f}s, using PyAV timestamp seeking"
        )

        # Extract one frame per second using precise seeking
        for second in range(int(duration)):
            timestamp = float(second)

            try:
                # Seek to the exact timestamp
                container.seek(
                    int(timestamp / video_stream.time_base), stream=video_stream
                )

                # Get the next frame after seeking
                for frame in container.decode(video_stream):
                    frame_time = float(frame.pts * video_stream.time_base)

                    # Check if this frame is close enough to our target timestamp
                    if abs(frame_time - timestamp) < 0.5:  # Within 0.5 seconds
                        # Use black/white frames if enabled
                        if self.use_black_white:
                            is_highlight = segments and self.is_highlight_frame(
                                timestamp, segments
                            )
                            frame_rgb = self.create_black_white_frame(is_highlight)
                            if is_highlight:
                                self.logger.debug(
                                    f"Created WHITE frame at {timestamp}s (highlight)"
                                )
                            else:
                                self.logger.debug(
                                    f"Created BLACK frame at {timestamp}s (non-highlight)"
                                )
                        else:
                            # Convert to numpy array and resize to standard size
                            frame_rgb = frame.to_rgb().to_ndarray()
                            # Resize to CLIP's expected input size (224x224) to save memory
                            if frame_rgb.shape[:2] != (224, 224):
                                frame_pil = Image.fromarray(frame_rgb)
                                frame_pil = frame_pil.resize((224, 224), Image.LANCZOS)
                                frame_rgb = np.array(frame_pil)

                        # Optionally add red dot for highlighted frames
                        if (
                            self.inject_hints
                            and segments
                            and self.is_highlight_frame(timestamp, segments)
                        ):
                            frame_rgb = self.add_red_dot_to_frame(frame_rgb)
                            self.logger.debug(f"Added red dot to frame at {timestamp}s")

                        frames.append((timestamp, frame_rgb))
                        break

            except Exception as e:
                self.logger.warning(f"Failed to extract frame at {timestamp}s: {e}")
                # Add a zero frame as placeholder
                frames.append((timestamp, np.zeros((240, 320, 3), dtype=np.uint8)))

        container.close()
        return frames

    @staticmethod
    def _extract_frames_static(
        video_path: str,
        segments: Optional[List[List[float]]],
        inject_hints: bool,
        use_black_white: bool,
        logger,
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Static method for frame extraction to be used in multiprocessing.
        """
        if not PYAV_AVAILABLE:
            raise ImportError("PyAV is not available. Install with: pip install av")

        frames = []
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        video_stream.thread_type = "AUTO"

        # Get video duration
        duration = float(video_stream.duration * video_stream.time_base)

        logger.debug(f"Video duration: {duration:.2f}s, using PyAV timestamp seeking")

        # Extract one frame per second using precise seeking
        for second in range(int(duration)):
            timestamp = float(second)

            try:
                # Seek to the exact timestamp
                container.seek(
                    int(timestamp / video_stream.time_base), stream=video_stream
                )

                # Get the next frame after seeking
                for frame in container.decode(video_stream):
                    frame_time = float(frame.pts * video_stream.time_base)

                    # Check if this frame is close enough to our target timestamp
                    if abs(frame_time - timestamp) < 0.5:  # Within 0.5 seconds
                        # Use black/white frames if enabled
                        if use_black_white:
                            is_highlight = (
                                segments
                                and VisualFeatureExtractorCLIP._is_highlight_frame_static(
                                    timestamp, segments
                                )
                            )
                            frame_rgb = VisualFeatureExtractorCLIP._create_black_white_frame_static(
                                is_highlight
                            )
                        else:
                            # Convert to numpy array and resize to standard size
                            frame_rgb = frame.to_rgb().to_ndarray()
                            # Resize to CLIP's expected input size (224x224) to save memory
                            if frame_rgb.shape[:2] != (224, 224):
                                frame_pil = Image.fromarray(frame_rgb)
                                frame_pil = frame_pil.resize((224, 224), Image.LANCZOS)
                                frame_rgb = np.array(frame_pil)

                        # Optionally add red dot for highlighted frames
                        if (
                            inject_hints
                            and segments
                            and VisualFeatureExtractorCLIP._is_highlight_frame_static(
                                timestamp, segments
                            )
                        ):
                            frame_rgb = (
                                VisualFeatureExtractorCLIP._add_red_dot_to_frame_static(
                                    frame_rgb
                                )
                            )

                        frames.append((timestamp, frame_rgb))
                        break

            except Exception as e:
                logger.warning(f"Failed to extract frame at {timestamp}s: {e}")
                # Add a zero frame as placeholder
                frames.append((timestamp, np.zeros((224, 224, 3), dtype=np.uint8)))

        container.close()
        return frames

    @staticmethod
    def _create_black_white_frame_static(
        is_highlight: bool, width: int = 224, height: int = 224
    ) -> np.ndarray:
        """Static version of create_black_white_frame for multiprocessing"""
        if is_highlight:
            # White frame for highlights
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        else:
            # Black frame for non-highlights
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        return frame

    @staticmethod
    def _add_red_dot_to_frame_static(frame: np.ndarray) -> np.ndarray:
        """Static version of add_red_dot_to_frame for multiprocessing"""
        # Convert to PIL Image
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        # Get center coordinates
        width, height = img.size
        center_x, center_y = width // 2, height // 2

        # Draw a large red circle in the center (radius = 10 pixels)
        radius = 10
        draw.ellipse(
            [
                (center_x - radius, center_y - radius),
                (center_x + radius, center_y + radius),
            ],
            fill="red",
            outline="darkred",
            width=2,
        )

        # Convert back to numpy array
        return np.array(img)

    @staticmethod
    def _is_highlight_frame_static(
        timestamp: float, segments: List[List[float]]
    ) -> bool:
        """Static version of is_highlight_frame for multiprocessing"""
        for start, end in segments:
            if start <= timestamp < end:
                return True
        return False

    def extract_clip_features(self, frames: List[Tuple[float, Any]]) -> np.ndarray:
        """
        Extract CLIP features from frames using proper batching.

        Args:
            frames: List of (timestamp, frame) tuples where frame can be numpy array or PIL Image

        Returns:
            numpy array of shape (num_frames, 512) containing CLIP features
        """
        features = []

        with torch.no_grad():
            # Process frames in batches for better GPU utilization
            for i in range(0, len(frames), self.batch_size):
                batch_frames = frames[i : i + self.batch_size]
                actual_batch_size = len(batch_frames)

                self.logger.debug(
                    f"Processing batch {i//self.batch_size + 1}: {actual_batch_size} frames"
                )

                # Prepare batch of images
                images = []
                for timestamp, frame in batch_frames:
                    # Convert numpy array to PIL Image if needed
                    if isinstance(frame, np.ndarray):
                        frame = Image.fromarray(frame)
                    images.append(frame)

                # Batch preprocess and inference
                image_inputs = torch.stack([self.preprocess(img) for img in images]).to(
                    self.device
                )
                image_features = self.model.encode_image(image_inputs)

                # Normalize features
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                # Add to results
                features.extend(image_features.cpu().numpy())

        return np.array(features)

    def extract_black_white_features(
        self,
        video_path: str,
        youtube_id: str,
        segments: Optional[List[List[float]]] = None,
    ) -> bool:
        """
        Extract features using black/white frames efficiently.
        Computes CLIP embeddings once for black and white, then reuses them.

        Args:
            video_path: Path to video (used to get duration)
            youtube_id: YouTube video ID for naming output file
            segments: List of [start, end] pairs for highlight segments

        Returns:
            bool: True if successful
        """
        if youtube_id in self.processed_videos:
            self.logger.info(
                f"Features for {youtube_id} already extracted, skipping..."
            )
            return True

        output_path = self.output_dir / f"{youtube_id}.npy"

        try:
            self.logger.info(f"Extracting BLACK/WHITE features for {youtube_id}...")

            # Get video duration
            container = av.open(video_path)
            video_stream = container.streams.video[0]
            duration = float(video_stream.duration * video_stream.time_base)
            container.close()

            num_frames = int(duration)
            self.logger.info(
                f"  Video duration: {duration:.1f}s, creating {num_frames} feature vectors"
            )

            # Compute CLIP embeddings once for black and white frames
            black_frame = self.create_black_white_frame(is_highlight=False)
            white_frame = self.create_black_white_frame(is_highlight=True)

            with torch.no_grad():
                # Process black frame
                black_img = Image.fromarray(black_frame)
                black_input = self.preprocess(black_img).unsqueeze(0).to(self.device)
                black_features = self.model.encode_image(black_input)
                black_features = black_features / black_features.norm(
                    dim=-1, keepdim=True
                )
                black_vec = black_features.cpu().numpy().squeeze()

                # Process white frame
                white_img = Image.fromarray(white_frame)
                white_input = self.preprocess(white_img).unsqueeze(0).to(self.device)
                white_features = self.model.encode_image(white_input)
                white_features = white_features / white_features.norm(
                    dim=-1, keepdim=True
                )
                white_vec = white_features.cpu().numpy().squeeze()

            self.logger.info("  Computed CLIP embeddings for black and white frames")

            # Create feature array by checking each timestamp
            features = []
            highlight_count = 0

            for second in range(num_frames):
                timestamp = float(second)
                is_highlight = segments and self.is_highlight_frame(timestamp, segments)

                if is_highlight:
                    features.append(white_vec)
                    highlight_count += 1
                else:
                    features.append(black_vec)

            features = np.array(features)

            self.logger.info(
                f"  Created {len(features)} vectors: {highlight_count} white (highlight), {len(features)-highlight_count} black"
            )

            # Save features
            np.save(output_path, features)

            self.processed_videos[youtube_id] = True
            self.save_progress()

            self.logger.info(
                f"Successfully extracted features for {youtube_id}, shape: {features.shape}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Feature extraction failed for {youtube_id}: {str(e)}")
            return False

    def extract_features_from_video(
        self,
        video_path: str,
        youtube_id: str,
        video_duration: Optional[float] = None,
        segments: Optional[List[List[float]]] = None,
    ) -> bool:
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
        # Use optimized black/white extraction if enabled
        if self.use_black_white:
            return self.extract_black_white_features(video_path, youtube_id, segments)

        if youtube_id in self.processed_videos:
            self.logger.info(
                f"Features for {youtube_id} already extracted, skipping..."
            )
            return True

        output_path = self.output_dir / f"{youtube_id}.npy"

        try:
            self.logger.info(f"Extracting visual features for {youtube_id}...")
            if self.inject_hints and segments:
                self.logger.info(
                    f"  Hint injection enabled: {len(segments)} highlight segments"
                )

            # Use PyAV only (no FFmpeg fallback)
            frames = None

            if not PYAV_AVAILABLE:
                self.logger.error(
                    "PyAV is required but not installed. Install with: pip install av"
                )
                return False

            try:
                frames = self.extract_frames_pyav(
                    video_path, max_duration=None, segments=segments
                )
                self.logger.debug(f"Extracted {len(frames)} frames using PyAV")
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
                f"Successfully extracted features for {youtube_id}, shape: {features.shape}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Feature extraction failed for {youtube_id}: {str(e)}")
            return False

    def extract_features_producer_consumer(
        self,
        video_segments: Dict[str, List[List[float]]],
        video_dir: str,
        max_videos: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Extract features using producer-consumer pattern.
        CPU threads decode videos and preprocess frames.
        Single GPU thread handles CLIP inference.
        """
        # Apply max_videos limit if specified
        if max_videos:
            video_ids = list(video_segments.keys())[:max_videos]
            video_segments = {k: video_segments[k] for k in video_ids}

        video_dir = Path(video_dir)
        total_videos = len(video_segments)

        self.logger.info(
            f"Starting producer-consumer extraction for {total_videos} videos..."
        )
        self.logger.info(
            f"Using {self.num_workers} CPU threads for decoding, 1 GPU thread for CLIP"
        )

        # Queues for producer-consumer with memory-conscious sizing
        frame_queue = queue.Queue(
            maxsize=min(50, self.num_workers * 2)
        )  # Limit frame queue to prevent OOM
        result_queue = queue.Queue(maxsize=100)  # (video_id, success) tuples
        video_queue = queue.Queue(maxsize=self.queue_size)  # Videos to process

        # Add videos to queue
        for youtube_id, segments in video_segments.items():
            # Try different video extensions
            video_file = video_dir / f"{youtube_id}.mp4"
            if not video_file.exists():
                video_file = video_dir / f"{youtube_id}.mkv"
            if not video_file.exists():
                video_file = video_dir / f"{youtube_id}.webm"

            if not video_file.exists():
                self.logger.warning(f"Video file not found for {youtube_id}")
                continue

            # Skip if already processed
            output_path = self.output_dir / f"{youtube_id}.npy"
            if output_path.exists():
                self.logger.info(
                    f"Features for {youtube_id} already exist, skipping..."
                )
                continue

            segments_for_hints = segments if self.inject_hints else None
            video_queue.put((str(video_file), youtube_id, segments_for_hints))

        videos_to_process = video_queue.qsize()
        self.logger.info(f"Added {videos_to_process} videos to processing queue")

        # Performance monitoring
        start_time = time.time()
        successful_extractions = 0
        failed_extractions = 0

        # Process monitoring - import here to avoid issues if not available
        try:
            import psutil

            main_process = psutil.Process()
            initial_cpu_count = psutil.cpu_count()
            initial_memory = main_process.memory_info().rss / 1024 / 1024  # MB
            self.logger.info(
                f"Process monitoring initialized - CPU cores: {initial_cpu_count}, Initial memory: {initial_memory:.1f}MB"
            )
        except ImportError:
            psutil = None
            self.logger.warning("psutil not available, process monitoring disabled")

        # Signal handler for graceful shutdown and crash detection
        def signal_handler(signum, frame):
            signal_name = (
                signal.Signals(signum).name
                if hasattr(signal, "Signals")
                else str(signum)
            )
            self.logger.error(
                f"Received signal {signum} ({signal_name}), shutting down gracefully..."
            )
            # Signal all threads to stop
            for _ in range(self.num_workers):
                try:
                    video_queue.put(None, timeout=1)
                except:
                    pass
            try:
                frame_queue.put(None, timeout=1)
            except:
                pass
            self.logger.error("Emergency shutdown complete")
            sys.exit(1)

        # Register signal handlers for crash detection
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        try:
            # Start CPU producer threads (decode frames)
            producer_threads = []
            for i in range(self.num_workers):
                thread = threading.Thread(
                    target=self._frame_producer_worker,
                    args=(video_queue, frame_queue, i),
                    daemon=True,
                )
                thread.start()
                producer_threads.append(thread)
                self.logger.info(f"Started CPU producer thread {i}")

            # Start single GPU consumer thread (CLIP inference)
            consumer_thread = threading.Thread(
                target=self._clip_consumer_worker,
                args=(frame_queue, result_queue),
                daemon=True,
            )
            consumer_thread.start()
            self.logger.info("Started GPU consumer thread")

            # Wait for all videos to be processed
            processed_videos = 0
            last_progress_time = time.time()
            last_resource_log = time.time()

            while processed_videos < videos_to_process:
                try:
                    video_id, success = result_queue.get(timeout=60)  # Reduced timeout
                    processed_videos += 1
                    current_time = time.time()

                    if success:
                        successful_extractions += 1
                        self.logger.info(
                            f"Completed {video_id} ({processed_videos}/{videos_to_process})"
                        )
                    else:
                        failed_extractions += 1
                        self.logger.warning(
                            f"Failed {video_id} ({processed_videos}/{videos_to_process})"
                        )

                    last_progress_time = current_time

                    # Log resource usage every 30 seconds
                    if psutil and (current_time - last_resource_log) > 30:
                        try:
                            cpu_percent = main_process.cpu_percent()
                            memory_info = main_process.memory_info()
                            memory_mb = memory_info.rss / 1024 / 1024

                            # GPU monitoring if available
                            gpu_info = ""
                            if torch.cuda.is_available():
                                gpu_memory_gb = torch.cuda.memory_allocated() / 1024**3
                                try:
                                    # Try to get GPU utilization, fallback gracefully if not available
                                    if hasattr(torch.cuda, "utilization"):
                                        gpu_utilization = torch.cuda.utilization()
                                        gpu_info = f", GPU: {gpu_memory_gb:.1f}GB, Util: {gpu_utilization}%"
                                    else:
                                        gpu_info = f", GPU: {gpu_memory_gb:.1f}GB"
                                except Exception:
                                    # pynvml or other GPU monitoring not available
                                    gpu_info = (
                                        f", GPU: {gpu_memory_gb:.1f}GB (util: N/A)"
                                    )

                            # Check thread health
                            alive_producers = sum(
                                1 for t in producer_threads if t.is_alive()
                            )
                            consumer_alive = consumer_thread.is_alive()

                            self.logger.info(
                                f"Resources - CPU: {cpu_percent:.1f}%, Memory: {memory_mb:.1f}MB{gpu_info}, "
                                f"Threads alive: {alive_producers}/{self.num_workers} producers, consumer: {consumer_alive}, "
                                f"Queues: video={video_queue.qsize()}, frame={frame_queue.qsize()}, result={result_queue.qsize()}"
                            )

                            # Check for dead threads
                            if alive_producers == 0:
                                self.logger.error("All producer threads died!")
                            if not consumer_alive:
                                self.logger.error("Consumer thread died!")

                            last_resource_log = current_time

                        except Exception as e:
                            self.logger.warning(f"Error collecting resource stats: {e}")

                except queue.Empty:
                    current_time = time.time()
                    time_since_progress = current_time - last_progress_time

                    self.logger.warning(
                        f"No results for {time_since_progress:.1f}s, checking system health..."
                    )

                    # Log system health when timing out
                    if psutil:
                        try:
                            cpu_percent = main_process.cpu_percent()
                            memory_info = main_process.memory_info()
                            memory_mb = memory_info.rss / 1024 / 1024

                            # Check thread health
                            alive_producers = sum(
                                1 for t in producer_threads if t.is_alive()
                            )
                            consumer_alive = consumer_thread.is_alive()

                            self.logger.error(
                                f"TIMEOUT DIAGNOSIS - CPU: {cpu_percent:.1f}%, Memory: {memory_mb:.1f}MB, "
                                f"Threads alive: {alive_producers}/{self.num_workers} producers, consumer: {consumer_alive}, "
                                f"Queues: video={video_queue.qsize()}, frame={frame_queue.qsize()}, result={result_queue.qsize()}"
                            )

                            # Check if we're stuck
                            if alive_producers == 0 and consumer_alive:
                                self.logger.error(
                                    "Producer threads died, consumer waiting for work"
                                )
                            elif alive_producers > 0 and not consumer_alive:
                                self.logger.error(
                                    "Consumer thread died, producers may be blocked"
                                )
                            elif alive_producers == 0 and not consumer_alive:
                                self.logger.error(
                                    "All threads died - process may have crashed"
                                )
                                break

                        except Exception as e:
                            self.logger.error(f"Error during timeout diagnosis: {e}")

                    # If no progress for too long, break
                    if time_since_progress > 300:  # 5 minutes
                        self.logger.error("No progress for 5 minutes, terminating...")
                        break

            # Signal shutdown
            for _ in range(self.num_workers):
                video_queue.put(None)
            frame_queue.put(None)

            # Wait for threads to finish
            for i, thread in enumerate(producer_threads):
                thread.join(timeout=10)
                self.logger.info(f"CPU thread {i} finished")

            consumer_thread.join(timeout=10)
            self.logger.info("GPU thread finished")

            # Save progress
            self.save_progress()

            # Performance metrics
            end_time = time.time()
            total_time = end_time - start_time
            videos_per_sec = (
                successful_extractions / total_time if total_time > 0 else 0
            )

            self.logger.info(f"Performance metrics:")
            self.logger.info(f"  Total time: {total_time:.1f}s")
            self.logger.info(f"  Videos/sec: {videos_per_sec:.2f}")

            stats = {
                "total_videos": videos_to_process,
                "successful_extractions": successful_extractions,
                "failed_extractions": failed_extractions,
                "success_rate": (
                    successful_extractions / videos_to_process * 100
                    if videos_to_process > 0
                    else 0
                ),
            }

            self.logger.info(
                f"Producer-consumer extraction complete: {successful_extractions}/{videos_to_process} successful "
                f"({stats['success_rate']:.1f}%)"
            )

            return stats

        except Exception as e:
            self.logger.error(f"Critical error in producer-consumer extraction: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")

            # Try to capture final system state for debugging
            if psutil:
                try:
                    final_memory = main_process.memory_info().rss / 1024 / 1024
                    self.logger.error(f"Final memory usage: {final_memory:.1f}MB")
                    alive_producers = sum(1 for t in producer_threads if t.is_alive())
                    consumer_alive = consumer_thread.is_alive()
                    self.logger.error(
                        f"Final thread state: {alive_producers}/{self.num_workers} producers alive, consumer: {consumer_alive}"
                    )
                except Exception as state_e:
                    self.logger.error(f"Could not capture final state: {state_e}")

            # Try to shutdown gracefully
            try:
                for _ in range(self.num_workers):
                    video_queue.put(None)
                frame_queue.put(None)
            except:
                pass

            raise

    def _frame_producer_worker(
        self, video_queue: queue.Queue, frame_queue: queue.Queue, worker_id: int
    ):
        """CPU worker that decodes videos and extracts frames"""
        import traceback
        import psutil
        import os

        pid = os.getpid()
        process = psutil.Process(pid)

        self.logger.info(f"Frame producer {worker_id} started (PID: {pid})")
        processed = 0

        try:
            while True:
                try:
                    # Log memory usage periodically and check for OOM risk
                    if processed % 5 == 0:
                        mem_info = process.memory_info()
                        memory_mb = mem_info.rss / 1024 / 1024
                        self.logger.info(
                            f"Producer {worker_id} memory: RSS={memory_mb:.1f}MB, "
                            f"Queue sizes: video={video_queue.qsize()}, frame={frame_queue.qsize()}"
                        )

                        # Check for excessive memory usage (potential OOM risk)
                        if memory_mb > 8000:  # 8GB threshold
                            self.logger.warning(
                                f"Producer {worker_id} high memory usage: {memory_mb:.1f}MB - potential OOM risk"
                            )
                            # Force garbage collection
                            import gc

                            gc.collect()

                    work_item = video_queue.get(timeout=5)
                    if work_item is None:  # Shutdown signal
                        self.logger.info(
                            f"Producer {worker_id} received shutdown signal"
                        )
                        break

                    video_path, video_id, segments = work_item
                    self.logger.info(
                        f"Producer {worker_id} decoding {video_id} from {video_path}"
                    )

                    # Extract frames using existing method
                    frames = self._extract_frames_static(
                        video_path,
                        segments,
                        self.inject_hints,
                        self.use_black_white,
                        self.logger,
                    )

                    if frames:
                        # Limit frames per video to prevent memory bloat
                        max_frames = 300  # Max 5 minutes at 1fps
                        if len(frames) > max_frames:
                            self.logger.warning(
                                f"Video {video_id} has {len(frames)} frames, limiting to {max_frames} to prevent OOM"
                            )
                            frames = frames[:max_frames]

                        frame_queue.put((video_id, frames))
                        processed += 1
                        self.logger.debug(
                            f"Producer {worker_id} queued {len(frames)} frames for {video_id}"
                        )
                    else:
                        frame_queue.put((video_id, None))  # Signal failure
                        self.logger.warning(
                            f"Producer {worker_id} failed to extract frames for {video_id}"
                        )

                    video_queue.task_done()

                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(
                        f"Producer {worker_id} error processing {video_id}: {e}"
                    )
                    self.logger.error(
                        f"Producer {worker_id} traceback: {traceback.format_exc()}"
                    )
                    frame_queue.put(
                        (video_id if "video_id" in locals() else "unknown", None)
                    )
                    continue

        finally:
            self.logger.info(
                f"Frame producer {worker_id} finished, processed {processed} videos"
            )

    def _clip_consumer_worker(
        self, frame_queue: queue.Queue, result_queue: queue.Queue
    ):
        """GPU worker that processes frames through CLIP"""
        import traceback
        import psutil
        import os

        pid = os.getpid()
        process = psutil.Process(pid)

        self.logger.info(f"CLIP consumer started (PID: {pid})")
        processed = 0

        # Ensure model is in eval mode and use inference context
        self.model.eval()

        try:
            with torch.inference_mode():
                while True:
                    try:
                        # Log GPU status periodically and check for OOM risk
                        if processed % 3 == 0:
                            mem_info = process.memory_info()
                            cpu_memory_mb = mem_info.rss / 1024 / 1024
                            gpu_mem = (
                                torch.cuda.memory_allocated() / 1024**3
                                if torch.cuda.is_available()
                                else 0
                            )
                            self.logger.info(
                                f"GPU consumer memory: CPU={cpu_memory_mb:.1f}MB, "
                                f"GPU={gpu_mem:.1f}GB, Queue size: {frame_queue.qsize()}"
                            )

                            # Check for excessive memory usage
                            if cpu_memory_mb > 12000:  # 12GB threshold for GPU process
                                self.logger.warning(
                                    f"GPU consumer high CPU memory: {cpu_memory_mb:.1f}MB - potential OOM risk"
                                )
                                import gc

                                gc.collect()

                            if (
                                torch.cuda.is_available() and gpu_mem > 10
                            ):  # 10GB GPU memory threshold
                                self.logger.warning(
                                    f"GPU consumer high GPU memory: {gpu_mem:.1f}GB - clearing cache"
                                )
                                torch.cuda.empty_cache()

                        self.logger.debug(
                            f"GPU consumer waiting for work... (processed: {processed})"
                        )
                        frame_item = frame_queue.get(timeout=60)

                        if frame_item is None:  # Shutdown signal
                            self.logger.info("GPU consumer received shutdown signal")
                            break

                        video_id, frames = frame_item

                        if frames is None:
                            # Frame extraction failed
                            self.logger.warning(
                                f"GPU consumer received None frames for {video_id}"
                            )
                            result_queue.put((video_id, False))
                            continue

                        self.logger.info(
                            f"GPU processing {len(frames)} frames for {video_id} (batch_size={self.batch_size})"
                        )

                        # Process frames through CLIP in batches
                        features = self.extract_clip_features(frames)

                        # Save features
                        output_path = self.output_dir / f"{video_id}.npy"
                        np.save(output_path, features)

                        # Mark as processed
                        self.processed_videos[video_id] = True

                        result_queue.put((video_id, True))
                        processed += 1

                        self.logger.debug(
                            f"GPU saved features for {video_id}, shape: {features.shape}"
                        )

                        # Explicit cleanup to prevent memory accumulation
                        del features, frames
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    except queue.Empty:
                        continue
                    except Exception as e:
                        self.logger.error(
                            f"GPU consumer error processing {video_id}: {e}"
                        )
                        self.logger.error(
                            f"GPU consumer traceback: {traceback.format_exc()}"
                        )

                        # Log GPU/memory state when error occurs
                        try:
                            mem_info = process.memory_info()
                            gpu_mem = (
                                torch.cuda.memory_allocated() / 1024**3
                                if torch.cuda.is_available()
                                else 0
                            )
                            self.logger.error(
                                f"Error state - CPU mem: {mem_info.rss/1024/1024:.1f}MB, "
                                f"GPU mem: {gpu_mem:.1f}GB"
                            )
                        except Exception as mem_e:
                            self.logger.error(f"Could not capture error state: {mem_e}")

                        result_queue.put((video_id, False))
                        continue

        finally:
            self.logger.info(f"CLIP consumer finished, processed {processed} videos")

    def process_from_multiple_datasets(
        self,
        dataset_paths: List[str],
        video_dir: str,
        max_videos: Optional[int] = None,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Process videos from multiple dataset JSON files (train/val/test).
        Merges segments from all splits to avoid data leakage with hint injection.

        Args:
            dataset_paths: List of paths to dataset JSON files
            video_dir: Directory containing video files
            max_videos: Maximum number of videos to process
            start_index: Start index for video processing (for parallel runs)
            end_index: End index for video processing (for parallel runs)

        Returns:
            Dict containing processing statistics
        """
        # First, aggregate all segments from all dataset files
        video_segments = {}  # youtube_id -> list of all segments
        split_membership = {}  # youtube_id -> list of splits it appears in

        for dataset_path in dataset_paths:
            if not Path(dataset_path).exists():
                self.logger.warning(
                    f"Dataset file not found: {dataset_path}, skipping..."
                )
                continue

            split_name = Path(dataset_path).stem  # e.g., 'train', 'val', 'test'
            self.logger.info(f"Loading segments from {split_name} split...")

            with open(dataset_path, "r") as f:
                dataset = json.load(f)

            for video_info in dataset:
                youtube_id = video_info["youtube_id"]

                # Track which splits this video appears in
                if youtube_id not in split_membership:
                    split_membership[youtube_id] = []
                split_membership[youtube_id].append(split_name)

                # Use 'segments' (absolute timestamps) not 'segmentsOffset' (relative)
                segments = video_info.get("segments", [])

                if youtube_id not in video_segments:
                    video_segments[youtube_id] = []

                # Add all segments for this video
                video_segments[youtube_id].extend(segments)

        # Remove duplicates and sort segments for each video
        for youtube_id in video_segments:
            segments = video_segments[youtube_id]
            if segments:
                # Remove duplicate segments (same start and end)
                unique_segments = []
                seen = set()
                for seg in segments:
                    seg_tuple = tuple(seg)
                    if seg_tuple not in seen:
                        seen.add(seg_tuple)
                        unique_segments.append(seg)

                # Sort by start time
                unique_segments.sort(key=lambda x: x[0])
                video_segments[youtube_id] = unique_segments

                # Log videos that appear in multiple splits
                if len(split_membership[youtube_id]) > 1:
                    self.logger.info(
                        f"  {youtube_id}: appears in {split_membership[youtube_id]}, "
                        f"merged to {len(unique_segments)} unique segments"
                    )

        # Apply start/end index slicing if specified
        if start_index is not None or end_index is not None:
            video_ids = sorted(video_segments.keys())  # Sort for consistent ordering
            total_videos = len(video_ids)

            # Set defaults
            start_idx = start_index if start_index is not None else 0
            end_idx = end_index if end_index is not None else total_videos

            # Validate indices
            start_idx = max(0, min(start_idx, total_videos))
            end_idx = max(start_idx, min(end_idx, total_videos))

            # Slice the video list
            selected_video_ids = video_ids[start_idx:end_idx]
            video_segments = {vid: video_segments[vid] for vid in selected_video_ids}

            self.logger.info(
                f"Processing subset: videos {start_idx} to {end_idx} (total: {len(selected_video_ids)} out of {total_videos})"
            )

        # Now process videos with merged segments using producer-consumer pattern
        return self.extract_features_producer_consumer(
            video_segments, video_dir, max_videos
        )


def main():
    parser = argparse.ArgumentParser(
        description="Extract visual features from videos using CLIP"
    )
    parser.add_argument(
        "--video-dir", required=True, help="Directory containing video files"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset JSON files (multiple files will merge segments across splits)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/video_clip_features",
        help="Output directory for features",
    )
    parser.add_argument(
        "--max-videos", type=int, help="Maximum number of videos to process"
    )
    parser.add_argument(
        "--inject-hints",
        action="store_true",
        help="Inject red dots into highlight frames (for debugging)",
    )
    parser.add_argument(
        "--use-black-white",
        action="store_true",
        help="Use black/white synthetic frames instead of actual video (white=highlight, black=non-highlight)",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--num-workers", type=int, default=8, help="Number of video decoding workers"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for CLIP inference"
    )
    parser.add_argument(
        "--queue-size", type=int, default=100, help="Queue size for task coordination"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Start index for video processing (for parallel runs)",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="End index for video processing (for parallel runs)",
    )

    args = parser.parse_args()

    extractor = VisualFeatureExtractorCLIP(
        args.output_dir,
        args.log_level,
        inject_hints=args.inject_hints,
        use_black_white=args.use_black_white,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        queue_size=args.queue_size,
    )

    try:
        # Process datasets with merged segments
        print(f"Processing {len(args.datasets)} dataset files with merged segments...")
        if args.start_index is not None or args.end_index is not None:
            print(
                f"Processing videos from index {args.start_index} to {args.end_index}"
            )
        if args.inject_hints:
            print(
                "WARNING: Hint injection enabled - segments will be merged across all splits"
            )
            print("         to prevent data leakage!")
        stats = extractor.process_from_multiple_datasets(
            args.datasets,
            args.video_dir,
            args.max_videos,
            start_index=args.start_index,
            end_index=args.end_index,
        )

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
