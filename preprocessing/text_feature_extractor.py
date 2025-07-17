import os
import json
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse
import subprocess
import tempfile
import re


class TextFeatureExtractor:
    def __init__(self, output_dir: str = "data/caption_features", log_level: str = "INFO"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create transcript directory
        self.transcript_dir = Path("data/transcripts")
        self.transcript_dir.mkdir(parents=True, exist_ok=True)

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
            import whisperx
            self.logger.info("whisperx library found")
        except ImportError:
            self.logger.warning(
                "whisperx library not found. Install with: pip install whisperx")
            self.logger.info("Will use whisper fallback for transcription")

        try:
            import whisper
            self.logger.info("whisper library found")
        except ImportError:
            self.logger.warning(
                "whisper library not found. Install with: pip install openai-whisper")

        try:
            from sentence_transformers import SentenceTransformer
            self.logger.info("sentence_transformers library found")
        except ImportError:
            self.logger.error(
                "sentence_transformers library not found. Install with: pip install sentence-transformers")
            raise ImportError("sentence_transformers library is required")

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
            "-ar", "16000",  # 16 kHz sample rate (required for Whisper)
            "-ac", "1",  # Mono
            audio_path,
            "-y"  # Overwrite output file
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg audio extraction failed: {result.stderr}")

        return audio_path
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get the duration of an audio file using ffprobe.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            float: Duration in seconds
        """
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", audio_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            self.logger.debug(f"Audio duration: {duration:.2f}s")
            return duration
        except (subprocess.CalledProcessError, ValueError) as e:
            self.logger.warning(f"Failed to get audio duration: {e}")
            # Fallback to a default minimum duration
            return 1.0

    def transcribe_with_whisperx(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Transcribe audio using WhisperX with word-level timestamps.

        Args:
            audio_path: Path to audio file

        Returns:
            List of transcription segments with timestamps
        """
        try:
            import whisperx

            device = "cuda" if self.check_gpu() else "cpu"

            # Load model
            model = whisperx.load_model("base", device)

            # Transcribe
            audio = whisperx.load_audio(audio_path)
            result = model.transcribe(audio)

            # Align whisper output
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"], device=device)
            result = whisperx.align(
                result["segments"], model_a, metadata, audio, device)

            return result["segments"]

        except Exception as e:
            self.logger.error(f"WhisperX transcription failed: {str(e)}")
            raise

    def transcribe_with_whisper(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Transcribe audio using OpenAI Whisper (fallback method).

        Args:
            audio_path: Path to audio file

        Returns:
            List of transcription segments with timestamps
        """
        try:
            import whisper

            model = whisper.load_model("base")
            result = model.transcribe(audio_path)

            return result["segments"]

        except Exception as e:
            self.logger.error(f"Whisper transcription failed: {str(e)}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\']', '', text)

        return text.strip()

    def save_transcript(self, youtube_id: str, segments: List[Dict[str, Any]]):
        """Save transcript segments to JSON file."""
        transcript_path = self.transcript_dir / f"{youtube_id}.json"
        with open(transcript_path, 'w') as f:
            json.dump(segments, f, indent=2)
        self.logger.debug(f"Saved transcript for {youtube_id}")

        # Also save human-readable text file
        self.save_text_transcript(youtube_id, segments)

    def save_text_transcript(self, youtube_id: str, segments: List[Dict[str, Any]]):
        """Save human-readable transcript to text file."""
        text_path = self.transcript_dir / f"{youtube_id}.txt"
        with open(text_path, 'w') as f:
            f.write(f"Transcript for {youtube_id}\n")
            f.write("=" * 50 + "\n\n")

            for segment in segments:
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                text = segment.get('text', '')

                # Format timestamp as [MM:SS - MM:SS]
                start_time = f"{int(start//60):02d}:{int(start%60):02d}"
                end_time = f"{int(end//60):02d}:{int(end%60):02d}"

                f.write(f"[{start_time} - {end_time}] {text}\n")

        self.logger.debug(f"Saved text transcript for {youtube_id}")

    def load_transcript(self, youtube_id: str) -> Optional[List[Dict[str, Any]]]:
        """Load transcript segments from JSON file if exists."""
        transcript_path = self.transcript_dir / f"{youtube_id}.json"
        if transcript_path.exists():
            with open(transcript_path, 'r') as f:
                segments = json.load(f)
            self.logger.info(f"Loaded existing transcript for {youtube_id}")
            return segments
        return None

    def extract_text_features(self, video_path: str, youtube_id: str, _video_duration: Optional[float] = None) -> bool:
        """
        Extract text features from video by transcribing speech and encoding with sentence transformers.
        Note: _video_duration parameter is ignored - features are extracted for the entire video.

        Args:
            video_path: Path to input video file
            youtube_id: YouTube video ID for naming output file
            _video_duration: IGNORED - kept for compatibility only

        Returns:
            bool: True if successful, False otherwise
        """
        output_path = self.output_dir / f"{youtube_id}.npy"
        
        # Check if features already exist and are marked as processed
        if youtube_id in self.processed_videos and output_path.exists():
            self.logger.info(
                f"Text features for {youtube_id} already extracted, skipping...")
            return True
        
        # If marked as processed but features don't exist, remove from processed list
        if youtube_id in self.processed_videos and not output_path.exists():
            self.logger.warning(
                f"Text features for {youtube_id} marked as processed but file missing, regenerating...")
            del self.processed_videos[youtube_id]
            self.save_progress()

        try:
            from sentence_transformers import SentenceTransformer

            # Load sentence transformer model
            # 384-dimensional embeddings
            model = SentenceTransformer('all-MiniLM-L6-v2')

            # Try to load existing transcript first
            segments = self.load_transcript(youtube_id)
            audio_duration_seconds = None

            if segments is None:
                # No existing transcript, need to transcribe
                self.logger.info(f"No existing transcript found for {youtube_id}, transcribing...")
                # Create temporary directory for audio extraction
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Extract audio from video
                    audio_path = self.extract_audio_from_video(
                        video_path, temp_dir)
                    
                    # Get actual audio duration using ffprobe
                    actual_duration = self.get_audio_duration(audio_path)

                    # Transcribe audio
                    try:
                        segments = self.transcribe_with_whisperx(audio_path)
                        self.logger.info(
                            f"Transcribed {youtube_id} using WhisperX")
                    except:
                        try:
                            segments = self.transcribe_with_whisper(audio_path)
                            self.logger.info(
                                f"Transcribed {youtube_id} using Whisper")
                        except:
                            self.logger.error(
                                f"Failed to transcribe {youtube_id}")
                            return False

                    # Save transcript for future use
                    self.save_transcript(youtube_id, segments)
                    
                    # Store actual duration for use after temp_dir cleanup
                    audio_duration_seconds = int(actual_duration)
            else:
                # Transcript exists, but we still need to get audio duration for full-length features
                self.logger.info(f"Using existing transcript for {youtube_id}, getting audio duration...")
                with tempfile.TemporaryDirectory() as temp_dir:
                    try:
                        audio_path = self.extract_audio_from_video(video_path, temp_dir)
                        actual_duration = self.get_audio_duration(audio_path)
                        audio_duration_seconds = int(actual_duration)
                        self.logger.debug(f"Got audio duration from existing transcript case: {audio_duration_seconds}s")
                    except Exception as e:
                        self.logger.warning(f"Failed to get audio duration for existing transcript: {e}")
                        # Fallback to transcription duration
                        if segments:
                            audio_duration_seconds = int(max([s.get('end', 0) for s in segments]) + 1)
                            self.logger.debug(f"Using transcription duration as fallback: {audio_duration_seconds}s")
                        else:
                            audio_duration_seconds = 1

                # Use the audio duration we determined above
                duration_seconds = audio_duration_seconds if audio_duration_seconds is not None else 1
                self.logger.debug(f"Using duration: {duration_seconds}s")

                # Process segments into 1-second intervals
                features = []

                # Group segments by second for the entire video duration
                for second in range(duration_seconds):
                    # Find segments that overlap with this second
                    overlapping_segments = []
                    for segment in segments:
                        start = segment.get('start', 0)
                        end = segment.get('end', 0)

                        if start <= second < end:
                            overlapping_segments.append(segment)

                    # Combine text from overlapping segments
                    if overlapping_segments:
                        combined_text = " ".join(
                            [self.clean_text(seg.get('text', '')) for seg in overlapping_segments])
                        combined_text = self.clean_text(combined_text)

                        if combined_text:
                            # Encode text to get 384-dimensional embedding
                            embedding = model.encode([combined_text])[0]
                            features.append(embedding)
                        else:
                            # Empty text - use zero vector
                            features.append(np.zeros(384))
                    else:
                        # No speech in this second - use zero vector
                        features.append(np.zeros(384))

                # Ensure we have at least some features
                if not features:
                    self.logger.warning(
                        f"No text features extracted for {youtube_id}")
                    # Create a single zero vector
                    features = [np.zeros(384)]

                features = np.array(features)

                # Save features
                np.save(output_path, features)

                self.processed_videos[youtube_id] = True
                self.save_progress()

                self.logger.info(
                    f"Successfully extracted text features for {youtube_id}, shape: {features.shape} (duration: {duration_seconds}s)")
                return True

        except Exception as e:
            self.logger.error(
                f"Text feature extraction failed for {youtube_id}: {str(e)}")
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
        Process all videos in a directory to extract text features.

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
            f"Starting text feature extraction for {total_videos} videos...")

        for i, video_file in enumerate(video_files, 1):
            youtube_id = video_file.stem

            self.logger.info(
                f"Processing video {i}/{total_videos}: {youtube_id}")

            if self.extract_text_features(str(video_file), youtube_id):
                successful_extractions += 1
            else:
                failed_extractions += 1

        stats = {
            'total_videos': total_videos,
            'successful_extractions': successful_extractions,
            'failed_extractions': failed_extractions,
            'success_rate': successful_extractions / total_videos * 100 if total_videos > 0 else 0
        }

        self.logger.info(f"Text feature extraction complete: {successful_extractions}/{total_videos} successful "
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
            f"Starting text feature extraction for {total_videos} videos from dataset...")

        for i, video_info in enumerate(dataset, 1):
            youtube_id = video_info['youtube_id']
            video_file = video_dir / f"{youtube_id}.mp4"

            if not video_file.exists():
                self.logger.warning(f"Video file not found: {video_file}")
                failed_extractions += 1
                continue

            self.logger.info(
                f"Processing video {i}/{total_videos}: {youtube_id}")

            # Extract features for the entire video (ignoring dataset timeRange)
            # The dataset loader will handle slicing based on timeRange at runtime
            if self.extract_text_features(str(video_file), youtube_id):
                successful_extractions += 1
            else:
                failed_extractions += 1

        stats = {
            'total_videos': total_videos,
            'successful_extractions': successful_extractions,
            'failed_extractions': failed_extractions,
            'success_rate': successful_extractions / total_videos * 100 if total_videos > 0 else 0
        }

        self.logger.info(f"Text feature extraction complete: {successful_extractions}/{total_videos} successful "
                         f"({stats['success_rate']:.1f}%)")

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract text features from videos")
    parser.add_argument("--video-dir", required=True,
                        help="Directory containing video files")
    parser.add_argument("--dataset", help="Path to dataset JSON file")
    parser.add_argument("--output-dir", default="data/caption_features",
                        help="Output directory for features")
    parser.add_argument("--max-videos", type=int,
                        help="Maximum number of videos to process")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    extractor = TextFeatureExtractor(args.output_dir, args.log_level)

    try:
        if args.dataset:
            stats = extractor.process_from_dataset(
                args.dataset, args.video_dir, args.max_videos)
        else:
            stats = extractor.process_video_directory(
                args.video_dir, args.max_videos)

        print(f"\nText Feature Extraction Statistics:")
        print(f"Total videos: {stats['total_videos']}")
        print(f"Successful: {stats['successful_extractions']}")
        print(f"Failed: {stats['failed_extractions']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")

    except KeyboardInterrupt:
        print("\nText feature extraction interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
