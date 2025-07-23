#!/usr/bin/env python3
"""
Improved video downloader using yt-dlp library with parallel processing and better bot detection avoidance.
"""

import json
import os
import time
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from progress_tracker import ProgressTracker, ErrorCategory

try:
    import yt_dlp
    HAS_YT_DLP = True
except ImportError:
    HAS_YT_DLP = False


@dataclass
class DownloadResult:
    """Result of a video download attempt"""
    youtube_id: str
    success: bool
    error_message: Optional[str] = None
    file_path: Optional[str] = None
    duration: Optional[float] = None
    retry_count: int = 0


class VideoDownloaderYTDLP:
    """Enhanced video downloader using yt-dlp library with parallel processing"""

    def __init__(self,
                 output_dir: str = "raw_videos",
                 log_level: str = "INFO",
                 max_workers: int = 3,
                 use_parallel: bool = True,
                 rate_limit: float = 1.0,
                 max_retries: int = 3,
                 cookies_file: Optional[str] = None):

        if not HAS_YT_DLP:
            raise ImportError(
                "yt-dlp not available. Install with: pip install yt-dlp")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.max_workers = max_workers
        self.use_parallel = use_parallel
        self.rate_limit = rate_limit  # seconds between downloads
        self.max_retries = max_retries
        self.cookies_file = cookies_file

        # Progress tracking
        self.progress_file = self.output_dir / "download_progress.json"
        self.downloaded_videos = self.load_progress()

        self.logger.info(
            f"VideoDownloader initialized with {max_workers} workers, parallel={use_parallel}")

        # Log cookies file status
        if self.cookies_file:
            if os.path.exists(self.cookies_file):
                self.logger.info(f"Using cookies file: {self.cookies_file}")
            else:
                self.logger.warning(
                    f"Cookies file not found: {self.cookies_file}")
        else:
            self.logger.info(
                "No cookies file specified - may encounter bot detection")

    def load_progress(self) -> Dict[str, bool]:
        """Load download progress from file."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {}

    def save_progress(self):
        """Save download progress to file."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.downloaded_videos, f, indent=2)

    def get_video_info(self, youtube_id: str) -> Optional[Dict[str, Any]]:
        """Get video information without downloading."""
        url = f"https://www.youtube.com/watch?v={youtube_id}"

        output_template = os.path.join(
            self.output_dir, f"{youtube_id}.%(ext)s")
        yt_dlp_opts = {
            "format": "bestvideo[height<=240]+bestaudio/best[height<=240]",
            "merge_output_format": "mp4",
            "outtmpl": output_template,
        }

        # Add cookies if available
        if self.cookies_file and os.path.exists(self.cookies_file):
            yt_dlp_opts['cookiefile'] = self.cookies_file

        try:
            with yt_dlp.YoutubeDL(yt_dlp_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info
        except Exception as e:
            self.logger.error(f"Failed to get info for {youtube_id}: {str(e)}")
            return None

    def check_video_exists(self, youtube_id: str) -> Optional[str]:
        """Check if video file already exists."""
        possible_extensions = ['.mp4', '.webm', '.mkv']
        for ext in possible_extensions:
            file_path = self.output_dir / f"{youtube_id}{ext}"
            if file_path.exists():
                return str(file_path)
        return None

    def download_single_video(self, youtube_id: str,
                              time_range: Optional[List[float]] = None,
                              retry_count: int = 0,
                              progress_tracker: Optional[ProgressTracker] = None) -> DownloadResult:
        """
        Download a single video using yt-dlp library.

        Args:
            youtube_id: YouTube video ID
            time_range: Optional [start, end] time range to download specific segment
            retry_count: Current retry attempt
            progress_tracker: Optional progress tracker for live updates

        Returns:
            DownloadResult with success status and details
        """
        # Check if file already exists
        existing_file = self.check_video_exists(youtube_id)
        if existing_file:
            self.logger.debug(
                f"Video {youtube_id} already exists at {existing_file}, skipping download...")
            if progress_tracker:
                progress_tracker.update_video_status(youtube_id, 'skipped')
            return DownloadResult(
                youtube_id=youtube_id,
                success=True,
                file_path=existing_file,
                retry_count=retry_count
            )
        
        # Check if previously marked as downloaded
        if youtube_id in self.downloaded_videos:
            # File was marked as downloaded but doesn't exist, remove from progress
            del self.downloaded_videos[youtube_id]
            self.save_progress()

        url = f"https://www.youtube.com/watch?v={youtube_id}"

        # Create download options for this specific video
        output_template = os.path.join(
            self.output_dir, f"{youtube_id}.%(ext)s")
        yt_dlp_opts = {
            "format": "bestvideo[height<=240]+bestaudio/best[height<=240]",
            "merge_output_format": "mp4",
            "outtmpl": output_template,
        }

        # Add cookies if available
        if self.cookies_file and os.path.exists(self.cookies_file):
            yt_dlp_opts['cookiefile'] = self.cookies_file

        # Add random delay to avoid rate limiting
        if retry_count > 0:
            delay = random.uniform(2, 8) * (retry_count + 1)
            self.logger.info(
                f"Retry {retry_count} for {youtube_id}, waiting {delay:.1f}s")
            time.sleep(delay)

        try:
            self.logger.info(
                f"Downloading {youtube_id}... (attempt {retry_count + 1})")
            
            if progress_tracker:
                progress_tracker.update_video_status(youtube_id, 'processing', retry_count=retry_count)

            with yt_dlp.YoutubeDL(yt_dlp_opts) as ydl:
                # Download the video
                ydl.download([url])

                # Check if file was created
                possible_paths = [
                    self.output_dir / f"{youtube_id}.mp4",
                    self.output_dir / f"{youtube_id}.webm",
                    self.output_dir / f"{youtube_id}.mkv"
                ]

                output_path = None
                for path in possible_paths:
                    if path.exists():
                        output_path = path
                        break

                if output_path:
                    # Get video info for duration
                    try:
                        info = self.get_video_info(youtube_id)
                        duration = info.get('duration') if info else None
                    except:
                        duration = None

                    self.downloaded_videos[youtube_id] = True
                    self.save_progress()

                    self.logger.info(f"Successfully downloaded {youtube_id}")
                    if progress_tracker:
                        progress_tracker.update_video_status(youtube_id, 'completed')
                    return DownloadResult(
                        youtube_id=youtube_id,
                        success=True,
                        file_path=str(output_path),
                        duration=duration,
                        retry_count=retry_count
                    )
                else:
                    error_msg = "Output file not found after download"
                    self.logger.error(
                        f"Download failed for {youtube_id}: {error_msg}")
                    if progress_tracker and retry_count >= self.max_retries:
                        progress_tracker.update_video_status(youtube_id, 'failed', error_msg)
                    return DownloadResult(
                        youtube_id=youtube_id,
                        success=False,
                        error_message=error_msg,
                        retry_count=retry_count
                    )

        except yt_dlp.DownloadError as e:
            error_msg = str(e)
            if "Sign in to confirm" in error_msg or "bot" in error_msg.lower():
                self.logger.warning(
                    f"Bot detection for {youtube_id}: {error_msg}")
                # For bot detection, we might want to increase delay
                time.sleep(random.uniform(5, 15))
            else:
                self.logger.error(
                    f"Download error for {youtube_id}: {error_msg}")

            if progress_tracker and retry_count >= self.max_retries:
                progress_tracker.update_video_status(youtube_id, 'failed', error_msg)
            return DownloadResult(
                youtube_id=youtube_id,
                success=False,
                error_message=error_msg,
                retry_count=retry_count
            )

        except Exception as e:
            error_msg = str(e)
            self.logger.error(
                f"Unexpected error downloading {youtube_id}: {error_msg}")
            if progress_tracker and retry_count >= self.max_retries:
                progress_tracker.update_video_status(youtube_id, 'failed', error_msg)
            return DownloadResult(
                youtube_id=youtube_id,
                success=False,
                error_message=error_msg,
                retry_count=retry_count
            )

    def download_video_with_retries(self, youtube_id: str,
                                    time_range: Optional[List[float]] = None,
                                    progress_tracker: Optional[ProgressTracker] = None) -> DownloadResult:
        """Download a video with retry logic."""
        last_result = None

        for attempt in range(self.max_retries + 1):
            result = self.download_single_video(
                youtube_id, time_range, attempt, progress_tracker)

            if result.success:
                return result

            last_result = result

            # Check if we should retry
            if attempt < self.max_retries:
                error_msg = result.error_message or ""
                
                # Check if error is non-retryable
                if progress_tracker:
                    error_category = progress_tracker.categorize_error(error_msg)
                    if not progress_tracker.is_retryable_error(error_category):
                        self.logger.info(
                            f"Not retrying {youtube_id} - non-retryable error: {error_category.value}")
                        break
                else:
                    # Fallback to old logic if no progress tracker
                    if any(skip_phrase in error_msg.lower() for skip_phrase in [
                        "private video", "deleted", "unavailable", "copyright", "terminated"
                    ]):
                        self.logger.info(
                            f"Not retrying {youtube_id} due to: {error_msg}")
                        break

                # Exponential backoff with jitter
                delay = random.uniform(2, 8) * (2 ** attempt)
                self.logger.info(f"Retrying {youtube_id} in {delay:.1f}s...")
                time.sleep(delay)

        return last_result

    def _download_worker(self, video_info: Tuple[str, Optional[List[float]]],
                         index: int, total: int,
                         progress_tracker: Optional[ProgressTracker] = None) -> DownloadResult:
        """Worker function for parallel downloads."""
        youtube_id, time_range = video_info

        # Don't log individual video processing when using progress tracker
        if not progress_tracker:
            self.logger.info(f"Processing video {index + 1}/{total}: {youtube_id}")

        # Rate limiting
        if index > 0:
            time.sleep(self.rate_limit + random.uniform(0, 1))

        return self.download_video_with_retries(youtube_id, time_range, progress_tracker)

    def download_from_dataset(self, dataset_path: str,
                              max_videos: Optional[int] = None,
                              use_progress_tracker: bool = True) -> Dict[str, Any]:
        """
        Download videos from a dataset JSON file with parallel processing.

        Args:
            dataset_path: Path to the dataset JSON file
            max_videos: Maximum number of videos to download (for testing)
            use_progress_tracker: Whether to use live progress tracking

        Returns:
            Dict containing download statistics
        """
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        if max_videos:
            dataset = dataset[:max_videos]

        # Prepare video list
        video_list = []
        for video_info in dataset:
            youtube_id = video_info['youtube_id']
            time_range = video_info.get('timeRange')
            video_list.append((youtube_id, time_range))

        self.logger.info(f"Starting download of {len(video_list)} videos...")
        
        # Initialize progress tracker
        progress_tracker = None
        if use_progress_tracker:
            progress_tracker = ProgressTracker(len(video_list), "Video Download")
            for youtube_id, _ in video_list:
                progress_tracker.add_video(youtube_id)

        start_time = time.time()
        results = []

        if self.use_parallel and len(video_list) > 1:
            if not progress_tracker:
                self.logger.info(
                    f"Using parallel processing with {self.max_workers} workers")

            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(video_list))) as executor:
                # Submit all download tasks
                future_to_video = {
                    executor.submit(self._download_worker, video_info, i, len(video_list), progress_tracker): video_info
                    for i, video_info in enumerate(video_list)
                }

                # Collect results as they complete
                for future in as_completed(future_to_video):
                    try:
                        result = future.result()
                        results.append(result)

                        if not progress_tracker:
                            if result.success:
                                self.logger.info(
                                    f"✓ Downloaded {result.youtube_id}")
                            else:
                                self.logger.warning(
                                    f"✗ Failed {result.youtube_id}: {result.error_message}")

                    except Exception as e:
                        video_info = future_to_video[future]
                        youtube_id = video_info[0]
                        error_msg = str(e)
                        self.logger.error(
                            f"Worker error for {youtube_id}: {error_msg}")
                        if progress_tracker:
                            progress_tracker.update_video_status(youtube_id, 'failed', error_msg)
                        results.append(DownloadResult(
                            youtube_id=youtube_id,
                            success=False,
                            error_message=error_msg
                        ))
        else:
            # Sequential processing
            if not progress_tracker:
                self.logger.info("Using sequential processing")
            for i, video_info in enumerate(video_list):
                result = self._download_worker(video_info, i, len(video_list), progress_tracker)
                results.append(result)

                if not progress_tracker:
                    if result.success:
                        self.logger.info(f"✓ Downloaded {result.youtube_id}")
                    else:
                        self.logger.warning(
                            f"✗ Failed {result.youtube_id}: {result.error_message}")

        # Finalize progress tracker
        if progress_tracker:
            progress_tracker.finalize()
        
        # Calculate statistics
        successful_downloads = sum(1 for r in results if r.success)
        failed_downloads = len(results) - successful_downloads
        skipped_downloads = sum(1 for r in results if r.success and r.file_path and "already exists" in str(r.error_message or ""))

        # Count retry statistics
        total_retries = sum(r.retry_count for r in results)
        bot_detection_errors = sum(1 for r in results
                                   if not r.success and r.error_message and
                                   ("sign in" in r.error_message.lower() or "bot" in r.error_message.lower()))

        processing_time = time.time() - start_time

        stats = {
            'total_videos': len(video_list),
            'successful_downloads': successful_downloads,
            'failed_downloads': failed_downloads,
            'skipped_downloads': progress_tracker.skipped_count if progress_tracker else skipped_downloads,
            'success_rate': successful_downloads / len(video_list) * 100 if video_list else 0,
            'total_retries': total_retries,
            'bot_detection_errors': bot_detection_errors,
            'processing_time': processing_time,
            'average_time_per_video': processing_time / len(video_list) if video_list else 0,
            'failed_videos': [r.youtube_id for r in results if not r.success]
        }

        if not progress_tracker:
            self.logger.info(f"Download complete: {successful_downloads}/{len(video_list)} successful "
                             f"({stats['success_rate']:.1f}%)")
            self.logger.info(f"Total processing time: {processing_time:.1f}s")
            if total_retries > 0:
                self.logger.info(f"Total retries: {total_retries}")
            if bot_detection_errors > 0:
                self.logger.warning(
                    f"Bot detection errors: {bot_detection_errors}")

        return stats

    def cleanup_failed_downloads(self):
        """Remove partially downloaded files."""
        patterns = ["*.part", "*.tmp", "*.temp"]
        cleaned_count = 0

        for pattern in patterns:
            for file_path in self.output_dir.glob(pattern):
                try:
                    file_path.unlink()
                    self.logger.info(f"Removed partial download: {file_path}")
                    cleaned_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to remove {file_path}: {e}")

        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} partial files")
        else:
            self.logger.info("No partial files to clean up")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Download videos for Repurpose dataset using yt-dlp library")
    parser.add_argument("--dataset", required=True,
                        help="Path to dataset JSON file")
    parser.add_argument("--output-dir", default="raw_videos",
                        help="Output directory for videos")
    parser.add_argument("--max-videos", type=int,
                        help="Maximum number of videos to download")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # Parallel processing options
    parser.add_argument("--parallel", action="store_true",
                        default=True, help="Enable parallel processing")
    parser.add_argument("--no-parallel", dest="parallel",
                        action="store_false", help="Disable parallel processing")
    parser.add_argument("--max-workers", type=int, default=3,
                        help="Maximum number of parallel workers")
    parser.add_argument("--rate-limit", type=float, default=1.0,
                        help="Minimum seconds between downloads")

    # Retry options
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Maximum number of retries per video")

    # Authentication options
    parser.add_argument(
        "--cookies", help="Path to cookies file for authentication")

    args = parser.parse_args()

    try:
        downloader = VideoDownloaderYTDLP(
            output_dir=args.output_dir,
            log_level=args.log_level,
            max_workers=args.max_workers,
            use_parallel=args.parallel,
            rate_limit=args.rate_limit,
            max_retries=args.max_retries,
            cookies_file=args.cookies
        )

        stats = downloader.download_from_dataset(args.dataset, args.max_videos)

        print(f"\n=== DOWNLOAD STATISTICS ===")
        print(f"Total videos: {stats['total_videos']}")
        print(f"Successful: {stats['successful_downloads']}")
        print(f"Failed: {stats['failed_downloads']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        print(f"Processing time: {stats['processing_time']:.1f}s")
        print(
            f"Average time per video: {stats['average_time_per_video']:.1f}s")

        if stats['total_retries'] > 0:
            print(f"Total retries: {stats['total_retries']}")

        if stats['bot_detection_errors'] > 0:
            print(f"Bot detection errors: {stats['bot_detection_errors']}")
            print("\nTo fix bot detection errors:")
            print("1. Export cookies from your browser:")
            print("   - Install browser extension: 'Get cookies.txt'")
            print("   - Visit youtube.com and export cookies")
            print("   - Use --cookies cookies.txt")
            print("2. Reduce parallel workers: --max-workers 1")
            print("3. Increase rate limiting: --rate-limit 3.0")

        if stats['failed_videos']:
            print(f"\nFailed videos ({len(stats['failed_videos'])}):")
            for vid in stats['failed_videos'][:10]:  # Show first 10
                print(f"  - {vid}")
            if len(stats['failed_videos']) > 10:
                print(f"  ... and {len(stats['failed_videos']) - 10} more")

    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
        try:
            downloader.cleanup_failed_downloads()
        except:
            pass
    except Exception as e:
        print(f"Error: {str(e)}")
        try:
            downloader.cleanup_failed_downloads()
        except:
            pass


if __name__ == "__main__":
    main()
