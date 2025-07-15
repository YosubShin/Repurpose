import os
import json
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

class VideoDownloader:
    def __init__(self, output_dir: str = "raw_videos", log_level: str = "INFO"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create progress file to track downloaded videos
        self.progress_file = self.output_dir / "download_progress.json"
        self.downloaded_videos = self.load_progress()
    
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
    
    def download_video(self, youtube_id: str, time_range: Optional[List[float]] = None) -> bool:
        """
        Download a single video using yt-dlp.
        
        Args:
            youtube_id: YouTube video ID
            time_range: Optional [start, end] time range to download specific segment
            
        Returns:
            bool: True if successful, False otherwise
        """
        if youtube_id in self.downloaded_videos:
            self.logger.info(f"Video {youtube_id} already downloaded, skipping...")
            return True
            
        output_path = self.output_dir / f"{youtube_id}.mp4"
        url = f"https://www.youtube.com/watch?v={youtube_id}"
        
        # Build yt-dlp command
        cmd = [
            "yt-dlp",
            "--format", "best[ext=mp4]",
            "--output", str(output_path),
            url
        ]
        
        # Add time range if specified
        if time_range:
            start_time, end_time = time_range
            cmd.extend([
                "--download-sections", f"*{start_time}-{end_time}"
            ])
        
        try:
            self.logger.info(f"Downloading {youtube_id}...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if output_path.exists():
                self.downloaded_videos[youtube_id] = True
                self.save_progress()
                self.logger.info(f"Successfully downloaded {youtube_id}")
                return True
            else:
                self.logger.error(f"Download failed for {youtube_id}: Output file not found")
                return False
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Download failed for {youtube_id}: {e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error downloading {youtube_id}: {str(e)}")
            return False
    
    def download_from_dataset(self, dataset_path: str, max_videos: Optional[int] = None) -> Dict[str, Any]:
        """
        Download videos from a dataset JSON file.
        
        Args:
            dataset_path: Path to the dataset JSON file
            max_videos: Maximum number of videos to download (for testing)
            
        Returns:
            Dict containing download statistics
        """
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        total_videos = len(dataset)
        if max_videos:
            dataset = dataset[:max_videos]
            
        successful_downloads = 0
        failed_downloads = 0
        
        self.logger.info(f"Starting download of {len(dataset)} videos...")
        
        for i, video_info in enumerate(dataset, 1):
            youtube_id = video_info['youtube_id']
            time_range = video_info.get('timeRange')
            
            self.logger.info(f"Processing video {i}/{len(dataset)}: {youtube_id}")
            
            if self.download_video(youtube_id, time_range):
                successful_downloads += 1
            else:
                failed_downloads += 1
        
        stats = {
            'total_videos': len(dataset),
            'successful_downloads': successful_downloads,
            'failed_downloads': failed_downloads,
            'success_rate': successful_downloads / len(dataset) * 100
        }
        
        self.logger.info(f"Download complete: {successful_downloads}/{len(dataset)} successful "
                        f"({stats['success_rate']:.1f}%)")
        
        return stats
    
    def get_video_info(self, youtube_id: str) -> Optional[Dict[str, Any]]:
        """Get video information without downloading."""
        url = f"https://www.youtube.com/watch?v={youtube_id}"
        cmd = ["yt-dlp", "--dump-json", url]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except Exception as e:
            self.logger.error(f"Failed to get info for {youtube_id}: {str(e)}")
            return None
    
    def cleanup_failed_downloads(self):
        """Remove partially downloaded files."""
        for file_path in self.output_dir.glob("*.part"):
            file_path.unlink()
            self.logger.info(f"Removed partial download: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Download videos for Repurpose dataset")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON file")
    parser.add_argument("--output-dir", default="raw_videos", help="Output directory for videos")
    parser.add_argument("--max-videos", type=int, help="Maximum number of videos to download")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    downloader = VideoDownloader(args.output_dir, args.log_level)
    
    try:
        stats = downloader.download_from_dataset(args.dataset, args.max_videos)
        print(f"\nDownload Statistics:")
        print(f"Total videos: {stats['total_videos']}")
        print(f"Successful: {stats['successful_downloads']}")
        print(f"Failed: {stats['failed_downloads']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        
    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
        downloader.cleanup_failed_downloads()
    except Exception as e:
        print(f"Error: {str(e)}")
        downloader.cleanup_failed_downloads()


if __name__ == "__main__":
    main()