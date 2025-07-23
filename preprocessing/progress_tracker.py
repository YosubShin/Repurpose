#!/usr/bin/env python3
"""
Progress tracker for video processing pipeline with detailed error categorization.
"""

import time
import sys
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import threading
from enum import Enum


class ErrorCategory(Enum):
    """Categories of download/processing errors"""
    PRIVATE_VIDEO = "private_video"
    DELETED_VIDEO = "deleted_video"
    TERMINATED_ACCOUNT = "terminated_account"
    FORMAT_UNAVAILABLE = "format_unavailable"
    COPYRIGHT = "copyright"
    BOT_DETECTION = "bot_detection"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"


@dataclass
class VideoStatus:
    """Status of a single video"""
    youtube_id: str
    status: str  # 'pending', 'downloading', 'processing', 'completed', 'failed', 'skipped'
    error_category: Optional[ErrorCategory] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class ProgressTracker:
    """Live progress tracker for video processing pipeline"""
    
    def __init__(self, total_videos: int, task_name: str = "Processing"):
        self.total_videos = total_videos
        self.task_name = task_name
        
        # Video status tracking
        self.videos: Dict[str, VideoStatus] = {}
        self.completed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.processing_count = 0
        
        # Error categorization
        self.error_counts = defaultdict(int)
        self.error_examples = defaultdict(list)
        
        # Display control
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.last_update_time = 0
        self.update_interval = 0.1  # Update display every 100ms
        
        # Terminal control
        self.is_tty = sys.stdout.isatty()
        self.last_line_count = 0
        
    def categorize_error(self, error_message: str) -> ErrorCategory:
        """Categorize error based on message content"""
        error_lower = error_message.lower()
        
        # Check for specific error patterns
        if "private" in error_lower and "video" in error_lower:
            return ErrorCategory.PRIVATE_VIDEO
        elif "no longer available" in error_lower or "terminated" in error_lower:
            return ErrorCategory.TERMINATED_ACCOUNT
        elif "deleted" in error_lower or "removed" in error_lower:
            return ErrorCategory.DELETED_VIDEO
        elif "format" in error_lower and "not available" in error_lower:
            return ErrorCategory.FORMAT_UNAVAILABLE
        elif "copyright" in error_lower:
            return ErrorCategory.COPYRIGHT
        elif "sign in" in error_lower or "bot" in error_lower:
            return ErrorCategory.BOT_DETECTION
        elif "network" in error_lower or "connection" in error_lower:
            return ErrorCategory.NETWORK_ERROR
        else:
            return ErrorCategory.UNKNOWN
    
    def is_retryable_error(self, error_category: ErrorCategory) -> bool:
        """Determine if an error category is worth retrying"""
        non_retryable = {
            ErrorCategory.PRIVATE_VIDEO,
            ErrorCategory.DELETED_VIDEO,
            ErrorCategory.TERMINATED_ACCOUNT,
            ErrorCategory.COPYRIGHT
        }
        return error_category not in non_retryable
    
    def add_video(self, youtube_id: str):
        """Add a video to track"""
        with self.lock:
            self.videos[youtube_id] = VideoStatus(
                youtube_id=youtube_id,
                status='pending'
            )
    
    def update_video_status(self, youtube_id: str, status: str, 
                           error_message: Optional[str] = None,
                           retry_count: int = 0):
        """Update the status of a video"""
        with self.lock:
            if youtube_id not in self.videos:
                self.videos[youtube_id] = VideoStatus(youtube_id=youtube_id, status=status)
            
            video = self.videos[youtube_id]
            old_status = video.status
            video.status = status
            video.retry_count = retry_count
            
            # Update counters
            if old_status == 'processing':
                self.processing_count -= 1
            
            if status == 'processing':
                self.processing_count += 1
                video.start_time = time.time()
            elif status == 'completed':
                self.completed_count += 1
                video.end_time = time.time()
            elif status == 'failed':
                self.failed_count += 1
                video.end_time = time.time()
                if error_message:
                    video.error_message = error_message
                    video.error_category = self.categorize_error(error_message)
                    self.error_counts[video.error_category] += 1
                    # Keep up to 3 examples per error type
                    if len(self.error_examples[video.error_category]) < 3:
                        self.error_examples[video.error_category].append(youtube_id)
            elif status == 'skipped':
                self.skipped_count += 1
                video.end_time = time.time()
            
            self.display_progress()
    
    def clear_previous_lines(self):
        """Clear previous output lines"""
        if self.is_tty and self.last_line_count > 0:
            # Move cursor up and clear lines
            for _ in range(self.last_line_count):
                sys.stdout.write('\033[1A')  # Move up one line
                sys.stdout.write('\033[2K')   # Clear line
    
    def display_progress(self, force: bool = False):
        """Display current progress"""
        current_time = time.time()
        
        # Rate limit updates unless forced
        if not force and current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # Clear previous output
        self.clear_previous_lines()
        
        # Calculate statistics
        elapsed_time = current_time - self.start_time
        processed = self.completed_count + self.failed_count + self.skipped_count
        remaining = self.total_videos - processed
        
        if processed > 0:
            avg_time_per_video = elapsed_time / processed
            eta = avg_time_per_video * remaining
        else:
            eta = 0
        
        # Build output lines
        lines = []
        
        # Progress bar
        progress_pct = (processed / self.total_videos * 100) if self.total_videos > 0 else 0
        bar_width = 40
        filled_width = int(bar_width * processed / self.total_videos) if self.total_videos > 0 else 0
        bar = '█' * filled_width + '░' * (bar_width - filled_width)
        
        lines.append(f"\n{self.task_name}: [{bar}] {progress_pct:.1f}% ({processed}/{self.total_videos})")
        
        # Status summary
        status_line = f"✓ Completed: {self.completed_count} | ⏭ Skipped: {self.skipped_count} | ✗ Failed: {self.failed_count}"
        if self.processing_count > 0:
            status_line += f" | ⟳ Processing: {self.processing_count}"
        lines.append(status_line)
        
        # Time information
        elapsed_str = self.format_time(elapsed_time)
        eta_str = self.format_time(eta)
        lines.append(f"Elapsed: {elapsed_str} | ETA: {eta_str}")
        
        # Error breakdown if any
        if self.error_counts:
            lines.append("\nError Breakdown:")
            for error_type, count in sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True):
                error_name = error_type.value.replace('_', ' ').title()
                examples = self.error_examples[error_type]
                example_str = f" (e.g., {', '.join(examples[:2])})" if examples else ""
                lines.append(f"  • {error_name}: {count}{example_str}")
        
        # Print all lines
        output = '\n'.join(lines)
        print(output, flush=True)
        
        # Track line count for next clear
        self.last_line_count = len(lines) + 1  # +1 for the initial newline
    
    def format_time(self, seconds: float) -> str:
        """Format time in human-readable format"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def finalize(self):
        """Display final summary"""
        with self.lock:
            self.display_progress(force=True)
            
            # Add final summary
            print("\n" + "="*60)
            print(f"{self.task_name} Complete!")
            print("="*60)
            
            success_rate = (self.completed_count / self.total_videos * 100) if self.total_videos > 0 else 0
            print(f"Success Rate: {success_rate:.1f}%")
            print(f"Total Time: {self.format_time(time.time() - self.start_time)}")
            
            if self.failed_count > 0:
                print(f"\nFailed Videos: {self.failed_count}")
                # Show non-retryable errors
                non_retryable = [ErrorCategory.PRIVATE_VIDEO, ErrorCategory.DELETED_VIDEO, 
                               ErrorCategory.TERMINATED_ACCOUNT, ErrorCategory.COPYRIGHT]
                non_retryable_count = sum(self.error_counts[cat] for cat in non_retryable if cat in self.error_counts)
                if non_retryable_count > 0:
                    print(f"  • Non-retryable errors: {non_retryable_count}")
                    print("    (These videos cannot be downloaded and should not be retried)")