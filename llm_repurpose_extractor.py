#!/usr/bin/env python3
"""
LLM-based Video Repurpose Extractor using Gemini-2.5-Flash

This script uses Google's Gemini-2.5-Flash model to extract narrative clips
suitable for social media from video transcripts. It processes one sample at a time
due to rate limits and exports results in human-readable JSON format.
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pydantic import BaseModel

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Import metrics from utils
from utils.metrics import calculate_tiou

# Google AI imports
from google import genai

# Configure Google AI
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# Pydantic models for structured output
class ClipPrediction(BaseModel):
    start_time: float
    end_time: float
    confidence: float
    reasoning: str
    hook_description: str
    narrative_summary: str


class ClipsResponse(BaseModel):
    clips: List[ClipPrediction]


@dataclass
class TimeRange:
    """Represents a time range with start and end times."""

    start: float
    end: float

    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> Dict[str, float]:
        return {"start": self.start, "end": self.end}


@dataclass
class NarrativeClip:
    """Represents a predicted narrative clip."""

    time_range: TimeRange
    confidence: float
    reasoning: str
    hook_description: str
    narrative_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_range": self.time_range.to_dict(),
            "duration": self.time_range.duration(),
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "hook_description": self.hook_description,
            "narrative_summary": self.narrative_summary,
        }


class LLMRepurposeExtractor:
    """Extract narrative clips using Gemini-2.5-Flash model."""

    def __init__(self, model_name: str = "gemini-2.5-flash", log_level: str = "INFO"):
        """
        Initialize the LLM extractor.

        Args:
            model_name: Name of the Gemini model to use
            log_level: Logging level
        """
        self.model_name = model_name
        self.setup_logging(log_level)

        # Rate limiting
        self.request_delay = 1.0  # Seconds between requests
        self.last_request_time = 0

        # Output directories
        self.output_dir = Path("llm_results")
        self.debug_dir = Path("llm_debug")
        self.viz_dir = Path("llm_visualizations")

        self.output_dir.mkdir(exist_ok=True)
        self.debug_dir.mkdir(exist_ok=True)
        self.viz_dir.mkdir(exist_ok=True)

        self.logger.info(f"Initialized {model_name} model with structured output")

    def setup_logging(self, log_level: str):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def load_transcript(self, transcript_path: Path) -> List[Dict[str, Any]]:
        """
        Load transcript from JSON file.

        Args:
            transcript_path: Path to transcript JSON file

        Returns:
            List of transcript segments
        """
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript = json.load(f)
            self.logger.debug(f"Loaded transcript with {len(transcript)} segments")
            return transcript
        except Exception as e:
            self.logger.error(f"Failed to load transcript {transcript_path}: {e}")
            raise

    def format_transcript_with_timestamps(self, segments: List[Dict[str, Any]]) -> str:
        """
        Format transcript segments with timestamps for the LLM.

        Args:
            segments: List of transcript segments

        Returns:
            Formatted transcript string
        """
        formatted_lines = []

        for segment in segments:
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            text = segment.get("text", "").strip()

            if text:
                # Format time in seconds for consistency with output format
                time_str = f"[{start_time:.0f}s - {end_time:.0f}s]"
                formatted_lines.append(f"{time_str} {text}")

        return "\n".join(formatted_lines)

    def create_extraction_prompt(
        self, formatted_transcript: str, video_duration: float
    ) -> str:
        """
        Create the prompt for narrative clip extraction.

        Args:
            formatted_transcript: Formatted transcript with timestamps
            video_duration: Total video duration in seconds

        Returns:
            Extraction prompt
        """
        duration_min = int(video_duration // 60)
        duration_sec = int(video_duration % 60)

        prompt = f"""You are an expert video editor specializing in creating engaging social media clips. Your task is to identify segments from this {duration_min:02d}:{duration_sec:02d} video transcript that would make excellent standalone narrative clips for platforms like YouTube Shorts, TikTok, or Instagram Reels.

TRANSCRIPT:
{formatted_transcript}

REQUIREMENTS for each narrative clip:
1. **Duration**: 15-60 seconds (optimal for social media)
2. **Completeness**: Must be a complete, self-contained story or concept
3. **Immediate Hook**: First 3 seconds must grab attention immediately
4. **Coherent Narrative**: Should flow logically from start to finish
5. **Social Media Ready**: Should be engaging enough for direct publication
6. **No Context Required**: Viewers should understand without seeing the full video

EVALUATION CRITERIA:
- Does it have an immediate attention-grabbing hook?
- Is the narrative complete and self-contained?
- Would someone watch this entire clip without getting bored?
- Does it deliver value (entertainment, education, emotion, etc.)?
- Is it suitable for social media audiences?

IMPORTANT NOTES:
- Focus on NARRATIVE CLIPS, not just highlights or summaries
- Each clip should tell a complete story or convey a complete idea
- Prioritize clips with strong emotional impact or valuable insights
- Avoid clips that require external context to understand
- Look for moments with natural story arcs (setup → development → conclusion)

Please identify 2-5 high-quality narrative clips. Each clip should have:
- start_time: Start time in seconds (float) - must match the timestamp format shown in transcript
- end_time: End time in seconds (float) - must match the timestamp format shown in transcript
- confidence: Confidence score from 0.0 to 1.0
- reasoning: Why this makes a great narrative clip
- hook_description: What grabs attention in the first 3 seconds
- narrative_summary: Complete story or concept being conveyed

IMPORTANT: Use the exact second values shown in the transcript timestamps (e.g., if transcript shows [160s - 180s], use 160.0 and 180.0 as start/end times)."""

        return prompt

    def extract_clips_no_retry(
        self, prompt: str
    ) -> tuple[List[ClipPrediction], Optional[str]]:
        """
        Extract clips using LLM with structured output (no retry, fail fast).

        Args:
            prompt: The extraction prompt

        Returns:
            Tuple of (List of ClipPrediction objects, error message if failed)
        """
        try:
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_request_time < self.request_delay:
                time.sleep(self.request_delay - (current_time - self.last_request_time))

            self.logger.debug("Making structured LLM request")

            # Generate response with structured output
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "temperature": 0.1,  # Low temperature for consistent results
                    "response_mime_type": "application/json",
                    "response_schema": ClipsResponse,
                },
            )

            self.last_request_time = time.time()

            # Parse structured response
            clips_response = response.parsed
            if clips_response and clips_response.clips:
                self.logger.debug(
                    f"Successfully parsed structured response with {len(clips_response.clips)} clips"
                )
                return clips_response.clips, None
            else:
                self.logger.warning("Empty structured response from model")
                return [], "Empty response from model"

        except Exception as e:
            error_msg = f"Structured LLM request failed: {e}"
            self.logger.error(error_msg)
            return [], error_msg

    def get_file_key(self, split: str, sample_idx: int, youtube_id: str) -> str:
        """Generate file key for caching."""
        return f"{split}_{sample_idx}_{youtube_id}"

    def check_cache(self, file_key: str) -> Optional[Dict[str, Any]]:
        """Check if result already exists in cache."""
        # Look for main result file without timestamp
        cache_file = self.output_dir / f"{file_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    result = json.load(f)
                self.logger.info(f"Found cached result: {cache_file}")

                return result

            except Exception as e:
                self.logger.warning(f"Failed to load cached result {cache_file}: {e}")

        return None

    def save_debug_info(self, file_key: str, debug_info: Dict[str, Any]):
        """Save debug information for a sample."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_file = self.debug_dir / f"{file_key}_{timestamp}.json"

        try:
            with open(debug_file, "w", encoding="utf-8") as f:
                json.dump(debug_info, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Saved debug info: {debug_file}")
        except Exception as e:
            self.logger.error(f"Failed to save debug info to {debug_file}: {e}")

    def save_visualization(
        self, file_key: str, result: Dict[str, Any]
    ) -> Optional[Path]:
        """Save timeline visualization with proper naming."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_file = self.viz_dir / f"{file_key}_{timestamp}_timeline.png"

        try:
            self.create_timeline_visualization(result, viz_file)
            return viz_file
        except Exception as e:
            self.logger.error(f"Failed to save visualization to {viz_file}: {e}")
            return None

    def extract_transcript_for_range(
        self,
        transcript_segments: List[Dict[str, Any]],
        start_time: float,
        end_time: float,
    ) -> Dict[str, Any]:
        """
        Extract transcript segments that fall within the given time range.

        Args:
            transcript_segments: Full list of transcript segments
            start_time: Start time of the range in seconds
            end_time: End time of the range in seconds

        Returns:
            Dictionary with matching segments and any errors
        """
        matching_segments = []
        partial_segments = []

        for segment in transcript_segments:
            seg_start = segment.get("start", 0)
            seg_end = segment.get("end", 0)
            seg_text = segment.get("text", "").strip()

            if not seg_text:
                continue

            # Check for overlap with the time range
            if seg_end <= start_time or seg_start >= end_time:
                # No overlap
                continue
            elif seg_start >= start_time and seg_end <= end_time:
                # Fully contained
                matching_segments.append(
                    {
                        "start": seg_start,
                        "end": seg_end,
                        "text": seg_text,
                        "match_type": "full",
                    }
                )
            else:
                # Partial overlap
                overlap_start = max(seg_start, start_time)
                overlap_end = min(seg_end, end_time)
                overlap_ratio = (overlap_end - overlap_start) / (seg_end - seg_start)

                partial_segments.append(
                    {
                        "start": seg_start,
                        "end": seg_end,
                        "text": seg_text,
                        "match_type": "partial",
                        "overlap_start": overlap_start,
                        "overlap_end": overlap_end,
                        "overlap_ratio": overlap_ratio,
                    }
                )

        # Combine full and partial matches
        all_matches = matching_segments + partial_segments
        all_matches.sort(key=lambda x: x["start"])

        # Check for gaps or issues
        has_content = len(all_matches) > 0
        coverage_ratio = 0.0

        if has_content:
            # Calculate coverage ratio
            covered_time = 0
            for seg in all_matches:
                if seg["match_type"] == "full":
                    covered_time += seg["end"] - seg["start"]
                else:
                    covered_time += seg["overlap_end"] - seg["overlap_start"]
            coverage_ratio = covered_time / (end_time - start_time)

        return {
            "segments": all_matches,
            "has_content": has_content,
            "coverage_ratio": coverage_ratio,
            "time_range": {"start": start_time, "end": end_time},
            "num_segments": len(all_matches),
            "num_full_matches": len(matching_segments),
            "num_partial_matches": len(partial_segments),
        }

    def process_sample(
        self,
        sample: Dict[str, Any],
        transcript_dir: Path,
        split: str,
        sample_idx: int,
        debug_first: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a single sample to extract narrative clips.

        Args:
            sample: Sample data containing youtube_id and ground truth
            transcript_dir: Directory containing transcript files
            debug_first: If True, print the complete prompt for debugging

        Returns:
            Processing results with predictions and ground truth
        """
        youtube_id = sample["youtube_id"]
        file_key = self.get_file_key(split, sample_idx, youtube_id)

        self.logger.info(
            f"Processing sample {sample_idx}: {youtube_id} (key: {file_key})"
        )

        # Check cache first
        cached_result = self.check_cache(file_key)
        if cached_result:
            self.logger.info(f"Using cached result for {file_key}")
            return cached_result

        # Load transcript
        transcript_path = transcript_dir / f"{youtube_id}.json"

        # Initialize debug info
        debug_info = {
            "file_key": file_key,
            "youtube_id": youtube_id,
            "split": split,
            "sample_idx": sample_idx,
            "transcript_path": str(transcript_path),
            "transcript_exists": transcript_path.exists(),
            "processing_timestamp": datetime.now().isoformat(),
            "error": None,
            "llm_error": None,
            "video_duration": None,
            "transcript_segments_count": 0,
            "predictions_count": 0,
            "ground_truth_count": len(sample.get("segments", [])),
        }

        if not transcript_path.exists():
            debug_info["error"] = "Transcript not found"
            debug_info["predictions_with_debug"] = []
            self.logger.error(f"Transcript not found: {transcript_path}")

            # Save debug info for missing transcript
            self.save_debug_info(file_key, debug_info)

            return {
                "youtube_id": youtube_id,
                "file_key": file_key,
                "predictions": [],
                "ground_truth": sample.get("segments", []),
                # Remove error from main result - it's in debug
            }

        try:
            transcript_segments = self.load_transcript(transcript_path)

            # Get video duration from transcript
            video_duration = (
                max(seg.get("end", 0) for seg in transcript_segments)
                if transcript_segments
                else 0
            )

            debug_info["video_duration"] = video_duration
            debug_info["transcript_segments_count"] = len(transcript_segments)

            # Format transcript
            formatted_transcript = self.format_transcript_with_timestamps(
                transcript_segments
            )

            # Create prompt
            prompt = self.create_extraction_prompt(formatted_transcript, video_duration)
            debug_info["prompt_length"] = len(prompt)

            # Save prompt to debug info (always for all samples now)
            debug_info["prompt"] = (
                prompt if len(prompt) < 50000 else prompt[:50000] + "... (truncated)"
            )

            # Extract clips using structured output (no retry)
            clip_predictions, llm_error = self.extract_clips_no_retry(prompt)

            if llm_error:
                debug_info["llm_error"] = llm_error
                self.logger.error(f"LLM failed for {file_key}: {llm_error}")

                # Save debug info and return error result
                debug_info["predictions_with_debug"] = []
                self.save_debug_info(file_key, debug_info)
                return {
                    "youtube_id": youtube_id,
                    "file_key": file_key,
                    "predictions": [],
                    "ground_truth": sample.get("segments", []),
                    # Remove error from main result - it's in debug
                }

            # Process predictions and extract matching transcripts
            predictions = []
            for clip_pred in clip_predictions:
                try:
                    # Create NarrativeClip from structured prediction
                    clip = NarrativeClip(
                        time_range=TimeRange(
                            start=clip_pred.start_time,
                            end=clip_pred.end_time,
                        ),
                        confidence=clip_pred.confidence,
                        reasoning=clip_pred.reasoning,
                        hook_description=clip_pred.hook_description,
                        narrative_summary=clip_pred.narrative_summary,
                    )

                    # Extract matching transcript segments
                    transcript_match = self.extract_transcript_for_range(
                        transcript_segments, clip.time_range.start, clip.time_range.end
                    )

                    # Create prediction dict with transcript info
                    pred_dict = clip.to_dict()
                    pred_dict["transcript_match"] = transcript_match

                    # Log warnings if no transcript content found
                    if not transcript_match["has_content"]:
                        self.logger.warning(
                            f"No transcript segments found for predicted range "
                            f"[{clip.time_range.start:.1f}, {clip.time_range.end:.1f}] "
                            f"in {youtube_id}"
                        )
                        pred_dict["transcript_error"] = (
                            "No transcript segments in predicted range"
                        )
                    elif transcript_match["coverage_ratio"] < 0.5:
                        self.logger.warning(
                            f"Low transcript coverage ({transcript_match['coverage_ratio']:.2f}) "
                            f"for predicted range [{clip.time_range.start:.1f}, {clip.time_range.end:.1f}] "
                            f"in {youtube_id}"
                        )
                        pred_dict["transcript_warning"] = (
                            f"Low coverage: {transcript_match['coverage_ratio']:.2f}"
                        )

                    predictions.append(pred_dict)

                except Exception as e:
                    self.logger.warning(
                        f"Failed to process structured clip prediction: {e}"
                    )

            debug_info["predictions_count"] = len(predictions)
            self.logger.info(f"Extracted {len(predictions)} clips for {file_key}")

            # Add predictions debug info to debug data
            debug_predictions = []
            for pred in predictions:
                debug_pred = (
                    pred.copy()
                )  # Include all fields including transcript_match
                debug_predictions.append(debug_pred)

            debug_info["predictions_with_debug"] = debug_predictions

            # Create result first before evaluation
            result = {
                "youtube_id": youtube_id,
                "file_key": file_key,
                "video_duration": video_duration,
                "transcript_segments": len(transcript_segments),
                "predictions": predictions,
                "ground_truth": sample.get("segments", []),
                "processing_time": time.time(),
            }

            debug_info["evaluation"] = self.evaluate_predictions(result)

            # Save debug info for successful processing
            self.save_debug_info(file_key, debug_info)

            return result

        except Exception as e:
            error_msg = f"Failed to process sample {file_key}: {e}"
            debug_info["error"] = str(e)
            self.logger.error(error_msg)

            # Save debug info for failed processing
            self.save_debug_info(file_key, debug_info)

            return {
                "youtube_id": youtube_id,
                "file_key": file_key,
                "predictions": [],
                "ground_truth": sample.get("segments", []),
                # Remove error from main result - it's in debug
            }

    def evaluate_predictions(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate predictions against ground truth using tIoU metrics.

        Args:
            result: Result dictionary with predictions and ground truth

        Returns:
            Evaluation metrics
        """
        predictions = result.get("predictions", [])

        # Handle both metadata format (cached) and flat format (newly processed)
        if "metadata" in result:
            ground_truth = result.get("ground_truth", {}).get("segments", [])
        else:
            ground_truth = result.get("ground_truth", [])

        if not predictions or not ground_truth:
            return {
                "num_predictions": len(predictions),
                "num_ground_truth": len(ground_truth),
                "tiou": {},
            }

        # Convert predictions to list of tuples (start, end)
        predicted_segments = [
            (pred["time_range"]["start"], pred["time_range"]["end"])
            for pred in predictions
        ]

        # Convert ground truth to list of tuples (start, end)
        reference_segments = [
            (float(gt_segment[0]), float(gt_segment[1])) for gt_segment in ground_truth
        ]

        # Use standard tIoU calculation with multiple thresholds
        tiou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        tiou_results = calculate_tiou(
            reference_segments, predicted_segments, tiou_thresholds
        )

        return {
            "num_predictions": len(predictions),
            "num_ground_truth": len(ground_truth),
            "tiou": tiou_results,
        }

    def create_timeline_visualization(self, result: Dict[str, Any], output_path: Path):
        """
        Create timeline visualization comparing predictions vs ground truth.

        Args:
            result: Result dictionary with predictions and ground truth
            output_path: Path to save the visualization
        """
        try:
            # Handle both metadata format (cached) and flat format (newly processed)
            if "metadata" in result:
                youtube_id = result["metadata"]["youtube_id"]
                video_duration = result["metadata"].get("video_duration_seconds", 0)
                ground_truth = result.get("ground_truth", {}).get("segments", [])
            else:
                youtube_id = result["youtube_id"]
                video_duration = result.get("video_duration", 0)
                ground_truth = result.get("ground_truth", [])

            predictions = result.get("predictions", [])

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
            fig.suptitle(
                f"Timeline Comparison: {youtube_id}", fontsize=16, fontweight="bold"
            )

            # Ground Truth timeline
            ax1.set_title("Ground Truth Segments", fontsize=14, pad=20)
            ax1.set_xlim(0, video_duration)
            ax1.set_ylim(-0.5, 0.5)

            for i, (start, end) in enumerate(ground_truth):
                width = end - start
                rect = patches.Rectangle(
                    (start, -0.2),
                    width,
                    0.4,
                    linewidth=2,
                    edgecolor="darkgreen",
                    facecolor="lightgreen",
                    alpha=0.7,
                )
                ax1.add_patch(rect)

                # Add duration label
                duration = end - start
                ax1.text(
                    start + width / 2,
                    0,
                    f"{duration:.1f}s",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )

            ax1.set_ylabel("Ground Truth", fontsize=12)
            ax1.set_yticks([])
            ax1.grid(True, axis="x", alpha=0.3)

            # Predictions timeline
            ax2.set_title("LLM Predictions", fontsize=14, pad=20)
            ax2.set_xlim(0, video_duration)
            ax2.set_ylim(-0.5, 0.5)

            for i, pred in enumerate(predictions):
                time_range = pred["time_range"]
                start, end = time_range["start"], time_range["end"]
                width = end - start
                confidence = pred.get("confidence", 0.5)

                # Color based on confidence
                color_intensity = confidence
                rect = patches.Rectangle(
                    (start, -0.2),
                    width,
                    0.4,
                    linewidth=2,
                    edgecolor="darkblue",
                    facecolor="lightblue",
                    alpha=0.4 + 0.6 * color_intensity,
                )
                ax2.add_patch(rect)

                # Add duration and confidence label
                duration = end - start
                ax2.text(
                    start + width / 2,
                    0,
                    f"{duration:.1f}s\n({confidence:.2f})",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )

            ax2.set_ylabel("Predictions", fontsize=12)
            ax2.set_xlabel("Time (seconds)", fontsize=12)
            ax2.set_yticks([])
            ax2.grid(True, axis="x", alpha=0.3)

            # Add time markers
            time_markers = np.arange(
                0, video_duration + 1, max(1, video_duration // 10)
            )
            for ax in [ax1, ax2]:
                ax.set_xticks(time_markers)
                ax.set_xticklabels(
                    [f"{int(t//60):02d}:{int(t%60):02d}" for t in time_markers]
                )

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            self.logger.debug(f"Timeline visualization saved: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to create timeline visualization: {e}")

    def export_results(self, result: Dict[str, Any]) -> Path:
        """
        Export results to human-readable JSON file with caching.

        Args:
            result: Processing result to export

        Returns:
            Path to exported file
        """
        # Handle both metadata format (cached) and flat format (newly processed)
        if "metadata" in result:
            file_key = result["metadata"]["file_key"]
        else:
            file_key = result.get("file_key", result["youtube_id"])

        # Create comprehensive result
        evaluation = self.evaluate_predictions(result)

        # Clean predictions - remove debug fields
        clean_predictions = []
        for pred in result.get("predictions", []):
            clean_pred = {
                "time_range": pred["time_range"],
                "duration": pred["duration"],
                "confidence": pred["confidence"],
                "reasoning": pred["reasoning"],
                "hook_description": pred["hook_description"],
                "narrative_summary": pred["narrative_summary"],
            }
            # Skip debug fields: transcript_match, transcript_error, transcript_warning
            clean_predictions.append(clean_pred)

        # Handle both metadata format (cached) and flat format (newly processed)
        if "metadata" in result:
            # Already in metadata format (cached result)
            youtube_id = result["metadata"]["youtube_id"]
            video_duration = result["metadata"].get("video_duration_seconds", 0)
            transcript_segments = result["metadata"].get("transcript_segments", 0)
            ground_truth_segments = result.get("ground_truth", {}).get("segments", [])
        else:
            # Flat format (newly processed)
            youtube_id = result["youtube_id"]
            video_duration = result.get("video_duration", 0)
            transcript_segments = result.get("transcript_segments", 0)
            ground_truth_segments = result.get("ground_truth", [])

        export_data = {
            "metadata": {
                "file_key": file_key,
                "youtube_id": youtube_id,
                "processed_at": datetime.now().isoformat(),
                "model_name": self.model_name,
                "video_duration_seconds": video_duration,
                "transcript_segments": transcript_segments,
            },
            "predictions": clean_predictions,
            "ground_truth": {
                "segments": ground_truth_segments,
                "num_segments": len(ground_truth_segments),
            },
            "evaluation": evaluation,
            # Remove errors from main results - they go to debug
        }

        # Export to JSON using file_key WITHOUT timestamp for caching
        output_file = self.output_dir / f"{file_key}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        # Create timeline visualization using the new method
        viz_file = self.save_visualization(file_key, result)

        self.logger.info(f"Results exported: {output_file}")
        return output_file

    def process_dataset(
        self,
        dataset_path: Path,
        transcript_dir: Path,
        max_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process entire dataset one sample at a time.

        Args:
            dataset_path: Path to dataset JSON file
            transcript_dir: Directory containing transcript files
            max_samples: Maximum number of samples to process (for testing)

        Returns:
            List of processing results
        """
        # Load dataset
        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        if max_samples:
            dataset = dataset[:max_samples]

        # Detect split from dataset path
        split = "unknown"
        if "train" in dataset_path.name.lower():
            split = "train"
        elif "val" in dataset_path.name.lower() or "dev" in dataset_path.name.lower():
            split = "val"
        elif "test" in dataset_path.name.lower():
            split = "test"

        self.logger.info(
            f"Processing {len(dataset)} samples from {dataset_path} (split: {split})"
        )

        results = []
        all_tiou_results = []

        for i, sample in enumerate(dataset):
            self.logger.info(f"Processing sample {i+1}/{len(dataset)}")

            # Process sample with new parameters
            result = self.process_sample(
                sample, transcript_dir, split, i, debug_first=(i == 0)
            )

            # Export individual result
            self.export_results(result)

            results.append(result)

            # Collect tIoU results for aggregation (only if we have predictions)
            if result.get("predictions"):  # Check if predictions exist and not empty
                evaluation = self.evaluate_predictions(result)
                tiou_data = evaluation.get("tiou", {})
                if tiou_data:
                    all_tiou_results.append(tiou_data)

            # Rate limiting between samples
            if i < len(dataset) - 1:  # Don't wait after last sample
                time.sleep(self.request_delay)

        # Calculate and print aggregated tIoU metrics
        if all_tiou_results:
            self.print_aggregated_metrics(all_tiou_results, split)

        return results

    def print_aggregated_metrics(
        self, all_tiou_results: List[Dict[float, float]], split: str
    ):
        """Print aggregated tIoU metrics across all samples."""
        if not all_tiou_results:
            self.logger.warning("No tIoU results to aggregate")
            return

        # Calculate mean per threshold
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        mean_per_threshold = {}

        for threshold in thresholds:
            values = [result.get(threshold, 0.0) for result in all_tiou_results]
            mean_per_threshold[threshold] = sum(values) / len(values) if values else 0.0

        # Calculate average across all thresholds
        overall_average = (
            sum(mean_per_threshold.values()) / len(mean_per_threshold)
            if mean_per_threshold
            else 0.0
        )

        # Print results
        print(f"\n{'='*60}")
        print(f"AGGREGATED tIoU METRICS ({split.upper()} SET)")
        print(f"{'='*60}")
        print(f"Number of successful samples: {len(all_tiou_results)}")
        print(f"Mean tIoU per threshold:")
        for threshold in sorted(mean_per_threshold.keys()):
            print(f"  @{threshold}: {mean_per_threshold[threshold]:.4f}")
        print(f"Overall average tIoU: {overall_average:.4f}")
        print(f"{'='*60}\n")

        # Log to file as well
        self.logger.info(
            f"Aggregated tIoU for {split}: {mean_per_threshold}, overall: {overall_average:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Extract narrative clips using LLM")
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Path to dataset JSON file"
    )
    parser.add_argument(
        "--transcripts",
        type=Path,
        default=Path("data/transcripts"),
        help="Directory containing transcript files",
    )
    parser.add_argument(
        "--max-samples", type=int, help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--model", default="gemini-2.5-flash-lite", help="Gemini model name to use"
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    args = parser.parse_args()

    # Validate paths
    if not args.dataset.exists():
        print(f"Dataset file not found: {args.dataset}")
        return 1

    if not args.transcripts.exists():
        print(f"Transcripts directory not found: {args.transcripts}")
        return 1

    # Initialize extractor
    try:
        extractor = LLMRepurposeExtractor(
            model_name=args.model, log_level=args.log_level
        )
    except Exception as e:
        print(f"Failed to initialize extractor: {e}")
        return 1

    # Process dataset
    try:
        results = extractor.process_dataset(
            args.dataset, args.transcripts, args.max_samples
        )

        # Summary statistics
        total_samples = len(results)
        successful_samples = len([r for r in results if "error" not in r])
        total_predictions = sum(len(r.get("predictions", [])) for r in results)

        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total samples: {total_samples}")
        print(f"Successful: {successful_samples}")
        print(f"Failed: {total_samples - successful_samples}")
        print(f"Total predictions: {total_predictions}")
        print(
            f"Average predictions per sample: {total_predictions/successful_samples:.2f}"
        )
        print(f"Results exported to: {extractor.output_dir}")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 1
    except Exception as e:
        import traceback

        print(f"Processing failed: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
