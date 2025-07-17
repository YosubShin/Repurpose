import os
import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def scan_and_remove_truncated_features(feature_dir: str, verbose: bool = True) -> Tuple[List[str], List[str]]:
    """
    Scan a feature directory and remove NPY files with exactly 1800 in the first dimension.
    
    Args:
        feature_dir: Directory containing NPY files
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (removed_files, kept_files) lists
    """
    feature_dir_path = Path(feature_dir)
    if not feature_dir_path.exists():
        print(f"Directory {feature_dir} does not exist")
        return [], []
    
    npy_files = list(feature_dir_path.glob("*.npy"))
    removed_files = []
    kept_files = []
    
    print(f"\nScanning {len(npy_files)} NPY files in {feature_dir}...")
    
    for npy_file in npy_files:
        try:
            # Load the NPY file to check its shape
            data = np.load(npy_file)
            shape = data.shape
            
            if shape[0] == 1800:
                # This file is truncated at exactly 1800 seconds
                if verbose:
                    print(f"  Removing truncated file: {npy_file.name} (shape: {shape})")
                
                # Delete the file
                npy_file.unlink()
                removed_files.append(npy_file.stem)  # YouTube ID without .npy
            else:
                kept_files.append(npy_file.stem)
                if verbose and shape[0] > 1800:
                    print(f"  Keeping full-length file: {npy_file.name} (shape: {shape})")
                    
        except Exception as e:
            print(f"  Error processing {npy_file.name}: {e}")
            kept_files.append(npy_file.stem)
    
    return removed_files, kept_files


def update_extraction_progress(progress_file: Path, removed_ids: List[str]):
    """
    Update extraction progress JSON file to remove deleted video IDs.
    
    Args:
        progress_file: Path to extraction_progress.json
        removed_ids: List of YouTube IDs that were removed
    """
    if not progress_file.exists():
        print(f"Progress file {progress_file} does not exist")
        return
    
    # Load existing progress
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    
    # Remove the deleted IDs
    initial_count = len(progress)
    for video_id in removed_ids:
        if video_id in progress:
            del progress[video_id]
    
    # Save updated progress
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
    
    removed_count = initial_count - len(progress)
    print(f"  Updated {progress_file.name}: removed {removed_count} entries")


def cleanup_truncated_features(base_dir: str = "data", verbose: bool = True) -> Dict[str, Dict[str, int]]:
    """
    Clean up truncated features from all feature directories.
    
    Args:
        base_dir: Base directory containing feature subdirectories
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary with statistics for each feature type
    """
    feature_dirs = {
        'visual': os.path.join(base_dir, 'video_clip_features'),
        'audio': os.path.join(base_dir, 'audio_pann_features'),
        'text': os.path.join(base_dir, 'caption_features')
    }
    
    results = {}
    
    for feature_type, feature_dir in feature_dirs.items():
        print(f"\n{'='*60}")
        print(f"Processing {feature_type.upper()} features")
        print(f"{'='*60}")
        
        # Scan and remove truncated files
        removed_files, kept_files = scan_and_remove_truncated_features(feature_dir, verbose)
        
        # Update extraction progress
        progress_file = Path(feature_dir) / "extraction_progress.json"
        if removed_files and progress_file.exists():
            update_extraction_progress(progress_file, removed_files)
        
        results[feature_type] = {
            'removed': len(removed_files),
            'kept': len(kept_files),
            'total': len(removed_files) + len(kept_files)
        }
        
        print(f"\nSummary for {feature_type} features:")
        print(f"  Removed: {len(removed_files)} truncated files")
        print(f"  Kept: {len(kept_files)} files")
        print(f"  Total processed: {results[feature_type]['total']} files")
    
    return results


def print_overall_summary(results: Dict[str, Dict[str, int]]):
    """Print overall cleanup summary."""
    print(f"\n{'='*60}")
    print("OVERALL CLEANUP SUMMARY")
    print(f"{'='*60}")
    
    total_removed = sum(r['removed'] for r in results.values())
    total_kept = sum(r['kept'] for r in results.values())
    total_processed = sum(r['total'] for r in results.values())
    
    print(f"\nTotal files processed: {total_processed}")
    print(f"Total files removed: {total_removed}")
    print(f"Total files kept: {total_kept}")
    
    if total_removed > 0:
        print(f"\n✓ Successfully removed {total_removed} truncated feature files.")
        print("  These files will need to be regenerated with the updated extractors.")
    else:
        print("\n✓ No truncated files found. All features appear to be full-length.")
    
    # Check if any feature types had disproportionate truncation
    for feature_type, stats in results.items():
        if stats['total'] > 0:
            truncation_rate = (stats['removed'] / stats['total']) * 100
            if truncation_rate > 0:
                print(f"\n  {feature_type.capitalize()}: {truncation_rate:.1f}% were truncated")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up truncated feature files (NPY files with exactly 1800 in first dimension)")
    parser.add_argument("--base-dir", default="data",
                        help="Base directory containing feature subdirectories")
    parser.add_argument("--feature-type", choices=["visual", "audio", "text"],
                        help="Process only a specific feature type")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without actually deleting")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be deleted")
        print("This mode is not yet implemented. Please run without --dry-run")
        return
    
    verbose = not args.quiet
    
    try:
        if args.feature_type:
            # Process only specific feature type
            feature_dirs = {
                'visual': os.path.join(args.base_dir, 'video_clip_features'),
                'audio': os.path.join(args.base_dir, 'audio_pann_features'),
                'text': os.path.join(args.base_dir, 'caption_features')
            }
            
            feature_dir = feature_dirs[args.feature_type]
            removed_files, kept_files = scan_and_remove_truncated_features(feature_dir, verbose)
            
            # Update extraction progress
            progress_file = Path(feature_dir) / "extraction_progress.json"
            if removed_files and progress_file.exists():
                update_extraction_progress(progress_file, removed_files)
            
            print(f"\nCleaned up {args.feature_type} features:")
            print(f"  Removed: {len(removed_files)} truncated files")
            print(f"  Kept: {len(kept_files)} files")
        else:
            # Process all feature types
            results = cleanup_truncated_features(args.base_dir, verbose)
            print_overall_summary(results)
            
    except KeyboardInterrupt:
        print("\nCleanup interrupted by user")
    except Exception as e:
        print(f"Error during cleanup: {e}")
        raise


if __name__ == "__main__":
    main()