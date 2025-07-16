#!/usr/bin/env python3
"""
Script to split the Repurpose dataset into multiple chunks for parallel processing.
This allows distributing preprocessing across multiple machines/jobs.
"""

import json
import argparse
import math
from pathlib import Path
from typing import List, Dict, Any


def split_dataset(input_file: str, output_dir: str, num_splits: int, prefix: str = "chunk") -> List[str]:
    """
    Split a dataset JSON file into multiple smaller files.
    
    Args:
        input_file: Path to the original dataset JSON file
        output_dir: Directory to save the split files
        num_splits: Number of splits to create
        prefix: Prefix for the output files
        
    Returns:
        List of paths to the created split files
    """
    # Load the dataset
    with open(input_file, 'r') as f:
        dataset = json.load(f)
    
    total_videos = len(dataset)
    videos_per_split = math.ceil(total_videos / num_splits)
    
    print(f"Splitting {total_videos} videos into {num_splits} chunks of ~{videos_per_split} videos each")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    split_files = []
    
    for i in range(num_splits):
        start_idx = i * videos_per_split
        end_idx = min((i + 1) * videos_per_split, total_videos)
        
        if start_idx >= total_videos:
            break
            
        chunk_data = dataset[start_idx:end_idx]
        chunk_size = len(chunk_data)
        
        # Create output filename
        input_stem = Path(input_file).stem
        output_file = output_dir / f"{input_stem}_{prefix}_{i:03d}.json"
        
        # Save the chunk
        with open(output_file, 'w') as f:
            json.dump(chunk_data, f, indent=2)
        
        split_files.append(str(output_file))
        print(f"  Created {output_file} with {chunk_size} videos (indices {start_idx}-{end_idx-1})")
    
    return split_files


def split_all_datasets(data_dir: str, output_dir: str, num_splits: int) -> Dict[str, List[str]]:
    """
    Split all dataset files (train, val, test) into chunks.
    
    Args:
        data_dir: Directory containing the original dataset files
        output_dir: Directory to save the split files
        num_splits: Number of splits to create for each dataset
        
    Returns:
        Dictionary mapping dataset names to lists of split file paths
    """
    data_dir = Path(data_dir)
    results = {}
    
    for dataset_file in ['train.json', 'val.json', 'test.json']:
        dataset_path = data_dir / dataset_file
        
        if dataset_path.exists():
            print(f"\nSplitting {dataset_file}...")
            split_files = split_dataset(
                str(dataset_path), 
                output_dir, 
                num_splits, 
                prefix="chunk"
            )
            results[dataset_file.replace('.json', '')] = split_files
        else:
            print(f"Warning: {dataset_path} not found, skipping...")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Split Repurpose dataset into chunks for parallel processing")
    parser.add_argument("--data-dir", default="data", help="Directory containing dataset JSON files")
    parser.add_argument("--output-dir", default="data/splits", help="Output directory for split files")
    parser.add_argument("--num-splits", type=int, required=True, help="Number of splits to create")
    parser.add_argument("--dataset", choices=["train", "val", "test", "all"], default="all", 
                       help="Which dataset to split (default: all)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dataset == "all":
        results = split_all_datasets(args.data_dir, args.output_dir, args.num_splits)
        
        print(f"\n=== SPLIT SUMMARY ===")
        total_files = 0
        for dataset_name, split_files in results.items():
            print(f"{dataset_name.upper()}: {len(split_files)} splits")
            total_files += len(split_files)
        
        print(f"Total split files created: {total_files}")
        print(f"Split files saved to: {args.output_dir}")
        
        # Create a manifest file listing all splits
        manifest = {
            "num_splits_per_dataset": args.num_splits,
            "total_split_files": total_files,
            "splits": results
        }
        
        manifest_path = output_dir / "split_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Split manifest saved to: {manifest_path}")
        
    else:
        # Split single dataset
        dataset_file = f"{args.dataset}.json"
        dataset_path = Path(args.data_dir) / dataset_file
        
        if not dataset_path.exists():
            print(f"Error: {dataset_path} not found")
            return
        
        split_files = split_dataset(str(dataset_path), args.output_dir, args.num_splits)
        
        print(f"\n=== SPLIT SUMMARY ===")
        print(f"Created {len(split_files)} splits for {args.dataset} dataset")
        print(f"Split files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()