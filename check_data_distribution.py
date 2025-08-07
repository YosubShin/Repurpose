#!/usr/bin/env python3
"""Check the distribution of positive/negative labels in the dataset."""

import json
import numpy as np
import argparse

def check_distribution(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    total_duration = 0
    total_highlight = 0
    videos_with_highlights = 0
    
    for entry in data:
        duration = entry.get('timeRangeOffset', [0, 0])
        video_duration = duration[1] - duration[0]
        segments = entry.get('segmentsOffset', [])
        
        if segments:
            videos_with_highlights += 1
            for start, end in segments:
                total_highlight += (end - start)
        
        total_duration += video_duration
    
    print(f"Dataset: {json_file}")
    print(f"Total entries: {len(data)}")
    print(f"Entries with highlights: {videos_with_highlights} ({100*videos_with_highlights/len(data):.1f}%)")
    print(f"Total duration: {total_duration:.0f} seconds")
    print(f"Highlight duration: {total_highlight:.0f} seconds")
    print(f"Highlight ratio: {100*total_highlight/total_duration:.1f}%")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    args = parser.parse_args()
    
    check_distribution(args.train)
    check_distribution(args.val)