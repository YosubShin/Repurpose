import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import time
import yaml

from video_downloader import VideoDownloader
from visual_feature_extractor_clip import VisualFeatureExtractorCLIP
from audio_feature_extractor import AudioFeatureExtractor
from text_feature_extractor import TextFeatureExtractor


class PreprocessingPipeline:
    def __init__(self, config_path: str = "preprocessing_config.yaml", log_level: str = "INFO"):
        """
        Initialize the preprocessing pipeline.

        Args:
            config_path: Path to configuration file
            log_level: Logging level
        """
        self.config_path = config_path
        self.config = self.load_config()

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.video_downloader = VideoDownloader(
            output_dir=self.config['directories']['raw_videos'],
            log_level=log_level
        )

        self.visual_extractor = VisualFeatureExtractorCLIP(
            output_dir=self.config['directories']['video_features'],
            log_level=log_level
        )

        self.audio_extractor = AudioFeatureExtractor(
            output_dir=self.config['directories']['audio_features'],
            log_level=log_level
        )

        self.text_extractor = TextFeatureExtractor(
            output_dir=self.config['directories']['text_features'],
            log_level=log_level
        )

        # Create output directories
        self.create_directories()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Return default configuration
            return self.get_default_config()

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'directories': {
                'raw_videos': 'raw_videos',
                'video_features': 'data/video_clip_features',
                'audio_features': 'data/audio_pann_features',
                'text_features': 'data/caption_features'
            },
            'processing': {
                'max_videos': None,
                'batch_size': 10,
                'resume_on_failure': True,
                'cleanup_raw_videos': False
            },
            'datasets': {
                'train': 'data/train.json',
                'val': 'data/val.json',
                'test': 'data/test.json'
            }
        }

    def create_directories(self):
        """Create necessary directories."""
        for dir_path in self.config['directories'].values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def save_config(self):
        """Save current configuration to file."""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)

    def process_dataset(self, dataset_path: str, steps: List[str] = None) -> Dict[str, Any]:
        """
        Process a complete dataset through the pipeline.

        Args:
            dataset_path: Path to dataset JSON file
            steps: List of steps to execute. Options: ['download', 'visual', 'audio', 'text']
                  If None, executes all steps.

        Returns:
            Dict containing processing statistics
        """
        if steps is None:
            steps = ['download', 'visual', 'audio', 'text']

        self.logger.info(f"Starting preprocessing pipeline for {dataset_path}")
        self.logger.info(f"Steps to execute: {steps}")

        results = {}
        start_time = time.time()

        try:
            # Step 1: Download videos
            if 'download' in steps:
                self.logger.info("=" * 60)
                self.logger.info("STEP 1: Downloading videos")
                self.logger.info("=" * 60)

                download_stats = self.video_downloader.download_from_dataset(
                    dataset_path=dataset_path,
                    max_videos=self.config['processing']['max_videos']
                )
                results['download'] = download_stats

                self.logger.info(
                    f"Download completed: {download_stats['success_rate']:.1f}% success rate")

            # Step 2: Extract visual features
            if 'visual' in steps:
                self.logger.info("=" * 60)
                self.logger.info("STEP 2: Extracting visual features")
                self.logger.info("=" * 60)

                visual_stats = self.visual_extractor.process_from_dataset(
                    dataset_path=dataset_path,
                    video_dir=self.config['directories']['raw_videos'],
                    max_videos=self.config['processing']['max_videos']
                )
                results['visual'] = visual_stats

                self.logger.info(f"Visual extraction completed: {visual_stats['success_rate']:.1f}% success rate")

            # Step 3: Extract audio features
            if 'audio' in steps:
                self.logger.info("=" * 60)
                self.logger.info("STEP 3: Extracting audio features")
                self.logger.info("=" * 60)

                audio_stats = self.audio_extractor.process_from_dataset(
                    dataset_path=dataset_path,
                    video_dir=self.config['directories']['raw_videos'],
                    max_videos=self.config['processing']['max_videos']
                )
                results['audio'] = audio_stats

                self.logger.info(
                    f"Audio extraction completed: {audio_stats['success_rate']:.1f}% success rate")

            # Step 4: Extract text features
            if 'text' in steps:
                self.logger.info("=" * 60)
                self.logger.info("STEP 4: Extracting text features")
                self.logger.info("=" * 60)

                text_stats = self.text_extractor.process_from_dataset(
                    dataset_path=dataset_path,
                    video_dir=self.config['directories']['raw_videos'],
                    max_videos=self.config['processing']['max_videos']
                )
                results['text'] = text_stats

                self.logger.info(
                    f"Text extraction completed: {text_stats['success_rate']:.1f}% success rate")

            # Cleanup raw videos if requested
            if self.config['processing']['cleanup_raw_videos'] and 'download' in steps:
                self.logger.info("Cleaning up raw video files...")
                self.cleanup_raw_videos()

            total_time = time.time() - start_time
            results['total_time'] = total_time

            self.logger.info("=" * 60)
            self.logger.info("PREPROCESSING COMPLETE")
            self.logger.info("=" * 60)
            self.logger.info(
                f"Total processing time: {total_time:.1f} seconds")

            return results

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

    def process_all_datasets(self, steps: List[str] = None) -> Dict[str, Any]:
        """
        Process all datasets (train, val, test) through the pipeline.

        Args:
            steps: List of steps to execute

        Returns:
            Dict containing processing statistics for all datasets
        """
        results = {}

        for dataset_name, dataset_path in self.config['datasets'].items():
            if not Path(dataset_path).exists():
                self.logger.warning(
                    f"Dataset {dataset_name} not found at {dataset_path}, skipping...")
                continue

            self.logger.info(f"Processing {dataset_name} dataset...")
            try:
                results[dataset_name] = self.process_dataset(
                    dataset_path, steps)
            except Exception as e:
                self.logger.error(
                    f"Failed to process {dataset_name} dataset: {str(e)}")
                results[dataset_name] = {'error': str(e)}

        return results

    def cleanup_raw_videos(self):
        """Remove raw video files to save disk space."""
        raw_videos_dir = Path(self.config['directories']['raw_videos'])

        if not raw_videos_dir.exists():
            return

        video_files = list(raw_videos_dir.glob("*.mp4"))

        for video_file in video_files:
            try:
                video_file.unlink()
                self.logger.debug(f"Removed raw video: {video_file}")
            except Exception as e:
                self.logger.warning(f"Failed to remove {video_file}: {str(e)}")

        self.logger.info(f"Cleaned up {len(video_files)} raw video files")

    def verify_features(self, dataset_path: str) -> Dict[str, Any]:
        """
        Verify that all required features have been extracted for a dataset.

        Args:
            dataset_path: Path to dataset JSON file

        Returns:
            Dict containing verification results
        """
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        video_features_dir = Path(self.config['directories']['video_features'])
        audio_features_dir = Path(self.config['directories']['audio_features'])
        text_features_dir = Path(self.config['directories']['text_features'])

        results = {
            'total_videos': len(dataset),
            'video_features_missing': [],
            'audio_features_missing': [],
            'text_features_missing': []
        }

        for video_info in dataset:
            youtube_id = video_info['youtube_id']

            # Check visual features
            if not (video_features_dir / f"{youtube_id}.npy").exists():
                results['video_features_missing'].append(youtube_id)

            # Check audio features
            if not (audio_features_dir / f"{youtube_id}.npy").exists():
                results['audio_features_missing'].append(youtube_id)

            # Check text features
            if not (text_features_dir / f"{youtube_id}.npy").exists():
                results['text_features_missing'].append(youtube_id)

        results['video_features_complete'] = len(
            dataset) - len(results['video_features_missing'])
        results['audio_features_complete'] = len(
            dataset) - len(results['audio_features_missing'])
        results['text_features_complete'] = len(
            dataset) - len(results['text_features_missing'])

        return results

    def print_verification_results(self, verification_results: Dict[str, Any]):
        """Print feature verification results."""
        total = verification_results['total_videos']

        print(f"\nFeature Verification Results:")
        print(f"Total videos in dataset: {total}")
        print(f"Video features: {verification_results['video_features_complete']}/{total} "
              f"({verification_results['video_features_complete']/total*100:.1f}%)")
        print(f"Audio features: {verification_results['audio_features_complete']}/{total} "
              f"({verification_results['audio_features_complete']/total*100:.1f}%)")
        print(f"Text features: {verification_results['text_features_complete']}/{total} "
              f"({verification_results['text_features_complete']/total*100:.1f}%)")

        if verification_results['video_features_missing']:
            print(
                f"\nMissing video features: {len(verification_results['video_features_missing'])} videos")
        if verification_results['audio_features_missing']:
            print(
                f"Missing audio features: {len(verification_results['audio_features_missing'])} videos")
        if verification_results['text_features_missing']:
            print(
                f"Missing text features: {len(verification_results['text_features_missing'])} videos")


def main():
    parser = argparse.ArgumentParser(
        description="Main preprocessing pipeline for Repurpose dataset")
    parser.add_argument(
        "--config", default="preprocessing_config.yaml", help="Path to configuration file")
    parser.add_argument("--dataset", help="Path to specific dataset JSON file")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Process all datasets (train, val, test)")
    parser.add_argument("--steps", nargs="+", choices=["download", "visual", "audio", "text"],
                        help="Specific steps to execute")
    parser.add_argument("--verify", action="store_true",
                        help="Verify feature extraction completeness")
    parser.add_argument("--max-videos", type=int,
                        help="Maximum number of videos to process")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = PreprocessingPipeline(args.config, args.log_level)

    # Override max_videos if specified
    if args.max_videos:
        pipeline.config['processing']['max_videos'] = args.max_videos

    try:
        if args.verify:
            # Verify feature extraction completeness
            if args.dataset:
                results = pipeline.verify_features(args.dataset)
                pipeline.print_verification_results(results)
            elif args.all_datasets:
                for dataset_name, dataset_path in pipeline.config['datasets'].items():
                    if Path(dataset_path).exists():
                        print(f"\n{dataset_name.upper()} DATASET:")
                        results = pipeline.verify_features(dataset_path)
                        pipeline.print_verification_results(results)
            else:
                print("Please specify --dataset or --all-datasets for verification")

        elif args.all_datasets:
            # Process all datasets
            results = pipeline.process_all_datasets(args.steps)
            print(f"\nOverall Results:")
            for dataset_name, stats in results.items():
                if 'error' in stats:
                    print(f"{dataset_name}: FAILED - {stats['error']}")
                else:
                    print(
                        f"{dataset_name}: Completed in {stats.get('total_time', 0):.1f}s")

        elif args.dataset:
            # Process specific dataset
            results = pipeline.process_dataset(args.dataset, args.steps)
            print(f"\nProcessing Results:")
            for step, stats in results.items():
                if step == 'total_time':
                    continue
                print(
                    f"{step.title()}: {stats.get('success_rate', 0):.1f}% success rate")

        else:
            print("Please specify --dataset or --all-datasets")
            return

        # Save configuration
        pipeline.save_config()

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
