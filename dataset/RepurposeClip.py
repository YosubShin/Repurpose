from torch.utils.data import Dataset
import torch
import numpy as np
import os
import json
import logging


class RepurposeClip(Dataset):
    def __init__(self, label_path, video_path, audio_path, text_path):
        self.label_path = label_path
        self.video_path = video_path
        self.audio_path = audio_path
        self.text_path = text_path

        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Load original labels
        original_labels = json.load(open(label_path))

        self.video_format = os.path.join(video_path, '{}.npy')
        self.audio_format = os.path.join(audio_path, '{}.npy')
        self.text_format = os.path.join(text_path, '{}.npy')

        # Filter labels to only include samples with all three modalities
        self.label = self._filter_available_samples(original_labels)

        self.video_ids = list(set([k['youtube_id'] for k in self.label]))

        for k in self.label:
            k['labels'] = self.generate_time_status_list(
                k['timeRangeOffset'], k['segmentsOffset'])
            k['reg_offset'] = self.generate_regression_offsets(
                k['timeRangeOffset'], k['segmentsOffset'])

    def _filter_available_samples(self, original_labels):
        """
        Filter dataset to only include samples where all three modalities exist and are valid.

        Args:
            original_labels: List of original label dictionaries

        Returns:
            List of filtered label dictionaries
        """
        filtered_labels = []
        missing_visual = []
        missing_audio = []
        missing_text = []
        invalid_data = []

        for label_item in original_labels:
            video_id = label_item['youtube_id']

            # Check if all three modality files exist
            visual_path = self.video_format.format(video_id)
            audio_path = self.audio_format.format(video_id)
            text_path = self.text_format.format(video_id)

            visual_exists = os.path.exists(visual_path)
            audio_exists = os.path.exists(audio_path)
            text_exists = os.path.exists(text_path)

            if visual_exists and audio_exists and text_exists:
                # Additional validation: check if data is valid
                try:
                    is_valid = self._validate_sample_data(
                        label_item, visual_path, audio_path, text_path)
                    if is_valid:
                        filtered_labels.append(label_item)
                    else:
                        invalid_data.append(video_id)
                except Exception as e:
                    self.logger.warning(
                        f"Validation failed for {video_id}: {e}")
                    invalid_data.append(video_id)
            else:
                # Track what's missing for detailed logging
                if not visual_exists:
                    missing_visual.append(video_id)
                if not audio_exists:
                    missing_audio.append(video_id)
                if not text_exists:
                    missing_text.append(video_id)

        # Log statistics
        total_original = len(original_labels)
        total_kept = len(filtered_labels)
        total_dropped = total_original - total_kept

        self.logger.info(f"Dataset filtering results:")
        self.logger.info(f"  Original samples: {total_original}")
        self.logger.info(f"  Kept samples: {total_kept}")
        self.logger.info(f"  Dropped samples: {total_dropped}")
        self.logger.info(f"  Keep rate: {total_kept/total_original*100:.2f}%")

        if total_dropped > 0:
            self.logger.info(f"Missing/invalid breakdown:")
            self.logger.info(
                f"  Missing visual: {len(missing_visual)} samples")
            self.logger.info(f"  Missing audio: {len(missing_audio)} samples")
            self.logger.info(f"  Missing text: {len(missing_text)} samples")
            self.logger.info(f"  Invalid data: {len(invalid_data)} samples")

            # Log some example missing IDs (first 5)
            if missing_visual:
                self.logger.debug(
                    f"  Example missing visual: {missing_visual[:5]}")
            if missing_audio:
                self.logger.debug(
                    f"  Example missing audio: {missing_audio[:5]}")
            if missing_text:
                self.logger.debug(
                    f"  Example missing text: {missing_text[:5]}")
            if invalid_data:
                self.logger.debug(
                    f"  Example invalid data: {invalid_data[:5]}")

        return filtered_labels

    def _validate_sample_data(self, label_item, visual_path, audio_path, text_path):
        """
        Validate that a sample has proper data that won't cause tensor errors.

        Args:
            label_item: Label dictionary for the sample
            visual_path: Path to visual features
            audio_path: Path to audio features  
            text_path: Path to text features

        Returns:
            bool: True if sample is valid, False otherwise
        """
        try:
            # Load the feature files
            visual_feats = np.load(visual_path, allow_pickle=True)
            audio_feats = np.load(audio_path, allow_pickle=True)
            text_feats = np.load(text_path, allow_pickle=True)

            # Check basic shape requirements
            if len(visual_feats.shape) != 2 or len(audio_feats.shape) != 2 or len(text_feats.shape) != 2:
                self.logger.debug(f"Invalid feature shapes for {label_item['youtube_id']}: "
                                  f"visual={visual_feats.shape}, audio={audio_feats.shape}, text={text_feats.shape}")
                return False

            # Check for empty features
            if visual_feats.shape[0] == 0 or audio_feats.shape[0] == 0 or text_feats.shape[0] == 0:
                self.logger.debug(f"Empty features for {label_item['youtube_id']}: "
                                  f"visual={visual_feats.shape[0]}, audio={audio_feats.shape[0]}, text={text_feats.shape[0]}")
                return False

            # Generate labels and regression offsets to check for validity
            labels = self.generate_time_status_list(
                label_item['timeRangeOffset'], label_item['segmentsOffset'])
            reg_offsets = self.generate_regression_offsets(
                label_item['timeRangeOffset'], label_item['segmentsOffset'])

            # Check that we have valid labels and offsets
            if len(labels) == 0 or len(reg_offsets) == 0:
                self.logger.debug(f"Empty labels or offsets for {label_item['youtube_id']}: "
                                  f"labels={len(labels)}, offsets={len(reg_offsets)}")
                return False

            # Check that regression offsets have proper shape
            if not isinstance(reg_offsets, list) or len(reg_offsets) > 0:
                if isinstance(reg_offsets[0], (list, tuple)) and len(reg_offsets[0]) != 2:
                    self.logger.debug(f"Invalid regression offset shape for {label_item['youtube_id']}: "
                                      f"expected 2D tuples, got {type(reg_offsets[0])}")
                    return False

            # Apply time range filtering to check final lengths
            timeRange = label_item['timeRange']
            if timeRange[0] != 0:
                visual_slice = visual_feats[int(
                    timeRange[0]):int(timeRange[1]), :]
                audio_slice = audio_feats[int(
                    timeRange[0]):int(timeRange[1]), :]
                text_slice = text_feats[int(timeRange[0]):int(timeRange[1]), :]
            else:
                visual_slice = visual_feats
                audio_slice = audio_feats
                text_slice = text_feats

            # Check minimum length after slicing
            min_len = min(visual_slice.shape[0], audio_slice.shape[0], text_slice.shape[0], len(
                labels), len(reg_offsets))
            if min_len <= 0:
                self.logger.debug(
                    f"Zero length after processing for {label_item['youtube_id']}: min_len={min_len}")
                return False

            return True

        except Exception as e:
            self.logger.debug(
                f"Validation error for {label_item['youtube_id']}: {e}")
            return False

    def generate_time_status_list(self, time_range, segments):
        """
        Generate a list where each second in the time range is marked as 1 if it falls within
        any of the given segments, and 0 otherwise. This version supports non-integer ranges and segments.

        :param time_range: A list [begin, end] representing the overall time range.
        :param segments: A list of lists, where each sublist represents a time segment [begin, end].
        :return: A list representing the status of each second in the time range.
        """
        # Determine the total number of seconds in the time range
        total_seconds = int(time_range[1] - time_range[0]) + 1

        # Initialize the list with 0s for each second in the time range
        status_list = [0] * total_seconds

        # Iterate over each segment and mark the corresponding seconds in the status list
        for segment in segments:
            start = max(int(segment[0]), int(time_range[0]))
            end = min(int(segment[1]), int(time_range[1]))
            for i in range(start, end + 1):
                status_list[i - int(time_range[0])] = 1
        return status_list

    def generate_regression_offsets(self, time_range, segments):
        """
        Generate regression offsets for each second in the time range.
        Each offset is a tuple (left_offset, right_offset) representing the distance
        to the segment's start and end if the second is within a segment. If a second is outside 
        any segments, offsets are set to a default value (e.g., float('inf')).

        :param time_range: A list [begin, end] representing the overall time range.
        :param segments: A list of lists, where each sublist represents a time segment [begin, end].
        :return: A list of tuples representing the regression offsets for each second in the time range.
        """
        # Handle edge cases
        if not time_range or len(time_range) != 2:
            if hasattr(self, 'logger'):
                self.logger.warning(f"Invalid time_range: {time_range}")
            return [(0.0, 0.0)]  # Return minimal valid data

        if time_range[1] <= time_range[0]:
            if hasattr(self, 'logger'):
                self.logger.warning(
                    f"Invalid time_range: end ({time_range[1]}) <= start ({time_range[0]})")
            return [(0.0, 0.0)]

        total_seconds = int(time_range[1] - time_range[0]) + 1
        if total_seconds <= 0:
            if hasattr(self, 'logger'):
                self.logger.warning(
                    f"Non-positive total_seconds: {total_seconds}")
            return [(0.0, 0.0)]

        regression_offsets = [(0.0, 0.0)] * total_seconds

        # Handle empty segments
        if not segments:
            return regression_offsets

        for second in range(total_seconds):
            current_time = time_range[0] + second

            for segment in segments:
                # Handle invalid segments
                if not segment or len(segment) != 2:
                    continue

                if segment[1] <= segment[0]:
                    continue  # Skip invalid segments

                if segment[0] <= current_time <= segment[1]:
                    # Current time is inside this segment
                    left_offset = float(current_time - segment[0])
                    right_offset = float(segment[1] - current_time)
                    regression_offsets[second] = (left_offset, right_offset)
                    break

        return regression_offsets

    def load_data(self, path):
        # Seq_len, feature_dim
        try:
            return np.load(path, allow_pickle=True)
        except FileNotFoundError:
            self.logger.error(f"Feature file not found: {path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading feature file {path}: {e}")
            raise

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        video_id = self.label[idx]['youtube_id']
        timeRange = self.label[idx]['timeRange']

        feats_visual = self.load_data(self.video_format.format(video_id))
        feats_audio = self.load_data(self.audio_format.format(video_id))
        feats_text = self.load_data(self.text_format.format(video_id))

        if timeRange[0] != 0:
            feats_visual = feats_visual[int(timeRange[0]):int(timeRange[1]), :]
            feats_audio = feats_audio[int(timeRange[0]):int(timeRange[1]), :]
            feats_text = feats_text[int(timeRange[0]):int(timeRange[1]), :]

        saliency_score = self.label[idx]['labels']
        reg_offset = self.label[idx]['reg_offset']

        # avoid audio and visual features have different lengths
        min_len = min(feats_visual.shape[0], feats_audio.shape[0], len(
            saliency_score), len(reg_offset))
        feats = {'visual': feats_visual[:min_len],
                 'audio': feats_audio[:min_len], 'text': feats_text[:min_len]}
        saliency_score = saliency_score[:min_len]
        reg_offset = reg_offset[:min_len]

        # return a data dict
        data_dict = {'video_id': video_id,
                     'feats': feats,      # seq_len, feature_dim
                     'segments': reg_offset,   # seq_len x 2
                     'labels': saliency_score,     # seq_len
                     'duration': min_len,
                     }
        return data_dict


@torch.no_grad()
def preprocessing(vis_feats, aud_feats, text_feats, labels, segments, padding_val=0.0):
    """
    Generate batched features, masks, labels, and segments from a list of dict items
    """
    feats_visual = vis_feats
    feats_audio = aud_feats
    feats_text = text_feats

    feats_lens = torch.as_tensor([feat_visual.shape[0]
                                 for feat_visual in feats_visual])
    max_len = feats_lens.max().item()

    # Handle edge case where max_len is 0
    if max_len == 0:
        raise ValueError("All sequences in the batch have zero length")

    batch_shape_visual = torch.full(
        (len(feats_visual), max_len, feats_visual[0].shape[1]), padding_val)
    for i, seq in enumerate(feats_visual):
        length = seq.shape[0]
        if length > 0:
            batch_shape_visual[i, :length, ...] = seq

    batch_shape_audio = torch.full(
        (len(feats_audio), max_len, feats_audio[0].shape[1]), padding_val)
    for i, seq in enumerate(feats_audio):
        length = seq.shape[0]
        if length > 0:
            batch_shape_audio[i, :length, ...] = seq

    batch_shape_text = torch.full(
        (len(feats_text), max_len, feats_text[0].shape[1]), padding_val)
    for i, seq in enumerate(feats_text):
        length = seq.shape[0]
        if length > 0:
            batch_shape_text[i, :length, ...] = seq

    batch_shape_labels = torch.full((len(labels), max_len), padding_val)
    for i, seq in enumerate(labels):
        length = seq.shape[0]
        if length > 0:
            batch_shape_labels[i, :length] = seq

    # Handle segments with robust shape checking
    if len(segments) == 0:
        raise ValueError("No segments provided to preprocessing function")

    # Check if all segments are empty
    segment_lengths = [seq.shape[0] for seq in segments]
    if all(length == 0 for length in segment_lengths):
        raise ValueError("All segments in the batch have zero length")

    # Find the expected number of dimensions for segments
    # Look for the first non-empty segment to determine the shape
    seg_dim = None
    for seq in segments:
        if seq.shape[0] > 0:
            seg_dim = seq.shape[1] if len(seq.shape) > 1 else 1
            break

    if seg_dim is None:
        raise ValueError("Could not determine segment dimensions")

    batch_shape_segments = torch.full(
        (len(segments), max_len, seg_dim), padding_val)

    for i, seq in enumerate(segments):
        length = seq.shape[0]
        if length > 0:
            # Ensure the sequence has the expected dimensions
            if len(seq.shape) == 1:
                seq = seq.unsqueeze(1)  # Add dimension if needed
            elif seq.shape[1] != seg_dim:
                logging.warning(
                    f"Segment {i} has unexpected shape: {seq.shape}, expected dim {seg_dim}")
                continue
            batch_shape_segments[i, :length, ...] = seq

    batched_masks = torch.arange(max_len).expand(
        len(feats_lens), max_len) < feats_lens.unsqueeze(1)

    batched_masks = batched_masks.unsqueeze(1)

    return batch_shape_visual, batch_shape_audio, batch_shape_text, batched_masks, batch_shape_labels, batch_shape_segments


def collate_fn(batch):
    try:
        vis_feats = [torch.tensor(item['feats']['visual']) for item in batch]
        aud_feats = [torch.tensor(item['feats']['audio']) for item in batch]
        text_feats = [torch.tensor(item['feats']['text']) for item in batch]
        labels = [torch.tensor(item['labels']) for item in batch]
        segments = [torch.tensor(item['segments']) for item in batch]
        video_ids = [item['video_id'] for item in batch]
        durations = [item['duration'] for item in batch]

        # Debug logging for tensor shapes
        logger = logging.getLogger(__name__)
        logger.debug(f"Batch size: {len(batch)}")
        logger.debug(f"Visual shapes: {[v.shape for v in vis_feats]}")
        logger.debug(f"Audio shapes: {[a.shape for a in aud_feats]}")
        logger.debug(f"Text shapes: {[t.shape for t in text_feats]}")
        logger.debug(f"Label shapes: {[l.shape for l in labels]}")
        logger.debug(f"Segment shapes: {[s.shape for s in segments]}")
        logger.debug(f"Video IDs: {video_ids}")

        batched_inputs_visual, batched_inputs_audio, batched_inputs_text, batched_masks, batched_labels, batched_segments = preprocessing(
            vis_feats, aud_feats, text_feats, labels, segments)
        return {
            'video_id': video_ids,
            'duration': durations,
            'visual_feats': batched_inputs_visual,
            'audio_feats': batched_inputs_audio,
            'text_feats': batched_inputs_text,
            'masks': batched_masks,
            'labels': batched_labels,
            'segments': batched_segments,
        }
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error in collate_fn: {e}")
        logger.error(
            f"Batch video IDs: {[item.get('video_id', 'unknown') for item in batch]}")
        logger.error(
            f"Batch sizes: {[(item.get('video_id', 'unknown'), item.get('duration', 'unknown')) for item in batch]}")
        raise


class RepurposeClipTest(Dataset):
    def __init__(self, label_path, video_path, audio_path, text_path):
        self.label_path = label_path
        self.video_path = video_path
        self.audio_path = audio_path
        self.text_path = text_path

        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Load original labels
        original_labels = json.load(open(label_path))

        self.video_format = os.path.join(video_path, '{}.npy')
        self.audio_format = os.path.join(audio_path, '{}.npy')
        self.text_format = os.path.join(text_path, '{}.npy')

        # Filter labels to only include samples with all three modalities
        self.label = self._filter_available_samples(original_labels)

        self.video_ids = list(set([k['youtube_id'] for k in self.label]))

        for k in self.label:
            k['labels'] = self.generate_time_status_list(
                k['timeRangeOffset'], k['segmentsOffset'])
            k['reg_offset'] = self.generate_regression_offsets(
                k['timeRangeOffset'], k['segmentsOffset'])

    def _filter_available_samples(self, original_labels):
        """
        Filter dataset to only include samples where all three modalities (visual, audio, text) exist.

        Args:
            original_labels: List of original label dictionaries

        Returns:
            List of filtered label dictionaries
        """
        filtered_labels = []
        missing_visual = []
        missing_audio = []
        missing_text = []

        for label_item in original_labels:
            video_id = label_item['youtube_id']

            # Check if all three modality files exist
            visual_path = self.video_format.format(video_id)
            audio_path = self.audio_format.format(video_id)
            text_path = self.text_format.format(video_id)

            visual_exists = os.path.exists(visual_path)
            audio_exists = os.path.exists(audio_path)
            text_exists = os.path.exists(text_path)

            if visual_exists and audio_exists and text_exists:
                filtered_labels.append(label_item)
            else:
                # Track what's missing for detailed logging
                if not visual_exists:
                    missing_visual.append(video_id)
                if not audio_exists:
                    missing_audio.append(video_id)
                if not text_exists:
                    missing_text.append(video_id)

        # Log statistics
        total_original = len(original_labels)
        total_kept = len(filtered_labels)
        total_dropped = total_original - total_kept

        self.logger.info(f"Test dataset filtering results:")
        self.logger.info(f"  Original samples: {total_original}")
        self.logger.info(f"  Kept samples: {total_kept}")
        self.logger.info(f"  Dropped samples: {total_dropped}")
        self.logger.info(f"  Keep rate: {total_kept/total_original*100:.2f}%")

        if total_dropped > 0:
            self.logger.info(f"Missing modality breakdown:")
            self.logger.info(
                f"  Missing visual: {len(missing_visual)} samples")
            self.logger.info(f"  Missing audio: {len(missing_audio)} samples")
            self.logger.info(f"  Missing text: {len(missing_text)} samples")

            # Log some example missing IDs (first 5)
            if missing_visual:
                self.logger.debug(
                    f"  Example missing visual: {missing_visual[:5]}")
            if missing_audio:
                self.logger.debug(
                    f"  Example missing audio: {missing_audio[:5]}")
            if missing_text:
                self.logger.debug(
                    f"  Example missing text: {missing_text[:5]}")

        return filtered_labels

    def _validate_sample_data(self, label_item, visual_path, audio_path, text_path):
        """
        Validate that a sample has proper data that won't cause tensor errors.

        Args:
            label_item: Label dictionary for the sample
            visual_path: Path to visual features
            audio_path: Path to audio features  
            text_path: Path to text features

        Returns:
            bool: True if sample is valid, False otherwise
        """
        try:
            # Load the feature files
            visual_feats = np.load(visual_path, allow_pickle=True)
            audio_feats = np.load(audio_path, allow_pickle=True)
            text_feats = np.load(text_path, allow_pickle=True)

            # Check basic shape requirements
            if len(visual_feats.shape) != 2 or len(audio_feats.shape) != 2 or len(text_feats.shape) != 2:
                self.logger.debug(f"Invalid feature shapes for {label_item['youtube_id']}: "
                                  f"visual={visual_feats.shape}, audio={audio_feats.shape}, text={text_feats.shape}")
                return False

            # Check for empty features
            if visual_feats.shape[0] == 0 or audio_feats.shape[0] == 0 or text_feats.shape[0] == 0:
                self.logger.debug(f"Empty features for {label_item['youtube_id']}: "
                                  f"visual={visual_feats.shape[0]}, audio={audio_feats.shape[0]}, text={text_feats.shape[0]}")
                return False

            # Generate labels and regression offsets to check for validity
            labels = self.generate_time_status_list(
                label_item['timeRangeOffset'], label_item['segmentsOffset'])
            reg_offsets = self.generate_regression_offsets(
                label_item['timeRangeOffset'], label_item['segmentsOffset'])

            # Check that we have valid labels and offsets
            if len(labels) == 0 or len(reg_offsets) == 0:
                self.logger.debug(f"Empty labels or offsets for {label_item['youtube_id']}: "
                                  f"labels={len(labels)}, offsets={len(reg_offsets)}")
                return False

            # Check that regression offsets have proper shape
            if not isinstance(reg_offsets, list) or len(reg_offsets) > 0:
                if isinstance(reg_offsets[0], (list, tuple)) and len(reg_offsets[0]) != 2:
                    self.logger.debug(f"Invalid regression offset shape for {label_item['youtube_id']}: "
                                      f"expected 2D tuples, got {type(reg_offsets[0])}")
                    return False

            # Apply time range filtering to check final lengths
            timeRange = label_item['timeRange']
            if timeRange[0] != 0:
                visual_slice = visual_feats[int(
                    timeRange[0]):int(timeRange[1]), :]
                audio_slice = audio_feats[int(
                    timeRange[0]):int(timeRange[1]), :]
                text_slice = text_feats[int(timeRange[0]):int(timeRange[1]), :]
            else:
                visual_slice = visual_feats
                audio_slice = audio_feats
                text_slice = text_feats

            # Check minimum length after slicing
            min_len = min(visual_slice.shape[0], audio_slice.shape[0], text_slice.shape[0], len(
                labels), len(reg_offsets))
            if min_len <= 0:
                self.logger.debug(
                    f"Zero length after processing for {label_item['youtube_id']}: min_len={min_len}")
                return False

            return True

        except Exception as e:
            self.logger.debug(
                f"Validation error for {label_item['youtube_id']}: {e}")
            return False

    def generate_time_status_list(self, time_range, segments):
        """
        Generate a list where each second in the time range is marked as 1 if it falls within
        any of the given segments, and 0 otherwise. This version supports non-integer ranges and segments.

        :param time_range: A list [begin, end] representing the overall time range.
        :param segments: A list of lists, where each sublist represents a time segment [begin, end].
        :return: A list representing the status of each second in the time range.
        """
        # Determine the total number of seconds in the time range
        total_seconds = int(time_range[1] - time_range[0]) + 1

        # Initialize the list with 0s for each second in the time range
        status_list = [0] * total_seconds

        # Iterate over each segment and mark the corresponding seconds in the status list
        for segment in segments:
            start = max(int(segment[0]), int(time_range[0]))
            end = min(int(segment[1]), int(time_range[1]))
            for i in range(start, end + 1):
                status_list[i - int(time_range[0])] = 1
        return status_list

    def generate_regression_offsets(self, time_range, segments):
        """
        Generate regression offsets for each second in the time range.
        Each offset is a tuple (left_offset, right_offset) representing the distance
        to the segment's start and end if the second is within a segment. If a second is outside 
        any segments, offsets are set to a default value (e.g., float('inf')).

        :param time_range: A list [begin, end] representing the overall time range.
        :param segments: A list of lists, where each sublist represents a time segment [begin, end].
        :return: A list of tuples representing the regression offsets for each second in the time range.
        """
        # Handle edge cases
        if not time_range or len(time_range) != 2:
            if hasattr(self, 'logger'):
                self.logger.warning(f"Invalid time_range: {time_range}")
            return [(0.0, 0.0)]  # Return minimal valid data

        if time_range[1] <= time_range[0]:
            if hasattr(self, 'logger'):
                self.logger.warning(
                    f"Invalid time_range: end ({time_range[1]}) <= start ({time_range[0]})")
            return [(0.0, 0.0)]

        total_seconds = int(time_range[1] - time_range[0]) + 1
        if total_seconds <= 0:
            if hasattr(self, 'logger'):
                self.logger.warning(
                    f"Non-positive total_seconds: {total_seconds}")
            return [(0.0, 0.0)]

        regression_offsets = [(0.0, 0.0)] * total_seconds

        # Handle empty segments
        if not segments:
            return regression_offsets

        for second in range(total_seconds):
            current_time = time_range[0] + second

            for segment in segments:
                # Handle invalid segments
                if not segment or len(segment) != 2:
                    continue

                if segment[1] <= segment[0]:
                    continue  # Skip invalid segments

                if segment[0] <= current_time <= segment[1]:
                    # Current time is inside this segment
                    left_offset = float(current_time - segment[0])
                    right_offset = float(segment[1] - current_time)
                    regression_offsets[second] = (left_offset, right_offset)
                    break

        return regression_offsets

    def load_data(self, path):
        # Seq_len, feature_dim
        try:
            return np.load(path, allow_pickle=True)
        except FileNotFoundError:
            self.logger.error(f"Feature file not found: {path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading feature file {path}: {e}")
            raise

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        video_id = self.label[idx]['youtube_id']
        timeRange = self.label[idx]['timeRange']

        feats_visual = self.load_data(self.video_format.format(video_id))
        feats_audio = self.load_data(self.audio_format.format(video_id))
        feats_text = self.load_data(self.text_format.format(video_id))

        if timeRange[0] != 0:
            feats_visual = feats_visual[int(timeRange[0]):int(timeRange[1]), :]
            feats_audio = feats_audio[int(timeRange[0]):int(timeRange[1]), :]
            feats_text = feats_text[int(timeRange[0]):int(timeRange[1]), :]

        saliency_score = self.label[idx]['labels']
        reg_offset = self.label[idx]['reg_offset']

        # avoid audio and visual features have different lengths
        min_len = min(feats_visual.shape[0], feats_audio.shape[0], len(
            saliency_score), len(reg_offset))
        feats = {'visual': feats_visual[:min_len],
                 'audio': feats_audio[:min_len], 'text': feats_text[:min_len]}
        saliency_score = saliency_score[:min_len]
        reg_offset = reg_offset[:min_len]

        # return a data dict
        data_dict = {'video_id': video_id,
                     'feats': feats,      # seq_len, feature_dim
                     'segments': reg_offset,   # seq_len x 2
                     'labels': saliency_score,     # seq_len
                     'duration': min_len,
                     'gt_segments': self.label[idx]['segmentsOffset'],
                     }
        return data_dict


def collate_fn_test(batch):
    try:
        vis_feats = [torch.tensor(item['feats']['visual']) for item in batch]
        aud_feats = [torch.tensor(item['feats']['audio']) for item in batch]
        text_feats = [torch.tensor(item['feats']['text']) for item in batch]
        labels = [torch.tensor(item['labels']) for item in batch]
        segments = [torch.tensor(item['segments']) for item in batch]
        video_ids = [item['video_id'] for item in batch]
        durations = [item['duration'] for item in batch]
        gt_segments = [item['gt_segments'] for item in batch]

        # Debug logging for tensor shapes
        logger = logging.getLogger(__name__)
        logger.debug(f"Test batch size: {len(batch)}")
        logger.debug(f"Test visual shapes: {[v.shape for v in vis_feats]}")
        logger.debug(f"Test audio shapes: {[a.shape for a in aud_feats]}")
        logger.debug(f"Test text shapes: {[t.shape for t in text_feats]}")
        logger.debug(f"Test label shapes: {[l.shape for l in labels]}")
        logger.debug(f"Test segment shapes: {[s.shape for s in segments]}")
        logger.debug(f"Test video IDs: {video_ids}")

        batched_inputs_visual, batched_inputs_audio, batched_inputs_text, batched_masks, batched_labels, batched_segments = preprocessing(
            vis_feats, aud_feats, text_feats, labels, segments)
        return {
            'video_id': video_ids,
            'duration': durations,
            'visual_feats': batched_inputs_visual,
            'audio_feats': batched_inputs_audio,
            'text_feats': batched_inputs_text,
            'masks': batched_masks,
            'labels': batched_labels,
            'segments': batched_segments,
            'gt_segments': gt_segments,
        }
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error in collate_fn_test: {e}")
        logger.error(
            f"Test batch video IDs: {[item.get('video_id', 'unknown') for item in batch]}")
        logger.error(
            f"Test batch sizes: {[(item.get('video_id', 'unknown'), item.get('duration', 'unknown')) for item in batch]}")
        raise
