from torch.utils.data import Dataset
import torch
import numpy as np
import os
import json

class RepurposeClip(Dataset):
    def __init__(self, label_path, video_path, audio_path, text_path):
        self.label_path = label_path
        self.video_path = video_path
        self.audio_path = audio_path
        self.text_path = text_path

        self.label = json.load(open(label_path))

        self.video_ids = list(set([k['youtube_id'] for k in self.label]))

        self.video_format = os.path.join(video_path, '{}.npy')
        self.audio_format = os.path.join(audio_path, '{}.npy')
        self.text_format = os.path.join(text_path, '{}.npy')

        for k in self.label:
            k['labels'] = self.generate_time_status_list(k['timeRangeOffset'], k['segmentsOffset'])
            k['reg_offset'] = self.generate_regression_offsets(k['timeRangeOffset'], k['segmentsOffset'])


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
        total_seconds = int(time_range[1] - time_range[0]) + 1
        regression_offsets = [(0, 0)] * total_seconds

        for second in range(total_seconds):
            current_time = time_range[0] + second

            for segment in segments:
                if segment[0] <= current_time <= segment[1]:
                    # Current time is inside this segment
                    left_offset = current_time - segment[0]
                    right_offset = segment[1] - current_time
                    regression_offsets[second] = (left_offset, right_offset)
                    break

        return regression_offsets

    def load_data(self, path):
        # Seq_len, feature_dim
        return np.load(path, allow_pickle=True)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        video_id = self.label[idx]['youtube_id']
        timeRange = self.label[idx]['timeRange']

        feats_visual = self.load_data(self.video_format.format(video_id))
        feats_audio = self.load_data(self.audio_format.format(video_id))
        feats_text = self.load_data(self.text_format.format(video_id))

        if timeRange[0] != 0:
            feats_visual = feats_visual[int(timeRange[0]):int(timeRange[1]),:]
            feats_audio = feats_audio[int(timeRange[0]):int(timeRange[1]),:]
            feats_text = feats_text[int(timeRange[0]):int(timeRange[1]),:]

        saliency_score = self.label[idx]['labels']
        reg_offset = self.label[idx]['reg_offset']

        #avoid audio and visual features have different lengths
        min_len = min(feats_visual.shape[0], feats_audio.shape[0], len(saliency_score), len(reg_offset))
        feats = {'visual': feats_visual[:min_len], 'audio': feats_audio[:min_len] , 'text': feats_text[:min_len]}
        saliency_score = saliency_score[:min_len]
        reg_offset = reg_offset[:min_len]
    
        # return a data dict
        data_dict = {'video_id'        : video_id,
                     'feats'           : feats,      # seq_len, feature_dim
                     'segments'        : reg_offset,   # seq_len x 2
                     'labels'          : saliency_score,     # seq_len
                     'duration'        : min_len,
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

    feats_lens = torch.as_tensor([feat_visual.shape[0] for feat_visual in feats_visual])
    max_len = feats_lens.max().item()

    batch_shape_visual = torch.full((len(feats_visual), max_len, feats_visual[0].shape[1]), padding_val)
    for i, seq in enumerate(feats_visual):
        length = seq.shape[0]
        batch_shape_visual[i, :length, ...] = seq
    
    batch_shape_audio = torch.full((len(feats_audio), max_len, feats_audio[0].shape[1]), padding_val)
    for i, seq in enumerate(feats_audio):
        length = seq.shape[0]
        batch_shape_audio[i, :length, ...] = seq

    batch_shape_text = torch.full((len(feats_text), max_len, feats_text[0].shape[1]), padding_val)
    for i, seq in enumerate(feats_text):
        length = seq.shape[0]
        batch_shape_text[i, :length, ...] = seq

    batch_shape_labels = torch.full((len(labels), max_len), padding_val)
    for i, seq in enumerate(labels):
        length = seq.shape[0]
        batch_shape_labels[i, :length] = seq

    batch_shape_segments = torch.full((len(segments), max_len, segments[0].shape[1]), padding_val)

    for i, seq in enumerate(segments):
        length = seq.shape[0]
        batch_shape_segments[i, :length, ...] = seq

    batched_masks = torch.arange(max_len).expand(len(feats_lens), max_len) < feats_lens.unsqueeze(1)

    batched_masks = batched_masks.unsqueeze(1)

    return batch_shape_visual, batch_shape_audio, batch_shape_text, batched_masks, batch_shape_labels, batch_shape_segments

def collate_fn(batch):
    vis_feats = [torch.tensor(item['feats']['visual']) for item in batch]
    aud_feats = [torch.tensor(item['feats']['audio']) for item in batch]
    text_feats = [torch.tensor(item['feats']['text']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]
    segments = [torch.tensor(item['segments']) for item in batch]
    video_ids = [item['video_id'] for item in batch]
    durations = [item['duration'] for item in batch]

    batched_inputs_visual, batched_inputs_audio, batched_inputs_text, batched_masks, batched_labels, batched_segments = preprocessing(vis_feats, aud_feats, text_feats, labels, segments)
    return {
        'video_id': video_ids,
        'duration': durations,
        'visual_feats': batched_inputs_visual,
        'audio_feats': batched_inputs_audio,
        'text_feats': batched_inputs_text,
        'masks': batched_masks,
        'labels': batched_labels, # 
        'segments': batched_segments,
    }


class RepurposeClipTest(Dataset):
    def __init__(self, label_path, video_path, audio_path, text_path):
        self.label_path = label_path
        self.video_path = video_path
        self.audio_path = audio_path
        self.text_path = text_path

        self.label = json.load(open(label_path))
        self.video_ids = list(set([k['youtube_id'] for k in self.label]))

        self.video_format = os.path.join(video_path, '{}.npy')
        self.audio_format = os.path.join(audio_path, '{}.npy')
        self.text_format = os.path.join(text_path, '{}.npy')

        for k in self.label:
            k['labels'] = self.generate_time_status_list(k['timeRangeOffset'], k['segmentsOffset'])
            k['reg_offset'] = self.generate_regression_offsets(k['timeRangeOffset'], k['segmentsOffset'])


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
        total_seconds = int(time_range[1] - time_range[0]) + 1
        regression_offsets = [(0, 0)] * total_seconds

        for second in range(total_seconds):
            current_time = time_range[0] + second

            for segment in segments:
                if segment[0] <= current_time <= segment[1]:
                    # Current time is inside this segment
                    left_offset = current_time - segment[0]
                    right_offset = segment[1] - current_time
                    regression_offsets[second] = (left_offset, right_offset)
                    break

        return regression_offsets

    def load_data(self, path):
        # Seq_len, feature_dim
        return np.load(path, allow_pickle=True)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        video_id = self.label[idx]['youtube_id']
        timeRange = self.label[idx]['timeRange']

        feats_visual = self.load_data(self.video_format.format(video_id))
        feats_audio = self.load_data(self.audio_format.format(video_id))
        feats_text = self.load_data(self.text_format.format(video_id))

        if timeRange[0] != 0:
            feats_visual = feats_visual[int(timeRange[0]):int(timeRange[1]),:]
            feats_audio = feats_audio[int(timeRange[0]):int(timeRange[1]),:]
            feats_text = feats_text[int(timeRange[0]):int(timeRange[1]),:]

        saliency_score = self.label[idx]['labels']
        reg_offset = self.label[idx]['reg_offset']

        #avoid audio and visual features have different lengths
        min_len = min(feats_visual.shape[0], feats_audio.shape[0], len(saliency_score), len(reg_offset))
        feats = {'visual': feats_visual[:min_len], 'audio': feats_audio[:min_len] , 'text': feats_text[:min_len]}
        saliency_score = saliency_score[:min_len]
        reg_offset = reg_offset[:min_len]
    
        # return a data dict
        data_dict = {'video_id'        : video_id,
                     'feats'           : feats,      # seq_len, feature_dim
                     'segments'        : reg_offset,   # seq_len x 2
                     'labels'          : saliency_score,     # seq_len
                     'duration'        : min_len,
                     'gt_segments'     : self.label[idx]['segmentsOffset'],
                     }
        return data_dict


def collate_fn_test(batch):
    vis_feats = [torch.tensor(item['feats']['visual']) for item in batch]
    aud_feats = [torch.tensor(item['feats']['audio']) for item in batch]
    text_feats = [torch.tensor(item['feats']['text']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]
    segments = [torch.tensor(item['segments']) for item in batch]
    video_ids = [item['video_id'] for item in batch]
    durations = [item['duration'] for item in batch]
    gt_segments = [item['gt_segments'] for item in batch]

    batched_inputs_visual, batched_inputs_audio, batched_inputs_text, batched_masks, batched_labels, batched_segments = preprocessing(vis_feats, aud_feats, text_feats, labels, segments)
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
