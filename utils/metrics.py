def calculate_ap(segments, labels):
    """
    Calculate the Average Precision (AP) for given predictions and segments.

    :param preds: A list of predictions.
    :param segments: A list of segments, each segment is a list of [t1, t2].
    :return: The Average Precision (AP) value.
    """
    # Initialize labels with zeros
    perds = [0] * len(labels)

    # Set labels to 1 for predictions that fall within any segment
    for segment in segments:
        # Find the index range for the segment
        start_index = int(segment[0]) if int(segment[0])>=0 else 0
        end_index = int(segment[1]) if int(segment[1])<len(labels) else len(labels)-1

        # Update labels within the segment range
        if start_index is not None and end_index is not None:
            for i in range(start_index, end_index + 1):
                perds[i] = 1

    # Calculate AP
    n_positives = sum(labels)
    if n_positives == 0:
        return 0.0

    cum_positive = 0
    cum_total = 0
    precision_sum = 0

    for i, pred in enumerate(perds):
        cum_total += 1
        if pred == 1 and labels[i] == 1:
            cum_positive += 1
            precision = cum_positive / cum_total
            precision_sum += precision

    ap = precision_sum / n_positives
    return ap


def calculate_recall(segments, labels):
    """
    Calculate the Recall for given predictions and segments.

    :param segments: A list of segments, each segment is a list of [t1, t2].
    :param labels: A list of labels indicating the ground truth.
    :return: The Recall value.
    """
    # Initialize labels with zeros
    preds = [0] * len(labels)

    # Set labels to 1 for predictions that fall within any segment
    for segment in segments:
        # Find the index range for the segment
        start_index = int(segment[0]) if int(segment[0]) >= 0 else 0
        end_index = int(segment[1]) if int(segment[1]) < len(labels) else len(labels) - 1

        # Update labels within the segment range
        if start_index is not None and end_index is not None:
            for i in range(start_index, end_index + 1):
                preds[i] = 1

    # Calculate True Positives (TP) and Total Positives (TP + FN)
    tp = 0
    total_positives = 0

    for pred, label in zip(preds, labels):
        if pred == 1 and label == 1:
            tp += 1
        if label == 1:
            total_positives += 1

    # Calculate Recall
    if total_positives == 0:
        return 0.0

    recall = tp / total_positives
    return recall

def calculate_tiou(reference_segments, predicted_segments, tiou_thresholds=[0.5]):
    """
    Calculate the temporal Intersection over Union (tIoU) for given reference and predicted segments, considering a list of thresholds.

    :param reference_segments: A list of reference segments, each segment is a tuple of (start, end).
    :param predicted_segments: A list of predicted segments, each segment is a tuple of (start, end).
    :param tiou_thresholds: A list of threshold values for tIoU above which a prediction is considered valid.
    :return: A dictionary where keys are the tIoU thresholds and the values are the precision values for each threshold.
    """
    def calculate_iou(segment1, segment2):
        start_max = max(segment1[0], segment2[0])
        end_min = min(segment1[1], segment2[1])

        intersection = max(0, end_min - start_max)
        union = (segment1[1] - segment1[0]) + (segment2[1] - segment2[0]) - intersection

        return intersection / union if union != 0 else 0
    
    # Calculate the maximum tIoU for each predicted segment
    max_tiou_scores = [max([calculate_iou(predicted_segment, ref_segment) for ref_segment in reference_segments], default=0) 
                       for predicted_segment in predicted_segments]
    
    # Calculate precision for each threshold
    precision_per_threshold = {}
    for threshold in tiou_thresholds:
        valid_predictions = sum(score >= threshold for score in max_tiou_scores)
        precision = valid_predictions / len(predicted_segments) if len(predicted_segments) > 0 else 0
        precision_per_threshold[threshold] = precision
    
    return precision_per_threshold