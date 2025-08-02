#!/usr/bin/env python3
"""
Content-based duplicate-aware evaluation for entity matching.

This module provides a shared evaluation function that accepts predictions
where the predicted record has identical content to the ground truth record,
even if the IDs differ. This handles cases where datasets have duplicate
records with different IDs.
"""

from typing import Dict, List, Tuple, Any


def records_have_same_content(record1: dict, record2: dict, ignore_fields: List[str] = None) -> bool:
    """
    Check if two records have identical content (ignoring specified fields).

    Args:
        record1: First record to compare
        record2: Second record to compare
        ignore_fields: List of field names to ignore (default: ["id"])

    Returns:
        bool: True if records have identical content
    """
    if record1 is None or record2 is None:
        return False

    if ignore_fields is None:
        ignore_fields = ["id"]

    filtered1 = {k: str(v).strip() if v is not None else None
                for k, v in record1.items() if k not in ignore_fields}
    filtered2 = {k: str(v).strip() if v is not None else None
                for k, v in record2.items() if k not in ignore_fields}
    return filtered1 == filtered2


def duplicate_aware_evaluate(
    predictions: Dict[int, int],
    pairs_data: List[Tuple[int, int, int]],  # (left_id, right_id, true_label)
    B_records: Dict[int, dict],
    verbose: bool = False
) -> Tuple[List[int], List[int]]:
    """
    Evaluate predictions with content-based duplicate awareness.

    Args:
        predictions: Dict mapping left_id -> predicted_right_id
        pairs_data: List of (left_id, right_id, true_label) tuples
        B_records: Dict mapping right_id -> record dict
        verbose: Whether to print duplicate-aware matches found

    Returns:
        Tuple of (predicted_labels, true_labels) for metric calculation
    """
    preds = []
    labels = []

    for left_id, right_id, true_label in pairs_data:
        # Check if we predicted a match and if it's correct
        if left_id in predictions:
            pred_right_id = predictions[left_id]

            # Standard exact ID match
            if pred_right_id == right_id:
                pred_label = 1
            # Content-based duplicate-aware match
            elif true_label == 1:  # Only check content for true positive cases
                pred_record = B_records.get(pred_right_id)
                true_record = B_records.get(right_id)
                if records_have_same_content(pred_record, true_record):
                    pred_label = 1  # Accept as correct match
                    if verbose:
                        print(f"  ✅ Duplicate-aware match: predicted ID {pred_right_id} matches content of ground truth ID {right_id}")
                else:
                    pred_label = 0
            else:
                pred_label = 0
        else:
            pred_label = 0  # No match predicted

        preds.append(pred_label)
        labels.append(true_label)

    return preds, labels


def duplicate_aware_detailed_evaluate(
    predictions: Dict[int, int],
    pairs_data: List[Tuple[int, int, int]],  # (left_id, right_id, true_label)
    A_records: Dict[int, dict],
    B_records: Dict[int, dict],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate predictions with detailed individual record results for failure analysis.

    Args:
        predictions: Dict mapping left_id -> predicted_right_id
        pairs_data: List of (left_id, right_id, true_label) tuples
        A_records: Dict mapping left_id -> record dict
        B_records: Dict mapping right_id -> record dict
        verbose: Whether to print duplicate-aware matches found

    Returns:
        Dict containing:
        - preds: List of predicted labels
        - labels: List of true labels  
        - individual_results: List of detailed results for each pair
        - false_positives: List of false positive records
        - false_negatives: List of false negative records
        - true_positives: List of true positive records
        - true_negatives: List of true negative records
    """
    preds = []
    labels = []
    individual_results = []
    false_positives = []
    false_negatives = []
    true_positives = []
    true_negatives = []

    for left_id, right_id, true_label in pairs_data:
        # Get the left record
        left_record = A_records[left_id]
        
        # Check if we predicted a match and if it's correct
        pred_right_id = predictions.get(left_id)
        
        if pred_right_id is not None:
            # Standard exact ID match
            if pred_right_id == right_id:
                pred_label = 1
            # Content-based duplicate-aware match
            elif true_label == 1:  # Only check content for true positive cases
                pred_record = B_records.get(pred_right_id)
                true_record = B_records.get(right_id)
                if records_have_same_content(pred_record, true_record):
                    pred_label = 1  # Accept as correct match
                    if verbose:
                        print(f"  ✅ Duplicate-aware match: predicted ID {pred_right_id} matches content of ground truth ID {right_id}")
                else:
                    pred_label = 0
            else:
                pred_label = 0
        else:
            pred_label = 0  # No match predicted

        preds.append(pred_label)
        labels.append(true_label)

        # Create detailed result for this pair
        result = {
            'left_id': left_id,
            'right_id': right_id,
            'pred_right_id': pred_right_id,
            'true_label': true_label,
            'pred_label': pred_label,
            'left_record': left_record,
            'true_right_record': B_records[right_id],
            'pred_right_record': B_records.get(pred_right_id) if pred_right_id is not None else None
        }
        individual_results.append(result)

        # Categorize the result
        if pred_label == 1 and true_label == 0:
            # False Positive
            false_positives.append({
                "left_record": left_record,
                "predicted_record": B_records.get(pred_right_id),
                "actual_record": B_records[right_id],
                "left_id": left_id,
                "pred_right_id": pred_right_id,
                "true_right_id": right_id
            })
        elif pred_label == 0 and true_label == 1:
            # False Negative
            false_negatives.append({
                "left_record": left_record,
                "missed_record": B_records[right_id],
                "left_id": left_id,
                "true_right_id": right_id,
                "pred_right_id": pred_right_id
            })
        elif pred_label == 1 and true_label == 1:
            # True Positive
            true_positives.append({
                "left_record": left_record,
                "matched_record": B_records[right_id],
                "left_id": left_id,
                "right_id": right_id,
                "pred_right_id": pred_right_id
            })
        elif pred_label == 0 and true_label == 0:
            # True Negative
            true_negatives.append({
                "left_record": left_record,
                "unmatched_record": B_records[right_id],
                "left_id": left_id,
                "right_id": right_id
            })

    return {
        "preds": preds,
        "labels": labels,
        "individual_results": individual_results,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "summary": {
            "total_fp": len(false_positives),
            "total_fn": len(false_negatives),
            "total_tp": len(true_positives),
            "total_tn": len(true_negatives)
        }
    }


def calculate_metrics(preds: List[int], labels: List[int]) -> Dict[str, Any]:
    """
    Calculate standard classification metrics from predictions and labels.

    Args:
        preds: List of predicted labels (0 or 1)
        labels: List of true labels (0 or 1)

    Returns:
        Dict containing tp, fp, fn, tn, precision, recall, f1, accuracy
    """
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(preds) if len(preds) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": tp + fp + fn + tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }
