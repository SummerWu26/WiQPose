import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def keypoint_similarity(pred_heatmap, gt_heatmap):
    mse = torch.mean((pred_heatmap - gt_heatmap) ** 2)
    return -mse.item()

def hungarian_matching_pose(pred_heatmaps, gt_heatmaps, valid_mask):
    B, max_persons, _, _, _ = pred_heatmaps.shape
    device = pred_heatmaps.device
    matched_gt_indices = torch.zeros((B, max_persons), dtype=torch.long, device=device)
    matched_valid = torch.zeros((B, max_persons), dtype=torch.bool, device=device)

    for b in range(B):
        cost_matrix = np.zeros((max_persons, max_persons))
        pred_b, gt_b, valid_b = pred_heatmaps[b].cpu(), gt_heatmaps[b].cpu(), valid_mask[b].cpu()
        for i in range(max_persons):
            for j in range(max_persons):
                cost_matrix[i, j] = -keypoint_similarity(pred_b[i], gt_b[j]) if valid_b[j] else 1e6

        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
        for p_idx, g_idx in zip(pred_indices, gt_indices):
            if valid_b[g_idx]:
                matched_gt_indices[b, p_idx] = g_idx
                matched_valid[b, p_idx] = True
    return matched_gt_indices, matched_valid