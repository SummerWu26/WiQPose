import torch
import torch.nn as nn
import torch.nn.functional as F
from matcher import hungarian_matching_pose

def peak_weight_map(gt_heatmaps, thresh=0.2, peak_weight=5.0):
    w = torch.ones_like(gt_heatmaps)
    w[gt_heatmaps > thresh] = peak_weight
    return w

def peak_mse_loss(pred, gt, mask):
    w = peak_weight_map(gt)
    diff = ((pred - gt) ** 2) * w * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    B, P, K, H, W = pred.shape
    return diff.sum() / (mask.sum() * K * H * W + 1e-6)

class End2EndPoseLoss(nn.Module):
    def __init__(self, alpha_count=1.0, alpha_heatmap=10.0, alpha_conf=1.5,
                 focal_gamma=2.0, use_hungarian=True, heatmap_mode="mse"):
        super().__init__()
        self.alpha_count, self.alpha_heatmap, self.alpha_conf = alpha_count, alpha_heatmap, alpha_conf
        self.focal_gamma, self.use_hungarian, self.heatmap_mode = focal_gamma, use_hungarian, heatmap_mode
        self.count_criterion = nn.CrossEntropyLoss()

    def focal_loss(self, pred_logits, targets):
        bce = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction='none')
        p_t = torch.exp(-bce)
        return ((1 - p_t) ** self.focal_gamma * bce).mean()

    def forward(self, predictions, targets):
        device = predictions['count_logits'].device
        loss_count = self.count_criterion(predictions['count_logits'], targets['count'])

        pred_hm, pred_conf, gt_hm, gt_valid = (predictions['pred_heatmaps'], predictions['pred_conf_logits'],
                                               targets['heatmaps'], targets['mask'])

        if self.use_hungarian and gt_valid.sum() > 0:
            match_idx, match_valid = hungarian_matching_pose(pred_hm, gt_hm, gt_valid)
            B, P, K, H, W = pred_hm.shape
            matched_gt = torch.zeros_like(pred_hm)
            for b in range(B):
                for p in range(P): matched_gt[b, p] = gt_hm[b, match_idx[b, p]]
            final_mask, final_gt = match_valid, matched_gt
        else:
            final_mask, final_gt = gt_valid, gt_hm

        loss_heatmap = torch.tensor(0.0, device=device)
        if final_mask.sum() > 0:
            loss_heatmap = peak_mse_loss(pred_hm, final_gt, final_mask)

        loss_conf = self.focal_loss(pred_conf, final_mask.float())
        total = self.alpha_count * loss_count + self.alpha_heatmap * loss_heatmap + self.alpha_conf * loss_conf
        return total, {'total': total.item(), 'count': loss_count.item(), 'heatmap': loss_heatmap.item(), 'conf': loss_conf.item()}