import torch
import torch.nn as nn
from typing import Dict, Tuple

class QueryAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, queries):
        attn_out, _ = self.attn(queries, queries, queries)
        return self.norm(queries + attn_out)

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, queries, memory):
        attn_out, _ = self.attn(query=queries, key=memory, value=memory)
        return self.norm(queries + attn_out)

class WiFiEnd2EndPoseNet(nn.Module):
    def __init__(self, max_persons: int = 3, num_keypoints: int = 17,
                 hidden_dim: int = 512, heatmap_size: Tuple[int, int] = (56, 56)):
        super().__init__()
        self.max_persons = max_persons
        self.num_keypoints = num_keypoints
        self.hidden_dim = hidden_dim
        self.heatmap_size = heatmap_size

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3,5,3), stride=(1,2,1), padding=(1,2,1)),
            nn.BatchNorm3d(32), nn.ReLU(True), nn.MaxPool3d((2,2,1)),
            nn.Conv3d(32, 64, kernel_size=(3,5,3), stride=(1,2,1), padding=(1,2,1)),
            nn.BatchNorm3d(64), nn.ReLU(True), nn.MaxPool3d((2,2,1)),
            nn.Conv3d(64, 128, kernel_size=(3,5,3), stride=(1,2,1), padding=(1,2,1)),
            nn.BatchNorm3d(128), nn.ReLU(True), nn.AdaptiveAvgPool3d((4, 8, 1))
        )

        self.query_embed = nn.Embedding(max_persons, hidden_dim)
        self.query_attention = QueryAttention(dim=hidden_dim, num_heads=4)
        self.memory_proj = nn.Linear(128, hidden_dim)
        self.cross_attn = CrossAttentionBlock(dim=hidden_dim, num_heads=4)

        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(True), nn.Linear(256, max_persons + 1)
        )

        self.pose_decoder = nn.ModuleList([self._make_pose_decoder() for _ in range(max_persons)])
        self.conf_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(True), nn.Linear(256, 1)
        )

    def _make_pose_decoder(self):
        return nn.Sequential(
            nn.Linear(self.hidden_dim, 1024), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(1024, 2048), nn.ReLU(True), nn.Dropout(0.3),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=9, stride=1, padding=0), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, self.num_keypoints, kernel_size=1)
        )

    def forward(self, csi: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = csi.size(0)
        x = csi.view(B, 1, 16, 114, 9)
        x = self.encoder(x)
        B, C, T, F, _ = x.shape
        memory = x.view(B, C, T*F).permute(0, 2, 1)
        memory = self.memory_proj(memory)

        global_feat = memory.mean(dim=1)
        count_logits = self.count_head(global_feat)

        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        queries = self.query_attention(queries)
        queries = self.cross_attn(queries, memory)

        heatmaps_list, conf_logits_list = [], []
        for i in range(self.max_persons):
            query_feat = queries[:, i, :]
            heatmaps_list.append(self.pose_decoder[i](query_feat))
            conf_logits_list.append(self.conf_head(query_feat).squeeze(-1))

        return {
            'count_logits': count_logits,
            'pred_heatmaps': torch.stack(heatmaps_list, dim=1),
            'pred_conf_logits': torch.stack(conf_logits_list, dim=1)
        }