import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from preprocessor import CSIPreprocessor

def generate_heatmaps(keypoints: List[List[float]], image_size: Tuple[int, int],
                      heatmap_size: Tuple[int, int], sigma: float = 2.0) -> np.ndarray:
    num_keypoints = len(keypoints)
    img_w, img_h = image_size
    hm_w, hm_h = heatmap_size
    heatmaps = np.zeros((num_keypoints, hm_h, hm_w), dtype=np.float32)
    y_grid, x_grid = np.meshgrid(np.arange(hm_h), np.arange(hm_w), indexing='ij')

    for i in range(num_keypoints):
        if keypoints[i][0] < 0 or keypoints[i][1] < 0:
            continue
        mu_x = keypoints[i][0] * (hm_w / img_w)
        mu_y = keypoints[i][1] * (hm_h / img_h)
        exponent = ((x_grid - mu_x) ** 2 + (y_grid - mu_y) ** 2) / (2 * sigma ** 2)
        heatmaps[i, :, :] = np.exp(-exponent)
    return heatmaps

class WiFiPoseDataset(Dataset):
    def __init__(self, data_session_dir: str, csi_preprocessor: CSIPreprocessor,
                 image_size: Tuple[int, int] = (1920, 1080), heatmap_size: Tuple[int, int] = (56, 56),
                 max_persons: int = 3, num_keypoints: int = 17,
                 negative_sample_ratio: float = 0.15, augment: bool = False):
        self.data_session_dir = data_session_dir
        self.preprocessor = csi_preprocessor
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.max_persons = max_persons
        self.num_keypoints = num_keypoints
        self.negative_sample_ratio = negative_sample_ratio
        self.augment = augment

        self.mags_dir = os.path.join(data_session_dir, 'mags')
        self.mmpose_dir = os.path.join(data_session_dir, 'mmpose')
        self.mag_files = sorted([f for f in os.listdir(self.mags_dir) if f.endswith('.npy')])
        self.pose_files = sorted([f for f in os.listdir(self.mmpose_dir) if f.endswith('.json')])
        assert len(self.mag_files) == len(self.pose_files)

    def __len__(self):
        return len(self.mag_files)

    def __getitem__(self, idx):
        csi_data = np.load(os.path.join(self.mags_dir, self.mag_files[idx])).astype(np.float32)
        if self.augment:
            if np.random.rand() > 0.5: csi_data += np.random.normal(0, 0.01, csi_data.shape).astype(np.float32)
            if np.random.rand() > 0.5: csi_data *= np.random.uniform(0.8, 1.2)
            if np.random.rand() > 0.5 and csi_data.shape[1] > 80:
                start = np.random.randint(0, csi_data.shape[1] - 80)
                csi_data = csi_data[:, start:start+80, :, :]
                pad = np.zeros((csi_data.shape[0], 34, csi_data.shape[2], csi_data.shape[3]))
                csi_data = np.concatenate([csi_data, pad], axis=1)
            if np.random.rand() > 0.5:
                phase_shift = np.random.uniform(-0.1, 0.1, csi_data.shape)
                csi_data = csi_data * np.exp(1j * phase_shift).real

        csi_data = self.preprocessor.transform(csi_data)

        with open(os.path.join(self.mmpose_dir, self.pose_files[idx]), 'r') as f:
            pose_annotations = json.load(f)

        if np.random.rand() < self.negative_sample_ratio:
            num_persons = 0
            heatmaps = np.zeros((self.max_persons, self.num_keypoints, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)
            person_valid = np.zeros(self.max_persons, dtype=np.float32)
        else:
            num_persons = min(len(pose_annotations), self.max_persons)
            heatmaps = np.zeros((self.max_persons, self.num_keypoints, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)
            person_valid = np.zeros(self.max_persons, dtype=np.float32)
            for i, p_anno in enumerate(pose_annotations[:self.max_persons]):
                kpts = np.array(p_anno['keypoints'], dtype=np.float32)
                heatmaps[i] = generate_heatmaps(kpts, self.image_size, self.heatmap_size, sigma=3.0)
                person_valid[i] = 1.0

        return {'csi': csi_data, 'num_persons': num_persons, 'heatmaps': heatmaps, 'person_valid': person_valid}

def pose_collate_fn(batch):
    return {
        'csi': torch.stack([torch.from_numpy(b['csi']).float() for b in batch]),
        'count': torch.tensor([int(b['num_persons']) for b in batch], dtype=torch.long),
        'heatmaps': torch.stack([torch.from_numpy(b['heatmaps']).float() for b in batch]),
        'mask': torch.stack([torch.from_numpy(b['person_valid']).bool() for b in batch])
    }