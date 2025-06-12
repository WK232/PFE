import os
from torch.utils.data import Dataset
import tifffile as tiff
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

root_dir = '/home/kharratw/Documents/tessssst/PFE/ReadyToBeUsedDataset/12days/EcrinPark'

class CoherencePhaseSegmentationDataset(Dataset):
    root_dir = '/home/kharratw/Documents/tessssst/PFE/ReadyToBeUsedDataset/12days/EcrinPark'
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.samples = []
        self.transform = transform
        self.target_transform = target_transform
        self.root_dir = root_dir

        # Automatically detect the 'segmentations' folder regardless of case
        self.segmentation_root = os.path.join(
            root_dir,
            next(f for f in os.listdir(root_dir) if f.lower() == 'segmentations')
        )

        for date_folder in os.listdir(root_dir):
            date_path = os.path.join(root_dir, date_folder)
            if not os.path.isdir(date_path) or date_folder.lower() == 'segmentations':
                continue

            coherence_path = os.path.join(date_path, 'coherence')
            phase_path = os.path.join(date_path, 'phase')
            segmentation_path = os.path.join(self.segmentation_root, date_folder)

            if not all(os.path.isdir(p) for p in [coherence_path, phase_path, segmentation_path]):
                continue

            for fname in os.listdir(coherence_path):
                if not fname.lower().endswith('.tif'):
                    continue

                coh_file = os.path.join(coherence_path, fname)
                pha_file = os.path.join(phase_path, fname)
                seg_file = os.path.join(segmentation_path, fname)

                if os.path.exists(pha_file) and os.path.exists(seg_file):
                    self.samples.append((coh_file, pha_file, seg_file))

        print(f"✅ Found {len(self.samples)} valid sample triplets.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        coh_path, pha_path, seg_path = self.samples[idx]

        coherence = tiff.imread(coh_path).astype('float32')
        phase = tiff.imread(pha_path).astype('float32')
        segmentation = tiff.imread(seg_path).astype('int64')

        coherence = torch.tensor(coherence).unsqueeze(0)  # (1, H, W)
        phase = torch.tensor(phase).unsqueeze(0)          # (1, H, W)
        segmentation = torch.tensor(segmentation)         # (H, W)

        if self.transform:
            coherence = self.transform(coherence)
            phase = self.transform(phase)
        if self.target_transform:
            segmentation = self.target_transform(segmentation)

        return (coherence, phase), segmentation


from torch.utils.data import DataLoader

dataset = CoherencePhaseSegmentationDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for (coh, pha), seg in dataloader:
    print("Coherence shape:", coh.shape)  # (B, 1, H, W)
    print("Phase shape:", pha.shape)      # (B, 1, H, W)
    print("Segmentation shape:", seg.shape)  # (B, H, W)
    break
