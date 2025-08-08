import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELNET_DIR = os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')

    os.makedirs(DATA_DIR, exist_ok = True)

    if not os.path.exists(MODELNET_DIR):
        print("ModelNet 40 Dataset not found, download it manually from https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip")
        print("Extract to:", MODELNET_DIR)

        raise FileNotFoundError("Missing Dataset: modelnet40_ply_hdf_2048")

def load_data(partition):
    assert partition in ('train', 'test')
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'modelnet40_ply_hdf5_2048')
    all_data, all_label = [], []

    pattern = f'ply_data_{partition}*.h5'
    for fn in glob.glob(os.path.join(DATA_DIR, pattern)):
        with h5py.File(fn, 'r') as f:
            all_data.append(f['data'][:].astype('float32'))
            all_label.append(f['label'][:].astype('int64'))
    points = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_label, axis=0).squeeze()
    return points, labels

def translate_pointcloud(pc):
    scales = np.random.uniform(2/3, 3/2, size=(3,))
    shifts = np.random.uniform(-0.2, 0.2, size=(3,))
    return (pc * scales + shifts).astype('float32')

def jitter_pointcloud(pc, sigma=0.01, clip=0.02):
    N, C = pc.shape
    noise = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return (pc + noise).astype('float32')

class ModelNet40PyG(Dataset):
    def __init__(self, num_points, partition='train'):
        super().__init__()
        assert partition in ('train', 'test')
        pts, lbls = load_data(partition)
        self.num_points = num_points
        self.partition = partition

        self.points = pts[:, :num_points, :]
        self.labels = lbls

    def __getitem__(self, idx):
        pc = self.points[idx].copy()
        label = int(self.labels[idx])
        if self.partition == 'train':
            pc = translate_pointcloud(pc)
            pc = jitter_pointcloud(pc)
            np.random.shuffle(pc)
        pos = torch.from_numpy(pc).float()
        y = torch.tensor(label, dtype = torch. long)
        return Data(pos=pos, y = y)

    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    print("Defines ModelNet40Pyg. Import it in the training script and wrap it with DataLoader.")
