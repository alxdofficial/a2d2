import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch

class DriveDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.image_paths = self.data[['cam_front_path', 'cam_left_path', 'cam_right_path']]
        self.input_data = self.data[['acceleration_x', 'acceleration_y', 'roll_angle']]
        self.labels = self.data[['accelerator_pedal', 'brake_pressure', 'steering_angle_calculated']]
        
        # Initialize scalers
        self.input_scalers = {col: MinMaxScaler() for col in self.input_data.columns}
        self.label_scalers = {col: MinMaxScaler() for col in self.labels.columns}
        
        # Fit scalers on the input data and normalize
        for col in self.input_data.columns:
            self.input_data.loc[:, col] = self.input_scalers[col].fit_transform(self.input_data[[col]]).astype(float)
        
        # Fit scalers on the labels and normalize
        for col in self.labels.columns:
            self.labels.loc[:, col] = self.label_scalers[col].fit_transform(self.labels[[col]]).astype(float)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Load and normalize images
        img_paths = self.image_paths.iloc[idx]
        images = []
        for path in img_paths:
            full_path = os.path.join(self.root_dir, path)
            if "camera_front_left" in path:
                full_path = full_path.replace("camera_frontcenter", "camera_frontleft")
            elif "camera_front_right" in path:
                full_path = full_path.replace("camera_frontcenter", "camera_frontright")
            images.append(np.array(Image.open(full_path)) / 255.0)
        images = [torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32) for img in images]
        
        # Load normalized input data
        input_data = torch.tensor(self.input_data.iloc[idx].values, dtype=torch.float32)
        
        # Load normalized labels
        labels = torch.tensor(self.labels.iloc[idx].values, dtype=torch.float32)
        
        return images, input_data, labels

def undo_normalization(numerical_data, scalers):
    original_data = {}
    for idx, col in enumerate(scalers.keys()):
        original_data[col] = scalers[col].inverse_transform(numerical_data[:, idx].reshape(-1, 1)).flatten()
    return original_data
