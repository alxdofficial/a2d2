import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DriveDataset
from model import Brain

# Hyperparameters
EPOCHS = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 1  # As specified, batch size is always 1
INTERNAL_DIM = 4096
FINAL_OUTPUT_DIM = 3  # Predicting 3 values: accelerator_pedal, brake_pressure, steering_angle_calculated
NUM_SUB_NETWORKS = 8
NUM_LAYERS = 5
NUM_MEMORY_NETWORKS = 24
DOPAMINE_HIDDEN_SIZE = 128
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1208
# Paths
CSV_FILE = '/home/alex/Documents/vscodeprojects/personal/a2d2research/src/data manifest/a2d2train.csv'
ROOT_DIR = '/media/alex/39eb242f-14ca-4925-8a70-35633885bff4/A2D2'

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset and DataLoader
dataset = DriveDataset(CSV_FILE, ROOT_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model, Loss, Optimizer
sub_network_dim = INTERNAL_DIM // NUM_SUB_NETWORKS

model = Brain(
    img_width=IMAGE_WIDTH,
    img_height=IMAGE_HEIGHT,
    internal_dim=INTERNAL_DIM,
    final_output_dim=FINAL_OUTPUT_DIM,
    sub_network_dim=sub_network_dim,
    num_layers=NUM_LAYERS,
    dopamine_hidden_size=DOPAMINE_HIDDEN_SIZE,
    num_sub_networks=NUM_SUB_NETWORKS,
    num_memory_networks=NUM_MEMORY_NETWORKS
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
for epoch in range(EPOCHS):
    epoch_loss = 0
    for i, (images, input_data, labels) in enumerate(dataloader):
        # Move inputs to GPU
        image_1, image_2, image_3 = [img.to(device) for img in images]
        input_data = input_data.to(device)
        labels = labels.to(device)

        # Initialize fovea_coords to be the center of the image
        fovea_coords = [
            IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2,
            IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2,
            IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2
        ]

        # Forward pass
        outputs = model(image_1, image_2, image_3, input_data, fovea_coords)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        epoch_loss += loss.item()

        # Update Hebbian weights
        model.hebbian_update(loss.item(), outputs)

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
    
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss / len(dataloader):.4f}')

# Save the model checkpoint
torch.save(model.state_dict(), 'brain_model.pth')
print('Model saved to brain_model.pth')
