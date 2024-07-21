import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DriveDataset
from model import Brain
from visualizememory import visualize_memory
from tqdm import tqdm

def main_worker():
    # Hyperparameters
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 1  # As specified, batch size is always 1
    INTERNAL_DIM = 4096
    FINAL_OUTPUT_DIM = 9  # Adjusted to 9 to include the 6 fovea coordinates
    NUM_SUB_NETWORKS = 8
    NUM_LAYERS = 5
    NUM_MEMORY_NETWORKS = 24
    EMOTIONAL_HIDDEN_SIZE = 128
    NUM_NEUROTRANSMITTERS = 4  # Number of neurotransmitters
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
        final_output_dim=FINAL_OUTPUT_DIM,  # Now 9 values
        sub_network_dim=sub_network_dim,
        num_layers=NUM_LAYERS,
        emotional_hidden_size=EMOTIONAL_HIDDEN_SIZE,
        num_sub_networks=NUM_SUB_NETWORKS,
        num_memory_networks=NUM_MEMORY_NETWORKS,
        num_neurotransmitters=NUM_NEUROTRANSMITTERS
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    for epoch in range(EPOCHS):
        epoch_loss = 0
        with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}/{EPOCHS}') as pbar:
            # Initialize fovea_coords to be the center of the image
            fovea_coords = [
                IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2,
                IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2,
                IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2
            ]
            for i, (images, input_data, labels) in enumerate(dataloader):
                # Move inputs to GPU
                image_1, image_2, image_3 = [img.to(device) for img in images]
                input_data = input_data.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(image_1, image_2, image_3, input_data, fovea_coords)

                # Separate the first 3 values for loss calculation and the latter 6 values as new fovea coordinates
                predictions = outputs[:, :3]
                fovea_coords_pred = outputs[:, 3:]

                # Compute loss using the first 3 values
                loss = criterion(predictions, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate loss
                epoch_loss += loss.item()
 
                # Update Hebbian weights
                model.update_neuro_modulators()

                # Update fovea coordinates for the next step (convert from percentage to pixel values and clip)
                new_fovea_coords = []
                for j in range(fovea_coords_pred.size(1)):
                    if j % 2 == 0:  # x-coordinate
                        coord = (fovea_coords_pred[:, j] * IMAGE_WIDTH).int().clamp(0, IMAGE_WIDTH - 1)
                    else:  # y-coordinate
                        coord = (fovea_coords_pred[:, j] * IMAGE_HEIGHT).int().clamp(0, IMAGE_HEIGHT - 1)
                    new_fovea_coords.append(coord.item())

                fovea_coords = new_fovea_coords

                # Print GPU memory usage
                gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024 * 1024 * 1024)  # Convert to GB
                gpu_memory_reserved = torch.cuda.memory_reserved(device) / (1024 * 1024 * 1024)    # Convert to GB

                pbar.set_postfix({
                    'Loss': loss.item(),
                    'GPU Memory Allocated (GB)': gpu_memory_allocated,
                    'GPU Memory Reserved (GB)': gpu_memory_reserved
                })
                pbar.update(1)

        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss / len(dataloader):.4f}')

    # Save the model checkpoint
    torch.save(model.state_dict(), 'brain_model.pth')
    print('Model saved to brain_model.pth')

if __name__ == '__main__':
    main_worker()
