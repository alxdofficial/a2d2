import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from model import Brain

def normalize_to_255(weights):
    # Normalize weights from [-3, 3] to [0, 255]
    normalized_weights = ((weights + 3) / 6) * 255
    return normalized_weights.astype(np.uint8)

def visualize_memory(model, frame_number, output_dir="visualizations"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_memory_networks = len(model.neural_memory_networks)
    num_subnetworks = model.neural_memory_networks[0].num_subnetworks
    num_layers = model.neural_memory_networks[0].num_layers

    fig, axes = plt.subplots(num_memory_networks, num_subnetworks, figsize=(40, 40))
    fig.suptitle(f'Weights Visualization for Frame {frame_number}', fontsize=10)

    for mem_idx, neural_memory_network in enumerate(model.neural_memory_networks):
        for sub_idx, subnetwork in enumerate(neural_memory_network.subnetworks):
            weight_grids = []
            recurrent_weight_grids = []
            for layer in subnetwork.layers:
                weights = layer.weight.detach().cpu().numpy()
                recurrent_weights = layer.recurrent_weight.detach().cpu().numpy()
                weight_grids.append(weights)
                recurrent_weight_grids.append(recurrent_weights)
                
            combined_weights = np.concatenate(weight_grids, axis=0)
            combined_recurrent_weights = np.concatenate(recurrent_weight_grids, axis=0)

            # Normalize weights to 0-255
            combined_weights = normalize_to_255(combined_weights)
            combined_recurrent_weights = normalize_to_255(combined_recurrent_weights)

            ax = axes[mem_idx, sub_idx]
            ax.imshow(combined_weights, aspect='auto', cmap='viridis')
            # ax.imshow(combined_recurrent_weights, aspect='auto', cmap='plasma', alpha=0.5)
            ax.set_title(f'Memory {mem_idx+1}, Subnet {sub_idx+1}', fontsize=8)
            ax.tick_params(axis='both', which='both', labelsize=6)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'weights_frame_{frame_number}.png')
    plt.savefig(output_path)
    plt.close(fig)
    print(f'Saved visualization for frame {frame_number} to {output_path}')
