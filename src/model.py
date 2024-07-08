import torch
import torch.nn as nn
import torch.nn.functional as F
# Define the input and output dimensions for internal modules
INTERNAL_DIM = 4096
FINAL_OUTPUT_DIM = 5
SUB_NETWORK_DIM = 64
NUM_SUB_NETWORKS = INTERNAL_DIM // SUB_NETWORK_DIM
NUM_LAYERS = 12
NUM_HEADS = 1

class FeatureExtractor(nn.Module):
    def __init__(self, input_channels=3, num_layers=5, kernel_size=5, stride=3, dropout=0.3):
        super(FeatureExtractor, self).__init__()
        channels = [16, 64, 256, 512, 1024]  # Specified number of channels for each layer
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(input_channels, channels[i], kernel_size=kernel_size, stride=stride))
            layers.append(nn.BatchNorm2d(channels[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_channels = channels[i]
        self.extractor = nn.Sequential(*layers)

    def forward(self, x):
        return self.extractor(x)

class FullyConnected(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super(FullyConnected, self).__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)
    
class VisionEncoder(nn.Module):
    def __init__(self):
        super(VisionEncoder, self).__init__()
        
        # Feature extractors
        self.fovea_feature_extractor = FeatureExtractor()
        self.peripheral_feature_extractor = FeatureExtractor()
        
        # Fully connected layers
        flattened_dim = 1024 * 1 * 1  # Output from feature extractor after flattening
        hidden_dims = [2048]
        output_dim = INTERNAL_DIM
        
        self.fovea_fc = FullyConnected(flattened_dim, hidden_dims, output_dim)
        self.peripheral_fc = FullyConnected(flattened_dim, hidden_dims, output_dim)

    def forward(self, image, x, y):
        # Crop the fovea image
        crop_size = 512
        half_crop = crop_size // 2
        x = min(max(x, half_crop), 1920 - half_crop)
        y = min(max(y, half_crop), 1920 - half_crop)
        fovea_image = image[:, :, y - half_crop:y + half_crop, x - half_crop:x + half_crop]

        # Resize the entire image
        peripheral_image = F.interpolate(image, size=(512, 512), mode='bilinear', align_corners=False)

        # Fovea feature extraction
        fovea_features = self.fovea_feature_extractor(fovea_image)
        fovea_features = fovea_features.view(fovea_features.size(0), -1)  # Flatten
        fovea_output = self.fovea_fc(fovea_features)

        # Peripheral feature extraction
        peripheral_features = self.peripheral_feature_extractor(peripheral_image)
        peripheral_features = peripheral_features.view(peripheral_features.size(0), -1)  # Flatten
        peripheral_output = self.peripheral_fc(peripheral_features)

        return fovea_output, peripheral_output 

class MemoryModule(nn.Module):
    def __init__(self, memory_size):
        super(MemoryModule, self).__init__()
        self.memory_size = memory_size
        self.memory_weights = nn.ParameterList([nn.Parameter(torch.randn(SUB_NETWORK_DIM, SUB_NETWORK_DIM)) for _ in range(memory_size)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(SUB_NETWORK_DIM) for _ in range(memory_size)])

    def forward(self, x):
        with torch.no_grad():
            for weight, layer_norm in zip(self.memory_weights, self.layer_norms):
                x = torch.relu(x @ weight.T)  # Apply ReLU activation
                x = layer_norm(x)  # Apply Layer Normalization
        return x  # Shape: (batch_size, SUB_NETWORK_DIM)

class AttentionMechanism(nn.Module):
    def __init__(self, num_heads):
        super(AttentionMechanism, self).__init__()
        self.query_projection = nn.Linear(SUB_NETWORK_DIM, SUB_NETWORK_DIM)
        self.key_projection = nn.Linear(SUB_NETWORK_DIM, SUB_NETWORK_DIM)
        self.value_projection = nn.Linear(SUB_NETWORK_DIM, SUB_NETWORK_DIM)
        self.attention = nn.MultiheadAttention(SUB_NETWORK_DIM, num_heads)

    def forward(self, queries, keys, values):
        # Project queries, keys, and values
        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)
        
        # queries, keys, values: (seq_len, batch_size, SUB_NETWORK_DIM)
        attn_output, attn_weights = self.attention(queries, keys, values)
        return attn_output, attn_weights

class NeuralMemoryNetwork(nn.Module):
    def __init__(self, memory_size, num_heads):
        super(NeuralMemoryNetwork, self).__init__()
        self.memory_module = MemoryModule(memory_size)
        self.attention_mechanism = AttentionMechanism(num_heads)

    def forward(self, x):
        with torch.no_grad():
            memory_outputs = self.memory_module(x)  # Shape: (batch_size, SUB_NETWORK_DIM)
        memory_outputs = memory_outputs.unsqueeze(1)  # Shape: (batch_size, 1, SUB_NETWORK_DIM)
        
        # Step 1: Calculate the average of the weights for each neuron in layer L
        avg_weights = torch.stack([weight.mean(dim=1) for weight in self.memory_module.memory_weights], dim=0)
        # avg_weights: Shape (memory_size, SUB_NETWORK_DIM)
        avg_weights = avg_weights.unsqueeze(0)  # Shape: (1, memory_size, SUB_NETWORK_DIM)
        
        # Step 2: Compute the attention between the memory output and avg_weights
        attn_output, attn_weights = self.attention_mechanism(avg_weights, memory_outputs, memory_outputs)
        
        # Step 3: Apply the weight change proportionally
        attn_output = attn_output.squeeze(0)  # Shape: (memory_size, SUB_NETWORK_DIM)
        
        for i, weight in enumerate(self.memory_module.memory_weights):
            avg_weight_per_neuron = weight.mean(dim=1)  # Shape: (SUB_NETWORK_DIM)
            delta = attn_output[i]  # Shape: (SUB_NETWORK_DIM)
            
            # Calculate the proportional change for each weight
            weight_update = (weight.T * delta / avg_weight_per_neuron).T  # Shape: (SUB_NETWORK_DIM, SUB_NETWORK_DIM)
            weight.data += weight_update.data

        # Step 5: Use the memory output as the final output
        output = memory_outputs.squeeze(1)  # Shape: (batch_size, SUB_NETWORK_DIM)
        
        return output

class MasterNeuralMemoryNetwork(nn.Module):
    def __init__(self, num_heads):
        super(MasterNeuralMemoryNetwork, self).__init__()
        assert INTERNAL_DIM == SUB_NETWORK_DIM * NUM_SUB_NETWORKS, "INTERNAL_DIM must be equal to SUB_NETWORK_DIM * NUM_SUB_NETWORKS"
        self.sub_memory_networks = nn.ModuleList([NeuralMemoryNetwork(NUM_LAYERS, num_heads) for _ in range(NUM_SUB_NETWORKS)])

    def forward(self, x):
        # Split the input into smaller vectors of shape (batch_size, SUB_NETWORK_DIM)
        sub_inputs = x.split(SUB_NETWORK_DIM, dim=1)  # List of (batch_size, SUB_NETWORK_DIM)

        # Process all sub-inputs through corresponding NeuralMemoryNetwork
        sub_outputs = [self.sub_memory_networks[i](sub_inputs[i]) for i in range(NUM_SUB_NETWORKS)]
        
        # Combine the outputs back into one vector of shape (batch_size, INTERNAL_DIM)
        combined_output = torch.cat(sub_outputs, dim=1)  # Shape: (batch_size, INTERNAL_DIM)

        return combined_output


class ActionDecoder(nn.Module):
    def __init__(self, internal_dim, final_output_dim):
        super(ActionDecoder, self).__init__()
        
        # Intermediate layers to create abstract action intent
        self.abstract_layers = nn.Sequential(
            nn.Linear(internal_dim, internal_dim),
            nn.ReLU(),
            nn.Linear(internal_dim, internal_dim),
            nn.ReLU(),
            nn.Linear(internal_dim, internal_dim),
            nn.ReLU()
        )
        
        # Gradually reducing layers to produce final output
        self.reduction_layers = nn.Sequential(
            nn.Linear(internal_dim, internal_dim // 2),
            nn.ReLU(),
            nn.Linear(internal_dim // 2, internal_dim // 4),
            nn.ReLU(),
            nn.Linear(internal_dim // 4, internal_dim // 8),
            nn.ReLU(),
            nn.Linear(internal_dim // 8, internal_dim // 16),
            nn.ReLU(),
            nn.Linear(internal_dim // 16, final_output_dim)
        )
        
    def forward(self, x):
        # Pass through abstract action intent layers
        x = self.abstract_layers(x)  # Shape: (batch_size, internal_dim)
        
        # Pass through reduction layers to get final output
        x = self.reduction_layers(x)  # Shape: (batch_size, final_output_dim)
        
        return x
