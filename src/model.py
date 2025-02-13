import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the input and output dimensions for internal modules
INTERNAL_DIM = 4096
FINAL_OUTPUT_DIM = 9
NUM_SUB_NETWORKS = 8
NUM_LAYERS = 5
NUM_MEMORY_NETWORKS = 24
DOPAMINE_HIDDEN_SIZE = 128

# 1. Feature Extractor
# The FeatureExtractor class consists of convolutional layers with batch normalization, ReLU activation, and dropout for feature extraction from images.
class FeatureExtractor(nn.Module):
    def __init__(self, input_channels=3, num_layers=5, kernel_size=5, stride=3, dropout=0.3):
        super(FeatureExtractor, self).__init__()
        channels = [16, 64, 256, 512, 1024]  # Specified number of channels for each layer
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(input_channels, channels[i], kernel_size=kernel_size, stride=stride))
            if i != num_layers - 1:
                layers.append(nn.BatchNorm2d(channels[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_channels = channels[i]
        self.extractor = nn.Sequential(*layers)

    def forward(self, x):
        return self.extractor(x)
# 2. Fully Connected Layer
# The FullyConnected class creates a fully connected network with ReLU activation and dropout.
class FullyConnected(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super(FullyConnected, self).__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.LayerNorm(hidden_dim))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)
# 3. Vision Encoder
# The VisionEncoder class uses the FeatureExtractor and FullyConnected layers to process images and generate feature vectors.    
class VisionEncoder(nn.Module):
    def __init__(self, img_width, img_height):
        super(VisionEncoder, self).__init__()
        self.img_width = img_width
        self.img_height = img_height
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
        x = min(max(x, half_crop), self.img_width - half_crop)
        y = min(max(y, half_crop), self.img_height - half_crop)
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

# 4. Hebbian Layer
# The HebbianLayer class defines a layer with Hebbian learning, where weights are updated based on the activity of neurons and the presence of a neuro modulators.    
class HebbianLayer(nn.Module):
    def __init__(self, input_size, output_size, scaling, decay):
        super(HebbianLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_size, input_size))
        self.recurrent_weight = nn.Parameter(torch.randn(output_size, output_size))
        self.alpha = scaling  # Scaling parameter for Hebbian learning
        self.decay = decay  # Decay parameter
        self.prev_state = None
        self.prev_input = None
        self.layer_norm = nn.LayerNorm(output_size)  # Layer normalization for the activations
        self.max_magnitude = 3

        # Neurotransmitter signals
        self.dopamine_signal = 0.0
        self.serotonin_signal = 0.0
        self.gaba_signal = 0.0
        self.glutamate_signal = 0.0

    def forward(self, x):
        if self.prev_state is None:
            self.prev_state = torch.zeros(x.size(0), self.weight.size(0), device=x.device)

        prev_state_detached = self.prev_state.detach()
        combined_input = x + torch.mm(prev_state_detached, self.recurrent_weight.detach())
        
        activations = torch.mm(combined_input, self.weight.detach().t())
        activations = self.layer_norm(activations)  # Apply layer normalization

        # Apply GABA and glutamate modulation
        threshold = torch.mean(activations)  # Base threshold
        threshold += self.gaba_signal  # Increase threshold with GABA
        threshold -= self.glutamate_signal  # Decrease threshold with glutamate
        activations = torch.relu(activations - threshold)  # Apply threshold and ReLU
        activations = self.layer_norm(activations)  # Apply layer normalization

        self.prev_input = x
        self.prev_state = activations

        self.hebbian_update()
        return activations

    def hebbian_update(self):
        with torch.no_grad():
            # Hebbian update for input weights
            hebbian_term_input = torch.mm(self.prev_state.t(), self.prev_input)
            weight_update = self.alpha * self.dopamine_signal * hebbian_term_input
            self.weight.data += weight_update

            # Hebbian update for recurrent weights
            hebbian_term_recurrent = torch.mm(self.prev_state.t(), self.prev_state)
            recurrent_weight_update = self.alpha * self.dopamine_signal * hebbian_term_recurrent
            self.recurrent_weight.data += recurrent_weight_update

            # Apply decay to all input weights
            self.weight.data -= self.decay * self.weight.data

            # Apply decay to all recurrent weights
            self.recurrent_weight.data -= self.decay * self.recurrent_weight.data

            # Normalize weights to have range -3 to 3
            self.weight.data = nn.functional.normalize(self.weight.data, p=2, dim=1) * self.max_magnitude
            self.recurrent_weight.data = nn.functional.normalize(self.recurrent_weight.data, p=2, dim=1) * self.max_magnitude

    def update_neuro_modulators(self, dopamine, serotonin, gaba, glutamate):
        # Update dopamine with serotonin modulation
        change_in_dopamine = dopamine - self.dopamine_signal
        self.dopamine_signal += change_in_dopamine * (1 - serotonin)

        # Update other neurotransmitters
        self.serotonin_signal = serotonin
        self.gaba_signal = gaba
        self.glutamate_signal = glutamate


# 5. SubNetwork
# The SubNetwork class stacks multiple HebbianLayers to create a more complex subnetwork.
class SubNetwork(nn.Module):
    def __init__(self, input_size, output_size, num_layers):
        super(SubNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            scaling = torch.FloatTensor(1).uniform_(0.0001, 0.05).item()
            decay = torch.FloatTensor(1).uniform_(0.0001, 0.01).item()
            self.layers.append(HebbianLayer(input_size, output_size, scaling, decay))
            input_size = output_size  # Update input size for the next layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def update_neuro_modulators(self, neurotransmitter_signals):
        for i, layer in enumerate(self.layers):
            dopamine_signal = neurotransmitter_signals[i, 0]
            serotonin_signal = neurotransmitter_signals[i, 1]
            gaba_signal = neurotransmitter_signals[i, 2]
            glutamate_signal = neurotransmitter_signals[i, 3]

            layer.update_neuro_modulators(dopamine_signal, serotonin_signal, gaba_signal, glutamate_signal)


# 6. EmotionalModule
# The EmotionalModule class generates dopamine,serotonin,gaba,etc signals based on the sensort input and the activations from the networks.
class EmotionalModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_subnetworks, num_layers, num_neurotransmitters):
        super(EmotionalModule, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_subnetworks * num_layers * num_neurotransmitters)
        self.num_layers = num_layers
        self.num_neurotransmitters = num_neurotransmitters

    def forward(self, combined_sensory_encoding, activations):
        x = torch.cat([combined_sensory_encoding, activations.flatten().unsqueeze(0)], dim=-1)  # Concatenate along the feature dimension
        x = F.relu(self.fc1(x))
        neurotransmitter_signals = torch.tanh(self.fc2(x))  # Output in range [-1, 1]
        return neurotransmitter_signals.view(-1, self.num_layers, self.num_neurotransmitters)  # Shape: (num_subnetworks, num_layers, num_neurotransmitters)

# 7. Neural Memory Network
# The NeuralMemoryNetwork class combines multiple subnetworks and an emotional module to form a memory network that can update its weights based on neurotransmitter signals.    
class NeuralMemoryNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_subnetworks, num_layers, emotional_hidden_size, num_neurotransmitters):
        super(NeuralMemoryNetwork, self).__init__()
        self.subnetworks = nn.ModuleList()
        self.hidden_size = hidden_size
        self.num_subnetworks = num_subnetworks
        self.num_layers = num_layers
        self.num_neurotransmitters = num_neurotransmitters
        for _ in range(num_subnetworks):
            self.subnetworks.append(SubNetwork(hidden_size, hidden_size, num_layers))

        self.output_layer = nn.Linear(INTERNAL_DIM, output_size)
        self.emotional_module = EmotionalModule(INTERNAL_DIM + INTERNAL_DIM, emotional_hidden_size, num_subnetworks, num_layers, num_neurotransmitters)
        self.layer_norm = nn.LayerNorm(output_size)  # Apply layer normalization

        self.prev_loss = None
        self.activations = None

    def forward(self, x):
        x_splits = x.split(self.hidden_size, dim=-1)
        activations_of_all_subnetworks = []
        for i, subnetwork in enumerate(self.subnetworks):
            activations_of_all_subnetworks.append(subnetwork(x_splits[i].squeeze(0)))

        # Combine final activations for the final output
        combined_activations = torch.cat(activations_of_all_subnetworks, dim=-1)
        
        # Use combined activations to generate the final output
        final_output = self.output_layer(combined_activations)
        final_output = self.layer_norm(final_output)
        self.activations = final_output
        return final_output

    def update_neuro_modulators(self, combined_sensory_encoding):
        # Get activations
        activations = self.activations.detach()

        # Compute neurotransmitter signals
        neurotransmitter_signals = self.emotional_module(combined_sensory_encoding, activations)

        # Iterate through each subnetwork and update neurotransmitters
        for i, subnetwork in enumerate(self.subnetworks):
            subnetwork.update_neuro_modulators(neurotransmitter_signals[i])


# 8. Action Decoder
# The ActionDecoder class reduces the internal representation to the final output using a series of fully connected layers.
class ActionDecoder(nn.Module):
    def __init__(self, internal_dim, final_output_dim):
        super(ActionDecoder, self).__init__()
        
        # Intermediate layers to create abstract action intent
        self.abstract_layers = nn.Sequential(
            nn.Linear(internal_dim, internal_dim),
            nn.LayerNorm(internal_dim),
            nn.ReLU(),
            nn.Linear(internal_dim, internal_dim),
            nn.LayerNorm(internal_dim),
            nn.ReLU(),
            nn.Linear(internal_dim, internal_dim),
            nn.LayerNorm(internal_dim),
            nn.ReLU()
        )
        
        # Gradually reducing layers to produce final output
        self.reduction_layers = nn.Sequential(
            nn.Linear(internal_dim, internal_dim // 2),
            nn.LayerNorm(internal_dim // 2),
            nn.ReLU(),
            nn.Linear(internal_dim // 2, internal_dim // 4),
            nn.LayerNorm(internal_dim // 4),
            nn.ReLU(),
            nn.Linear(internal_dim // 4, internal_dim // 8),
            nn.LayerNorm(internal_dim // 8),
            nn.ReLU(),
            nn.Linear(internal_dim // 8, internal_dim // 16),
            nn.LayerNorm(internal_dim // 16),
            nn.ReLU(),
            nn.Linear(internal_dim // 16, final_output_dim)
        )
        
    def forward(self, x):
        # Pass through abstract action intent layers
        x = self.abstract_layers(x)  # Shape: (batch_size, internal_dim)
        
        # Pass through reduction layers to get final output
        x = self.reduction_layers(x)  # Shape: (batch_size, final_output_dim)
        
        return x
# 9. Numeric Feature Adaptor
# The NumericFeatureAdaptor class adapts numeric input features into the internal dimensionality.
class NumericFeatureAdaptor(nn.Module):
    def __init__(self, input_size, output_size):
        super(NumericFeatureAdaptor, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return F.relu(self.fc(x))
# 10. Attention Module
# The AttentionModule class applies an attention mechanism to combine multilpe sources of embeddings into 1
class AttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, query_input, key_value_inputs):
        # Prepare for attention mechanism
        key_value_tensor = torch.stack(key_value_inputs)  # Shape: (num_sources, batch_size, internal_dim)
        # Query input
        query = query_input.unsqueeze(1)  # Shape: (1, batch_size, internal_dim)
        

        # print(key_value_tensor.shape, query.shape)
        # Apply attention with query input, and key_value_inputs as key and value
        attn_output, _ = self.attention(query, key_value_tensor, key_value_tensor)
        
        # Remove the singleton dimension from the output
        attn_output = attn_output.squeeze(1)  # Shape: (batch_size, internal_dim)

        return attn_output

# 11. Brain
# The Brain class integrates all the components to simulate a neural network inspired by the human brain, combining vision processing,
# numeric feature adaptation, memory networks, and attention mechanisms.
class Brain(nn.Module):
    def __init__(self, img_width, img_height, internal_dim, final_output_dim, sub_network_dim, num_layers, emotional_hidden_size, num_sub_networks, num_memory_networks, num_neurotransmitters):
        super(Brain, self).__init__()

        self.internal_dim = internal_dim
        self.final_output_dim = final_output_dim

        # Vision encoders
        self.vision_encoder_1 = VisionEncoder(img_width, img_height)
        self.vision_encoder_2 = VisionEncoder(img_width, img_height)
        self.vision_encoder_3 = VisionEncoder(img_width, img_height)

        # Numeric feature adaptor
        self.numeric_feature_adaptor = NumericFeatureAdaptor(input_size=3, output_size=internal_dim)

        # Neural memory networks
        self.neural_memory_networks = nn.ModuleList([
            NeuralMemoryNetwork(
                input_size=internal_dim,
                hidden_size=sub_network_dim,
                output_size=internal_dim,
                num_subnetworks=num_sub_networks,
                num_layers=num_layers,
                emotional_hidden_size=emotional_hidden_size,
                num_neurotransmitters=num_neurotransmitters
            ) for _ in range(num_memory_networks)
        ])

        # Attention mechanism
        self.attention = AttentionModule(embed_dim=internal_dim, num_heads=8)

        # Action decoder
        self.action_decoder = ActionDecoder(internal_dim=internal_dim, final_output_dim=final_output_dim)

        self.combined_sensory_encoding = None

    def forward(self, image_1, image_2, image_3, accel_data, fovea_coords):
        fovea_x1, fovea_y1, fovea_x2, fovea_y2, fovea_x3, fovea_y3 = fovea_coords

        # Process images through their respective vision encoders
        fovea_output_1, peripheral_output_1 = self.vision_encoder_1(image_1, fovea_x1, fovea_y1)
        fovea_output_2, peripheral_output_2 = self.vision_encoder_2(image_2, fovea_x2, fovea_y2)
        fovea_output_3, peripheral_output_3 = self.vision_encoder_3(image_3, fovea_x3, fovea_y3)

        # Process accelerometer data through the numeric feature adaptor
        accel_output = self.numeric_feature_adaptor(accel_data)

        # Combine encodings of images and numeric input using attention
        image_encodings = [fovea_output_1, fovea_output_2, fovea_output_3, peripheral_output_1, peripheral_output_2, peripheral_output_3]
        combined_encoding = self.attention(accel_output, image_encodings).squeeze(1).detach()
        self.combined_sensory_encoding = combined_encoding  # save combined_sensory_encoding for emotional module in hebbian update

        # Use CUDA streams to parallelize NeuralMemoryNetwork execution
        streams = [torch.cuda.Stream() for _ in range(len(self.neural_memory_networks))]
        final_outputs_of_all_memory_networks = []

        for i, (neural_memory_network, stream) in enumerate(zip(self.neural_memory_networks, streams)):
            with torch.cuda.stream(stream):
                memory_output = neural_memory_network(combined_encoding.unsqueeze(0).to('cuda'))
                final_outputs_of_all_memory_networks.append(memory_output)

        torch.cuda.synchronize()  # Ensure all streams are finished

        # Apply attention mechanism on combined activations from memory networks
        attn_output = self.attention(combined_encoding, final_outputs_of_all_memory_networks).detach()
        # Pass the output of the attention mechanism through the action decoder
        final_output = self.action_decoder(attn_output)
        return final_output

    def update_neuro_modulators(self):
        for neural_memory_network in self.neural_memory_networks:
            neural_memory_network.update_neuro_modulators(self.combined_sensory_encoding.detach())