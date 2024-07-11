import torch
import torch.nn as nn
import torch.nn.functional as F
# Define the input and output dimensions for internal modules
INTERNAL_DIM = 4096
FINAL_OUTPUT_DIM = 9
NUM_SUB_NETWORKS = 16
NUM_LAYERS = 12

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
    


class HebbianLayer(nn.Module):
    def __init__(self, input_size, output_size, scaling, decay):
        super(HebbianLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_size, input_size))
        self.recurrent_weight = nn.Parameter(torch.randn(output_size, output_size))
        self.alpha = scaling  # Scaling parameter for Hebbian learning
        self.decay = decay  # Decay parameter

    def forward(self, x, prev_state):
        combined_input = x + torch.mm(prev_state.detach(), self.recurrent_weight.detach())
        activations = F.relu(torch.mm(combined_input, self.weight.detach().t()))
        return activations, activations

    def hebbian_update(self, activations, x, prev_state, dopamine_signal):
        with torch.no_grad():
            # Hebbian update for input weights
            hebbian_term_input = torch.mm(activations.t(), x)
            self.weight += self.alpha * dopamine_signal * hebbian_term_input

            # Hebbian update for recurrent weights
            hebbian_term_recurrent = torch.mm(activations.t(), prev_state)
            self.recurrent_weight += self.alpha * dopamine_signal * hebbian_term_recurrent

            # Apply decay to input weights
            self.weight -= self.decay * (self.weight * (activations.sum(dim=0, keepdim=True) == 0).float().t())

            # Apply decay to recurrent weights
            self.recurrent_weight -= self.decay * (self.recurrent_weight * (activations.sum(dim=0, keepdim=True) == 0).float().t())

class SubNetwork(nn.Module):
    def __init__(self, input_size, output_size, num_layers):
        super(SubNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            scaling = torch.FloatTensor(1).uniform_(0.1, 1).item()
            decay = torch.FloatTensor(1).uniform_(0.01, 0.2).item()
            self.layers.append(HebbianLayer(input_size, output_size, scaling, decay))

    def forward(self, x, prev_state):
        hidden_states = []
        new_state = prev_state
        for layer in self.layers:
            hidden_state, new_state = layer(x, new_state)
            hidden_states.append(hidden_state)
        return hidden_states, new_state

    def hebbian_update(self, hidden_states, x, prev_state, dopamine_signal):
        for i, layer in enumerate(self.layers):
            layer.hebbian_update(hidden_states[i], x, prev_state, dopamine_signal)

class DopamineNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_subnetworks):
        super(DopamineNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_subnetworks)

    def forward(self, prev_loss, current_loss, activations):
        loss_diff = prev_loss - current_loss
        x = torch.cat([loss_diff.unsqueeze(0), activations.flatten().unsqueeze(0)], dim=-1)
        x = F.relu(self.fc1(x))
        dopamine_signals = torch.tanh(self.fc2(x))  # Output in range [-1, 1]
        return dopamine_signals
    
class NeuralMemoryNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_subnetworks, num_layers, dopamine_hidden_size):
        super(NeuralMemoryNetwork, self).__init__()
        self.subnetworks = nn.ModuleList()
        self.hidden_size = hidden_size
        self.num_subnetworks = num_subnetworks
        for _ in range(num_subnetworks):
            self.subnetworks.append(SubNetwork(input_size, hidden_size, num_layers))

        self.output_layer = nn.Linear(hidden_size * num_subnetworks, output_size)
        self.dopamine_network = DopamineNetwork(hidden_size * num_subnetworks + 1, dopamine_hidden_size, num_subnetworks)
        self.prev_loss = None

    def forward(self, x, prev_states):
        x_splits = x.split(self.hidden_size, dim=-1)
        all_hidden_states = []
        new_states = []
        activations = []
        for i, subnetwork in enumerate(self.subnetworks):
            hidden_states, new_state = subnetwork(x_splits[i], prev_states[i])
            all_hidden_states.append(hidden_states)
            new_states.append(new_state)
            activations.append(hidden_states[-1])

        combined_activations = torch.cat(activations, dim=-1)
        combined_output = torch.cat([hs[-1] for hs in all_hidden_states], dim=-1)
        final_output = self.output_layer(combined_output)
        return final_output, new_states, all_hidden_states, combined_activations

    def hebbian_update(self, all_hidden_states, x, prev_states, current_loss, combined_activations):
        if self.prev_loss is None:
            self.prev_loss = current_loss

        dopamine_signals = self.dopamine_network(self.prev_loss, current_loss, combined_activations)

        for i, subnetwork in enumerate(self.subnetworks):
            subnetwork.hebbian_update(all_hidden_states[i], x.split(self.hidden_size, dim=-1)[i], prev_states[i], dopamine_signals[0, i])

        self.prev_loss = current_loss


        
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



class NumericFeatureAdaptor(nn.Module):
    def __init__(self, input_size, output_size):
        super(NumericFeatureAdaptor, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return F.relu(self.fc(x))



class Brain(nn.Module):
    def __init__(self, internal_dim, final_output_dim, sub_network_dim, num_layers, dopamine_hidden_size):
        super(Brain, self).__init__()
        
        self.internal_dim = internal_dim
        self.final_output_dim = final_output_dim
        
        # Vision encoders
        self.vision_encoder_1 = VisionEncoder()
        self.vision_encoder_2 = VisionEncoder()
        self.vision_encoder_3 = VisionEncoder()
        
        # Numeric feature adaptor
        self.numeric_feature_adaptor = NumericFeatureAdaptor(input_size=3, output_size=internal_dim)
        
        # Neural memory network
        self.neural_memory = NeuralMemoryNetwork(input_size=internal_dim, 
                                                 hidden_size=sub_network_dim, 
                                                 output_size=internal_dim, 
                                                 num_subnetworks=internal_dim // sub_network_dim, 
                                                 num_layers=num_layers, 
                                                 dopamine_hidden_size=dopamine_hidden_size)
        
        # Action decoder
        self.action_decoder = ActionDecoder(internal_dim=internal_dim, final_output_dim=final_output_dim)
        
        # Initialize previous states for the neural memory network
        self.prev_states = [torch.zeros(1, sub_network_dim) for _ in range(internal_dim // sub_network_dim)]

    def forward(self, image_1, image_2, image_3, accel_data, fovea_coords):
        fovea_x1, fovea_y1, fovea_x2, fovea_y2, fovea_x3, fovea_y3 = fovea_coords
        
        # Process images through their respective vision encoders
        fovea_output_1, peripheral_output_1 = self.vision_encoder_1(image_1, fovea_x1, fovea_y1)
        fovea_output_2, peripheral_output_2 = self.vision_encoder_2(image_2, fovea_x2, fovea_y2)
        fovea_output_3, peripheral_output_3 = self.vision_encoder_3(image_3, fovea_x3, fovea_y3)
        
        # Process accelerometer data through the numeric feature adaptor
        accel_output = self.numeric_feature_adaptor(accel_data)
        
        # Combine encodings using max pooling
        combined_encoding = torch.max(torch.stack([fovea_output_1, peripheral_output_1, 
                                                   fovea_output_2, peripheral_output_2, 
                                                   fovea_output_3, peripheral_output_3, 
                                                   accel_output]), dim=0)[0]
        
        # Pass combined encoding through the neural memory network
        memory_output, new_states, all_hidden_states, combined_activations = self.neural_memory(combined_encoding.unsqueeze(0), self.prev_states)
        
        # Update previous states for the next forward pass
        self.prev_states = new_states
        
        # Pass the output of the neural memory network through the action decoder
        final_output = self.action_decoder(memory_output.squeeze(0))
        
        return final_output
