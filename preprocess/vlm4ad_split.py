import json
import random

# Load the original JSON data
input_path = "/home/alex/Documents/vscodeprojects/personal/a2d2research/EM-VLM4AD/data/a2d2/a2d2master.json"
with open(input_path, 'r') as f:
    data = json.load(f)

# Shuffle the data to ensure randomness
random.shuffle(data)

# Calculate the split indices
total_length = len(data)
train_end = int(total_length * 0.8)
val_end = train_end + int(total_length * 0.1)

# Split the data
train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

# Define output paths
train_output_path = "/home/alex/Documents/vscodeprojects/personal/a2d2research/EM-VLM4AD/data/a2d2/a2d2master_train.json"
val_output_path = "/home/alex/Documents/vscodeprojects/personal/a2d2research/EM-VLM4AD/data/a2d2/a2d2master_val.json"
test_output_path = "/home/alex/Documents/vscodeprojects/personal/a2d2research/EM-VLM4AD/data/a2d2/a2d2master_test.json"

# Save the split data to JSON files
with open(train_output_path, 'w') as f:
    json.dump(train_data, f, indent=4)

with open(val_output_path, 'w') as f:
    json.dump(val_data, f, indent=4)

with open(test_output_path, 'w') as f:
    json.dump(test_data, f, indent=4)

print(f"Data has been split and saved into:\n- Training set: {train_output_path}\n- Validation set: {val_output_path}\n- Test set: {test_output_path}")