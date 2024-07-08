import os
import json
import pandas as pd
from glob import glob

# Paths to image directories
center_dirs = [
    "/mnt/alxd-v1-r2-code/datasets/A2D2/images/1center",
    "/mnt/alxd-v1-r2-code/datasets/A2D2/images/2center",
    "/mnt/alxd-v1-r2-code/datasets/A2D2/images/3center"
]
left_dirs = [
    "/mnt/alxd-v1-r2-code/datasets/A2D2/images/1left",
    "/mnt/alxd-v1-r2-code/datasets/A2D2/images/2left",
    "/mnt/alxd-v1-r2-code/datasets/A2D2/images/3left"
]
right_dirs = [
    "/mnt/alxd-v1-r2-code/datasets/A2D2/images/1right",
    "/mnt/alxd-v1-r2-code/datasets/A2D2/images/2right",
    "/mnt/alxd-v1-r2-code/datasets/A2D2/images/3right"
]

# Paths to the previously collected DataFrames
df_paths = {
    'acceleration_x': '/home/alex/Documents/vscodeprojects/personal/a2d2research/localdatacache/processed/acceleration_x.csv',
    'acceleration_y': '/home/alex/Documents/vscodeprojects/personal/a2d2research/localdatacache/processed/acceleration_y.csv',
    'roll_angle': '/home/alex/Documents/vscodeprojects/personal/a2d2research/localdatacache/processed/roll_angle.csv',
    'accelerator_pedal': '/home/alex/Documents/vscodeprojects/personal/a2d2research/localdatacache/processed/accelerator_pedal.csv',
    'brake_pressure': '/home/alex/Documents/vscodeprojects/personal/a2d2research/localdatacache/processed/brake_pressure.csv',
    'steering_angle_calculated': '/home/alex/Documents/vscodeprojects/personal/a2d2research/localdatacache/processed/steering_angle_calculated.csv'
}

# Load all DataFrames
print("Loading DataFrames...")
dfs = {field: pd.read_csv(path) for field, path in df_paths.items()}
print("DataFrames loaded.")

# Function to find the closest timestamp's values in a DataFrame
def find_closest_value(df, timestamp):
    idx = df['timestamp'].searchsorted(timestamp)
    if idx == len(df):
        return df.iloc[-1]
    elif idx == 0:
        return df.iloc[0]
    else:
        before = df.iloc[idx - 1]
        after = df.iloc[idx]
        if abs(after['timestamp'] - timestamp) < abs(before['timestamp'] - timestamp):
            return after
        else:
            return before

# Main function to process images and collect data
def process_images_and_collect_data(center_dirs, left_dirs, right_dirs, dfs):
    processed_centers = 0
    print_interval = 100  # Print progress every 100 groups
    data = []

    for center_dir, left_dir, right_dir in zip(center_dirs, left_dirs, right_dirs):
        center_images = sorted(glob(os.path.join(center_dir, "*.png")))
        for center_image in center_images:
            base_name = os.path.basename(center_image).replace('camera_frontcenter', 'camera_frontleft')
            left_image = os.path.join(left_dir, base_name)
            base_name = os.path.basename(center_image).replace('camera_frontcenter', 'camera_frontright')
            right_image = os.path.join(right_dir, base_name)
            
            # Get the JSON file corresponding to the center image
            json_file = center_image.replace('.png', '.json')
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            center_timestamp = metadata['cam_tstamp']
            item = {'CAM_FRONT': center_image,
                    'CAM_FRONT_LEFT': left_image,
                    'CAM_FRONT_RIGHT': right_image,
                    'center_timestamp': center_timestamp
                    }
            for field, df in dfs.items():
                closest_value = find_closest_value(df, center_timestamp)
                item[field] = closest_value[field]

            # Collect data from the DataFrames
            json_record = [
                {
                    "Q": f"{item['acceleration_x']:+09.3f} {item['acceleration_y']:+09.3f} {item['roll_angle']:+09.3f}",
                    "A": f"{item['accelerator_pedal']:+09.3f} {item['brake_pressure']:+09.3f} {item['steering_angle_calculated']:+09.3f}",
                    "C": None,
                    "con_up": None,
                    "con_down": None,
                    "cluster": None,
                    "layer": None
                },
                {
                    "CAM_FRONT": item['CAM_FRONT'],
                    "CAM_FRONT_LEFT": item['CAM_FRONT_LEFT'],
                    "CAM_FRONT_RIGHT": item['CAM_FRONT_RIGHT'],
                    "CAM_BACK": "/mnt/alxd-v1-r2-code/datasets/A2D2/images/blank_image.png",
                    "CAM_BACK_LEFT": "/mnt/alxd-v1-r2-code/datasets/A2D2/images/blank_image.png",
                    "CAM_BACK_RIGHT": "/mnt/alxd-v1-r2-code/datasets/A2D2/images/blank_image.png"
                }
            ]
            data.append(json_record)

            processed_centers += 1
            if processed_centers % print_interval == 0:
                print(f"Processed {processed_centers} image groups.")

    return data

# Save the collected data to a JSON file
output_path = "/home/alex/Documents/vscodeprojects/personal/a2d2research/EM-VLM4AD/data/a2d2/a2d2master.json"
print("Processing images and collecting data...")
collected_data = process_images_and_collect_data(center_dirs, left_dirs, right_dirs, dfs)
print("Data collection complete. Saving to JSON...")
with open(output_path, 'w') as f:
    json.dump(collected_data, f, indent=4)
print(f"Saved collected data to {output_path}")