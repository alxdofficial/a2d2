import os
import json
import pandas as pd
from glob import glob

# Root directory path
root_dir = "/media/alex/39eb242f-14ca-4925-8a70-35633885bff4/A2D2/"

# Paths to image directories (relative to root_dir)
center_dirs = [
    "1center/camera_lidar/20180810_150607/camera/cam_front_center",
    "2center/camera_lidar/20190401_121727/camera/cam_front_center",
    "3center/camera_lidar/20190401_145936/camera/cam_front_center"
]
left_dirs = [
    "1left/camera_lidar/20180810_150607/camera/cam_front_left",
    "2left/camera_lidar/20190401_121727/camera/cam_front_left",
    "3left/camera_lidar/20190401_145936/camera/cam_front_left"
]
right_dirs = [
    "1right/camera_lidar/20180810_150607/camera/cam_front_right",
    "2right/camera_lidar/20190401_121727/camera/cam_front_right",
    "3right/camera_lidar/20190401_145936/camera/cam_front_right"
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

# Load all DataFrames and ensure they are sorted by timestamp
print("Loading DataFrames...")
dfs = {field: pd.read_csv(path).sort_values('timestamp') for field, path in df_paths.items()}
print("DataFrames loaded and sorted by timestamp.")

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
def process_images_and_collect_data(root_dir, center_dirs, left_dirs, right_dirs, dfs):
    processed_centers = 0
    print_interval = 100  # Print progress every 100 groups
    data = []
    drive_session = 0
    for center_dir, left_dir, right_dir in zip(center_dirs, left_dirs, right_dirs):
        drive_session += 1
        print("drive: ", drive_session)
        center_images = sorted(glob(os.path.join(root_dir, center_dir, "*.png")))
        print(f"{len(center_images)} images found in drive")
        for frame_num, center_image in enumerate(center_images, 1):
            # Generate paths for left and right images
            left_image = center_image.replace(center_dir, left_dir).replace('camera_front_center', 'camera_front_left').replace("frontcenter","frontleft")
            right_image = center_image.replace(center_dir, right_dir).replace('camera_front_center', 'camera_front_right').replace("frontcenter","frontright")
            
            # Get the JSON file corresponding to the center image
            json_file = center_image.replace('.png', '.json')
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            center_timestamp = metadata['cam_tstamp']
            
            # Generate relative paths for DataFrame
            relative_center_image = os.path.relpath(center_image, root_dir)
            relative_left_image = os.path.relpath(left_image, root_dir)
            relative_right_image = os.path.relpath(right_image, root_dir)
        
            item = {'drive_session': drive_session,
                    'frame_num': frame_num,
                    'cam_front_path': relative_center_image,
                    'cam_left_path': relative_left_image,
                    'cam_right_path': relative_right_image
                    }
            for field, df in dfs.items():
                closest_value = find_closest_value(df, center_timestamp)
                item[field] = closest_value[field]

            # Collect data into the list
            data.append(item)

            processed_centers += 1
            if processed_centers % print_interval == 0:
                print(f"Processed {processed_centers} image groups.")

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data, columns=[
        'drive_session', 'frame_num', 'cam_front_path', 'cam_left_path', 'cam_right_path',
        'acceleration_x', 'acceleration_y', 'roll_angle', 'accelerator_pedal', 'brake_pressure', 'steering_angle_calculated'
    ])
    
    return df

# Save the collected data to a CSV file
output_path = "/home/alex/Documents/vscodeprojects/personal/a2d2research/src/data manifest/a2d2master.csv"
print("Processing images and collecting data...")
collected_data = process_images_and_collect_data(root_dir, center_dirs, left_dirs, right_dirs, dfs)
print("Data collection complete. Saving to CSV...")
collected_data.to_csv(output_path, index=False)
print(f"Saved collected data to {output_path}")
