import json
import pandas as pd
import os
import glob

# Define the directory containing the JSON files and the output directory
json_directory = "../localdatacache"
output_directory = "../localdatacache/processed"

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# List of fields to process
fields_to_process = [
    'acceleration_x', 'acceleration_y', 'roll_angle',
    'accelerator_pedal', 'brake_pressure', 'steering_angle_calculated'
]

# Function to aggregate data for each field from all JSON files
def aggregate_data_from_files():
    aggregated_data = {field: [] for field in fields_to_process}
    aggregated_data['steering_angle_calculated_sign'] = []
    json_files = glob.glob(os.path.join(json_directory, '*.json'))
    print(f"Found {len(json_files)} JSON files to process.")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as file:
                data = json.load(file)
                for field in fields_to_process + ['steering_angle_calculated_sign']:
                    if field in data:
                        aggregated_data[field].extend(data[field]['values'])
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    return aggregated_data

# Function to create and save DataFrame for each field
def create_and_save_dataframes(aggregated_data):
    for field, values in aggregated_data.items():
        if field == 'steering_angle_calculated_sign':
            continue
        if values:
            # Create DataFrame
            df = pd.DataFrame(values, columns=['timestamp', field])
            # Drop duplicates based on timestamp
            df = df.drop_duplicates(subset=['timestamp'])
            # Sort DataFrame by timestamp
            df_sorted = df.sort_values(by='timestamp').reset_index(drop=True)
            
            if field == 'steering_angle_calculated':
                # Adjust the steering angle based on the sign
                sign_df = pd.DataFrame(aggregated_data['steering_angle_calculated_sign'], columns=['timestamp', 'sign'])
                df_sorted = df_sorted.merge(sign_df, on='timestamp')
                df_sorted['steering_angle_calculated'] = df_sorted.apply(
                    lambda row: row['steering_angle_calculated'] if row['sign'] == 1 else -row['steering_angle_calculated'], axis=1)
                df_sorted = df_sorted[['timestamp', 'steering_angle_calculated']]  # Drop the sign column
            
            # Save DataFrame to CSV
            output_path = os.path.join(output_directory, f"{field}.csv")
            df_sorted.to_csv(output_path, index=False)
            print(f"Saved {field} DataFrame to {output_path}")

# Main function to aggregate data and save DataFrames
def process_and_save_all_fields():
    aggregated_data = aggregate_data_from_files()
    create_and_save_dataframes(aggregated_data)

# Run the processing function
if __name__ == "__main__":
    process_and_save_all_fields()
