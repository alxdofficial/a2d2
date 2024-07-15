import pandas as pd

# Paths to input and output CSV files
input_csv_path = "/home/alex/Documents/vscodeprojects/personal/a2d2research/src/data manifest/a2d2master.csv"
train_csv_path = "/home/alex/Documents/vscodeprojects/personal/a2d2research/src/data manifest/a2d2train.csv"
val_csv_path = "/home/alex/Documents/vscodeprojects/personal/a2d2research/src/data manifest/a2d2val.csv"

# Read the input CSV file
data = pd.read_csv(input_csv_path)

# Initialize empty DataFrames for training and validation
train_data = pd.DataFrame()
val_data = pd.DataFrame()

# Process each drive session separately
for drive_session in [1, 2, 3]:
    # Filter data for the current drive session
    session_data = data[data['drive_session'] == drive_session]
    
    # Calculate the split index for 90% training and 10% validation
    split_index = int(len(session_data) * 0.9)
    
    # Split the data into training and validation sets
    train_session_data = session_data.iloc[:split_index]
    val_session_data = session_data.iloc[split_index:]
    
    # Append the split data to the respective DataFrames
    train_data = pd.concat([train_data, train_session_data], ignore_index=True)
    val_data = pd.concat([val_data, val_session_data], ignore_index=True)

# Save the resulting DataFrames to CSV files
train_data.to_csv(train_csv_path, index=False)
val_data.to_csv(val_csv_path, index=False)

print(f"Training data saved to {train_csv_path}")
print(f"Validation data saved to {val_csv_path}")
