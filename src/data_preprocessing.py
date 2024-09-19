import pandas as pd
from sklearn.model_selection import train_test_split
import os
import zipfile

def unzip_file(zip_path, extract_to_folder):
    """Unzips a zip file into the specified folder."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_folder)
        print(f"Extracted {zip_path} to {extract_to_folder}")

def load_data(file_path):
    """Loads data from a CSV file into a DataFrame."""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """Preprocess the data for training."""
    # Drop rows with missing target values
    data = data.dropna(subset=['target'])
    
    # Separate features and target variable
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def main():
    # Define paths
    zip_file_path = 'data/archive.zip'  
    extraction_path = 'data/unzipped_data/'
    extracted_file_name = 'heart-disease.csv'  
    csv_file_path = os.path.join(extraction_path, extracted_file_name)    

    # Ensure extraction path exists
    os.makedirs(extraction_path, exist_ok=True)

    # Unzip the file if it hasn't been extracted yet
    if not os.path.isfile(csv_file_path):
        unzip_file(zip_file_path, extraction_path)
    else:
        print(f"{csv_file_path} already exists, skipping extraction.")

    # Load the data
    data = load_data(csv_file_path)
    print("Data loaded successfully.") 
    
    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Save the preprocessed data to CSV files
    X_train.to_csv(os.path.join('data', 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join('data', 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join('data', 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join('data', 'y_test.csv'), index=False)
    
    print("Data preprocessing completed and saved.")

if __name__ == "__main__":
    main()
