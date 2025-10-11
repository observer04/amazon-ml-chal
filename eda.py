

import pandas as pd

def load_and_explore_data():
    """
    Loads the training data and performs initial exploration.
    """
    # Load the training data
    try:
        train_df = pd.read_csv("dataset/train.csv")
    except FileNotFoundError:
        print("Error: train.csv not found in the dataset directory.")
        return

    # Display the first few rows
    print("First 5 rows of the training data:")
    print(train_df.head())

    # Display basic information about the dataset
    print("\nBasic information about the training data:")
    train_df.info()

    # Display descriptive statistics
    print("\nDescriptive statistics of the training data:")
    print(train_df.describe())

if __name__ == "__main__":
    load_and_explore_data()

