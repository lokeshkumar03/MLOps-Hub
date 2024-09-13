# Import libraries
import argparse
import glob
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mlflow.sklearn import autolog
from sklearn.metrics import accuracy_score

# Define functions
def main(args):
    # Enable autologging
    autolog()

    # Read data
    df = get_csvs_df(args.training_data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # Train model
    model = train_model(args.reg_rate, X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)

def split_data(df, test_size=0.2):
    if "Diabetic" not in df.columns:
        raise RuntimeError("Target column 'Diabetic' not found in the DataFrame.")
    
    X = df.drop("Diabetic", axis=1)
    y = df["Diabetic"]
    return train_test_split(X, y, test_size=test_size)

def train_model(reg_rate, X_train, y_train):
    model = LogisticRegression(C=1/reg_rate, solver="liblinear")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.2f}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", dest='training_data', type=str, required=True)
    parser.add_argument("--reg_rate", dest='reg_rate', type=float, default=0.01)
    return parser.parse_args()

# Run script
if __name__ == "__main__":
    print("\n\n")
    print("*" * 60)
    args = parse_args()
    main(args)
    print("*" * 60)
    print("\n\n")