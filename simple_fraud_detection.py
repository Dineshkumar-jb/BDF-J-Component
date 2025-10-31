import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import os

def prepare_features(df):
    """Prepare features for model training"""
    # Identify categorical columns
    categorical_columns = ['merchant', 'category', 'first', 'last', 'gender', 
                         'street', 'city', 'state', 'job']
    
    # Create label encoders for each categorical column
    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        encoders[col] = le
    
    # Select features for model
    feature_columns = ([col for col in df.columns 
                       if col.endswith('_encoded')] +
                      ['cc_num', 'amt', 'zip', 'lat', 'long', 
                       'city_pop', 'unix_time', 'merch_lat', 'merch_long'])
    
    return df, feature_columns, encoders

def train_model(train_data_path):
    """Train the fraud detection model"""
    # Read training data
    train_df = pd.read_csv(train_data_path)
    
    # Prepare features
    train_df, feature_columns, encoders = prepare_features(train_df)
    
    # Train model
    X = train_df[feature_columns]
    y = train_df['is_fraud']
    
    model = LogisticRegression()
    model.fit(X, y)
    
    return model, feature_columns, encoders

def process_transactions(model, feature_columns, encoders, test_data_path, threshold=0.3):
    """Process transactions and detect fraud"""
    # Read test data
    test_df = pd.read_csv(test_data_path)
    
    # Encode categorical features
    for col, encoder in encoders.items():
        test_df[f'{col}_encoded'] = encoder.transform(test_df[col])
    
    # Make predictions
    X_test = test_df[feature_columns]
    probabilities = model.predict_proba(X_test)[:, 1]  # Probability of fraud
    predictions = (probabilities >= threshold).astype(int)  # Use custom threshold
    
    # Add predictions to results
    results = test_df[['trans_date_trans_time', 'trans_num', 'cc_num', 'amt', 
                      'merchant', 'category', 'is_fraud']].copy()
    results['prediction'] = predictions
    results['fraud_probability'] = probabilities
    results['processed_at'] = datetime.now()
    
    # Save results
    output_file = 'results/fraud_detection_results.csv'
    results.to_csv(output_file, index=False)
    
    # Print summary
    total = len(results)
    fraud = len(results[results['prediction'] == 1])
    print(f"\nProcessed {total} transactions")
    print(f"Detected {fraud} potentially fraudulent transactions ({fraud/total*100:.1f}%)")
    print(f"\nSample of detected fraud cases:")
    print(results[results['prediction'] == 1][['trans_num', 'amt', 'merchant', 'fraud_probability']].head())
    print(f"\nResults saved to {output_file}")
    
    return results

def main():
    print("Credit Card Fraud Detection System")
    print("=================================")
    
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Check if we have data files
    if not os.path.exists('data/train_transactions.csv'):
        print("\nGenerating sample data...")
        os.system('python generate_data.py')
    
    print("\nTraining fraud detection model...")
    model, feature_columns, encoders = train_model('data/train_transactions.csv')
    
    print("\nProcessing transactions...")
    results = process_transactions(model, feature_columns, encoders, 'data/test_transactions.csv')
    
    print("\nGenerating visualizations...")
    os.system('python visualize_results.py')

if __name__ == "__main__":
    main()