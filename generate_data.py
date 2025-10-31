import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_sample_data(num_records=1000):
    """Generate sample transaction data"""
    
    # Create sample data
    data = {
        'trans_date_trans_time': [(datetime.now() - timedelta(minutes=i)).strftime('%Y-%m-%d %H:%M:%S') 
                                 for i in range(num_records)],
        'cc_num': [random.randint(4000000000000000, 4999999999999999) for _ in range(num_records)],
        'merchant': [f'merchant_{i%100}' for i in range(num_records)],
        'category': [random.choice(['grocery', 'shopping', 'entertainment', 'travel', 'food']) 
                    for _ in range(num_records)],
        'amt': [round(random.uniform(10.0, 1000.0), 2) for _ in range(num_records)],
        'first': [f'first_{i%50}' for i in range(num_records)],
        'last': [f'last_{i%50}' for i in range(num_records)],
        'gender': [random.choice(['M', 'F']) for _ in range(num_records)],
        'street': [f'street_{i%100}' for i in range(num_records)],
        'city': [f'city_{i%20}' for i in range(num_records)],
        'state': [f'state_{i%50}' for i in range(num_records)],
        'zip': [random.randint(10000, 99999) for _ in range(num_records)],
        'lat': [random.uniform(25.0, 50.0) for _ in range(num_records)],
        'long': [random.uniform(-120.0, -70.0) for _ in range(num_records)],
        'city_pop': [random.randint(1000, 1000000) for _ in range(num_records)],
        'job': [f'job_{i%30}' for i in range(num_records)],
        'dob': [(datetime.now() - timedelta(days=random.randint(6570, 29200))).strftime('%Y-%m-%d') 
                for _ in range(num_records)],
        'trans_num': [f'TRANS_{i}' for i in range(num_records)],
        'unix_time': [int(datetime.now().timestamp()) - i*60 for i in range(num_records)],
        'merch_lat': [random.uniform(25.0, 50.0) for _ in range(num_records)],
        'merch_long': [random.uniform(-120.0, -70.0) for _ in range(num_records)]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add fraud label (mostly non-fraud with some fraud cases)
    df['is_fraud'] = [1 if random.random() < 0.1 else 0 for _ in range(num_records)]
    
    return df

if __name__ == "__main__":
    print("Generating sample transaction data...")
    
    # Generate training data
    train_df = generate_sample_data(5000)
    train_df.to_csv('data/train_transactions.csv', index=False)
    print("Created training data with 5000 records")
    
    # Generate test data
    test_df = generate_sample_data(1000)
    test_df.to_csv('data/test_transactions.csv', index=False)
    print("Created test data with 1000 records")
    
    print("\nData files created successfully!")