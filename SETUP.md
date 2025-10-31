# Credit Card Fraud Detection System - Setup Guide

## System Requirements
- Python 3.11 or higher
- Git (for cloning the repository)

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/Sarita-Joshi/Credit-Card-Fraud-Detection-Spark.git
cd Credit-Card-Fraud-Detection-Spark
```

2. Create a virtual environment:
```bash
# For Linux/Mac:
python3 -m venv .venv
source .venv/bin/activate

# For Windows:
python -m venv .venv
.venv\Scripts\activate
```

3. Install required packages:
```bash
pip install pandas numpy scikit-learn
```

4. Run the fraud detection system:
```bash
python simple_fraud_detection.py
```

## Project Structure
- `simple_fraud_detection.py`: Main script for fraud detection
- `generate_data.py`: Script to generate sample transaction data
- `data/`: Directory containing training and test data
- `results/`: Directory containing fraud detection results

## Expected Output
The system will:
1. Generate sample transaction data if not present
2. Train a fraud detection model
3. Process transactions and detect potential fraud
4. Save results to `results/fraud_detection_results.csv`

## Troubleshooting
- If you get "ModuleNotFoundError", make sure you've activated the virtual environment and installed all required packages
- If you get permission errors, make sure you have write permissions in the project directory

## Notes
- This is a simplified version that uses CSV files for data storage
- Results will be saved in the `results` directory
- Sample data is generated automatically if not present

## Original Project Components (Optional)
If you want to run the full version with all features:
- Apache Spark
- Apache Kafka
- Apache Cassandra
These components are not required for the simplified version.