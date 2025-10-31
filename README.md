# Transaction Fraud Detection Using PySpark ML

A scalable real-time fraud detection system leveraging Apache Spark's machine learning capabilities, integrated with Kafka streaming and Cassandra for processing financial transactions at scale.

### DEVELOPER :  
*JB DINESH KUMAR*
---

### Under Guidence Of:  
*Prof.Suganeshwari*
---


## 📋 Project Overview

This project demonstrates an end-to-end implementation of a fraud detection system designed to identify fraudulent transactions in real-time. Built on a distributed architecture, the system combines stream processing with machine learning to analyze transaction patterns and flag suspicious activities instantly.

### Key Highlights

- **Real-time Detection**: Processes transactions as they occur using Kafka streaming
- **Machine Learning**: Random Forest classifier trained on 14.8M transactions
- **Distributed Computing**: Leverages Spark for handling large-scale data
- **Scalable Storage**: Cassandra database for high-throughput writes
- **Live Monitoring**: Interactive Streamlit dashboard for visualization

## 🎯 Problem Statement

Financial fraud causes billions in losses annually. Traditional batch processing systems detect fraud after the fact, while this system identifies suspicious transactions in real-time, enabling immediate action to prevent fraud before it completes.

## 🏗️ Architecture Overview

The system follows a microservices-inspired architecture:
```
Data Generation → Kafka Stream → Spark Processing → ML Prediction → Cassandra → Dashboard
```

**Components:**
- **Streaming Layer**: Apache Kafka + Zookeeper
- **Processing Engine**: Apache Spark
- **ML Framework**: PySpark MLlib
- **Database**: Apache Cassandra
- **Frontend**: Python Streamlit
- **Orchestration**: Python scripts

## 💻 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Data Processing | Apache Spark 3.x | Distributed computing |
| Streaming | Apache Kafka | Message queue |
| Coordination | Apache Zookeeper | Service coordination |
| Database | Apache Cassandra | NoSQL storage |
| ML Library | PySpark MLlib | Model training |
| Visualization | Streamlit | Dashboard |
| Backend | Python 3.11 | Scripting & orchestration |

## 📊 Dataset Information

The project uses a comprehensive synthetic dataset generated using the Sparkov tool, which simulates realistic transaction patterns.

**Dataset Specifications:**
- **Volume**: 1GB / 14.8 million records

### Feature Set

The dataset contains 22 attributes across multiple categories:

**Transaction Attributes:**
- Timestamp, Amount, Merchant, Category, Transaction ID

**Customer Profile:**
- Personal details (name, DOB, gender)
- Location (address, coordinates, city population)
- Occupation

**Merchant Information:**
- Location coordinates
- Business category

**Target Variable:**
- `is_fraud` (Binary: 0 = Legitimate, 1 = Fraudulent)

## 🚀 Getting Started

### System Requirements

- **OS**: Windows/Linux/MacOS
- **Java**: Version 8 or higher
- **Python**: 3.11+
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB free space

### Installation Guide

#### 1. Apache Spark
```bash
# Download from https://spark.apache.org/downloads.html
# Extract and set environment variables
export SPARK_HOME=/path/to/spark
export PATH=$PATH:$SPARK_HOME/bin
```

#### 2. Apache Kafka
```bash
# Download from https://kafka.apache.org/downloads
# Extract to desired location
export KAFKA_HOME=/path/to/kafka
export PATH=$PATH:$KAFKA_HOME/bin
```

#### 3. Apache Cassandra
```bash
# Download from https://cassandra.apache.org/download/
# Extract and configure
export CASSANDRA_HOME=/path/to/cassandra
export PATH=$PATH:$CASSANDRA_HOME/bin
```

#### 4. Python Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pyspark==3.5.0
kafka-python==2.0.2
cassandra-driver==3.28.0
streamlit==1.28.0
pandas==2.1.0
numpy==1.24.3
scikit-learn==1.3.0
faker==19.6.2
matplotlib==3.7.2
seaborn==0.12.2
```

## 🔧 Running the System

### Step 1: Initialize Services

**Terminal 1 - Zookeeper:**
```bash
# Windows
kafka_2.12-3.7.0\bin\windows\zookeeper-server-start.bat config\zookeeper.properties

# Linux/Mac
bin/zookeeper-server-start.sh config/zookeeper.properties
```

**Terminal 2 - Kafka:**
```bash
# Windows
kafka_2.12-3.7.0\bin\windows\kafka-server-start.bat config\server.properties

# Linux/Mac
bin/kafka-server-start.sh config/server.properties
```

**Terminal 3 - Cassandra:**
```bash
# Navigate to Cassandra bin directory
cd $CASSANDRA_HOME/bin
./cassandra  # Linux/Mac
cassandra.bat  # Windows
```

### Step 2: Prepare the Model
```bash
# Train and save the fraud detection model
python scripts/model_training.py
```

This performs:
- Data loading and exploration
- Feature engineering and preprocessing
- Model training with Random Forest
- Performance evaluation
- Model serialization

### Step 3: Start Transaction Stream
```bash
# Launch the transaction producer
python scripts/transaction_producer.py
```

Simulates real-world transaction generation and publishes to Kafka topic.

### Step 4: Activate Fraud Detection
```bash
# Start the consumer and prediction engine
python scripts/fraud_detector.py
```

Consumes transactions, applies ML model, stores predictions in Cassandra.

### Step 5: Launch Analytics Dashboard
```bash
# Start the Streamlit dashboard
streamlit run scripts/analytics_dashboard.py
```

Access at `http://localhost:8501` for real-time monitoring.

## 📂 Repository Structure
```
transaction-fraud-detection/
│
├── data/
│   ├── raw/
│   │   └── fraudTrain.csv
│   └── processed/
│
├── models/
│   └── rf_fraud_model/
│       ├── metadata/
│       └── stages/
│
├── scripts/
│   ├── model_training.py
│   ├── transaction_producer.py
│   ├── fraud_detector.py
│   └── analytics_dashboard.py
│
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── model_evaluation.ipynb
│
├── config/
│   ├── kafka_config.py
│   ├── cassandra_config.py
│   └── spark_config.py
│
├── utils/
│   ├── data_preprocessing.py
│   └── feature_engineering.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚡ System Workflow

1. **Data Preprocessing**: Clean and transform raw transaction data
2. **Feature Engineering**: Create relevant features from transaction attributes
3. **Model Development**: Train Random Forest classifier on historical data
4. **Stream Generation**: Simulate real-time transactions via producer
5. **Real-time Inference**: Consume streams and predict fraud probability
6. **Data Persistence**: Store transactions and predictions in Cassandra
7. **Visualization**: Display analytics and insights on dashboard

## 🎨 Dashboard Features

- **Real-time Metrics**: Transaction volume, fraud rate, alert count
- **Visualizations**: Time-series plots, category distributions, geographic heatmaps
- **Fraud Analytics**: Top risky merchants, high-value fraud attempts
- **Model Performance**: Accuracy, precision, recall metrics
- **Alert System**: Recent flagged transactions with details

## 🔍 Machine Learning Approach

**Algorithm**: Random Forest Classifier

**Rationale:**
- Handles non-linear relationships effectively
- Robust to outliers
- Provides feature importance insights
- Excellent for imbalanced datasets

**Training Process:**
- Train/test split: 80/20
- Cross-validation for hyperparameter tuning
- Class weight balancing for imbalanced data
- Feature scaling and normalization

## 📈 Performance Metrics

The model is evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Fraud detection accuracy
- **Recall**: Fraud capture rate
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Model discrimination capability

## 🚧 Technical Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Platform compatibility | Configured cross-platform scripts and paths |
| Data generation realism | Used Faker library with custom rules |
| Memory constraints in Cassandra | Optimized JVM heap settings and batch sizes |
| Kafka throughput | Tuned producer/consumer configurations |
| Real-time latency | Implemented micro-batching in Spark |

## 🔮 Future Roadmap

- [ ] Deep learning models (LSTM, Autoencoders)
- [ ] Online learning for model updates
- [ ] Multi-model ensemble approach
- [ ] Email/SMS alert integration
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] API endpoints for external integration
- [ ] Advanced anomaly detection techniques

## 📚 Learning Outcomes

This project demonstrates expertise in:
- **Big Data Technologies**: Spark, Kafka, Cassandra
- **Machine Learning**: Classification, model evaluation
- **Stream Processing**: Real-time data pipelines
- **System Design**: Distributed architecture
- **Python Development**: Production-grade code

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation


## 📬 Contact

**Dinesh Kumar**
- GitHub: https://github.com/Dineshkumar-jb
- LinkedIn: https://www.linkedin.com/in/dineshkumar-jb/
- Email: dineshkumar.j.b2005@gmail.com
