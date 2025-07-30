Amazon Review Authenticity Detector
A complete machine learning system built to identify potentially AI-generated or inauthentic reviews on Amazon using deep learning and real-time Kafka pipelines.

🧱 System Overview
This project follows a two-phase pipeline architecture for review scraping, feature extraction, and fraud detection:

📦 Phase 1 – Batch Data Ingestion & Model Training
Data Collection: Scrapes reviews via Amazon Scraper (Apify API) using ASINs

Preprocessing: Cleans and converts raw JSON reviews into structured data

Feature Engineering: TF-IDF vectors + linguistic signals

Model Training: Trains various classifiers including Neural Networks, XGBoost, and Random Forest

Offline Evaluation: Evaluates all models on static test data

🔄 Phase 2 – Real-time Stream Processing with Kafka
Streaming Setup: Kafka handles real-time review ingestion

Live Feature Extraction: Extracts features on-the-fly

Online Prediction: Applies trained ensemble models to streamed reviews

Live Feedback: Stores prediction confidence and metrics for monitoring

🔍 Key Capabilities
⚙️ Smart Preprocessing
Custom n-gram TF-IDF vectorizer

Filters noise and filler terms

Measures vocabulary richness, writing complexity, sentiment shift

Supports dataset balancing and augmentation

🤖 Modular Model Design
Deep neural architecture: 512 → 256 → 128 → 64

Integrated dropout and batch norm layers

Uses weighted loss for class imbalance

Early stopping and checkpointing enabled

Ensemble support to boost generalization

⚡ Real-time Review Monitoring
Kafka-based live data ingestion

Real-time review scoring and labeling

Dashboard-ready JSON outputs

Modular Kafka consumer/producer logic

🚀 How to Use
1. Setup the Environment



pip install -r requirements.txt
docker-compose up -d  # Starts Kafka, Zookeeper, etc.
2. Collect & Train (Phase 1)



python main.py --mode train --run-scraper      # Scrape and train
python main.py --mode train                    # Only training
3. Evaluate Model Performance



python main.py --mode test
python main.py --mode test --run-scraper       # Test on new scraped data
4. Real-time Pipeline (Phase 2)



python main.py --mode serve                    # Kafka consumer
python -m src.kafka.producer --simulate 10 5   # Simulated producer
5. Full Pipeline Trigger



python main.py --mode kafka-pipeline           # Run scraper → Kafka pipeline
python amazon_scraper.py --send-to-kafka       # Manual sending
⚙️ Configuration Overview
Configuration is handled via config/config.yaml:

yaml


kafka:
  bootstrap_servers: "localhost:39092"
  topic: "amazon_reviews"
  group_id: "review_detector"

model:
  input_features: 5000
  hidden_layers: [512, 256, 128, 64]
  dropout_rates: [0.2, 0.2, 0.2, 0.1]
  learning_rate: 0.0005
  batch_size: 32
  epochs: 100
  early_stopping_patience: 10
  use_batch_normalization: true
  use_class_weights: true
🗂️ Folder Layout



├── Data/
│   ├── Scraped_data/         # Review data from Amazon
│   ├── Training_data/        # Final training sets
│   ├── Testing_data/         # Holdout test data
│
├── config/
│   └── config.yaml
│
├── models/                   # Saved models
├── results/                  # Evaluation output
│   ├── training/
│   ├── testing/
│   └── real_time_predictions.csv
│
├── src/
│   ├── kafka/
│   │   ├── consumer.py
│   │   └── producer.py
│   ├── processing/
│   │   └── data_preprocessing.py
│   └── utils/
│       └── logger.py
│
├── amazon_scraper.py
├── combine_datasets.py
├── docker-compose.yml
├── main.py
└── requirements.txt
📈 Evaluation Dashboard
After model runs, the system automatically generates metrics:

Accuracy, Precision, Recall, F1-score

Class-specific metrics (Original vs. Generated reviews)

Visual reports:

Confusion matrix

Probability histograms

Label distribution pie charts

Performance summary bar plots

Optional HTML summary report with embedded charts

Generate manually via:




python src/evaluation.py
python src/evaluation.py --threshold 0.9 --set-labels --output results/custom_eval
📊 Output Samples
results/testing/test_results.csv: Final prediction labels with probabilities

results/training/roc_curves_ieee.png: ROC curves

results/training/feature_importance_ieee.png: Feature importance plot

results/evaluation/report.html: Full evaluation summary

📌 Highlights
Model ensemble reduces overfitting

Dropout and batch norm improve stability

Live data labeling with Kafka

Feature name consistency fixed across pipeline

Clean folder organization for training vs. testing
