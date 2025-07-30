# 🛡️ Amazon Review Authenticity Detector

A complete machine learning system built to identify potentially AI-generated or inauthentic reviews on Amazon using deep learning and real-time Kafka pipelines.

---

## 🧱 System Overview

This project follows a two-phase pipeline architecture for review scraping, feature extraction, and fraud detection:

### 📦 Phase 1 – Batch Data Ingestion & Model Training

- **Data Collection**: Scrapes reviews via Amazon Scraper (Apify API) using ASINs  
- **Preprocessing**: Cleans and converts raw JSON reviews into structured data  
- **Feature Engineering**: TF-IDF vectors + linguistic signals  
- **Model Training**: Trains classifiers (Neural Networks, XGBoost, Random Forest)  
- **Offline Evaluation**: Evaluates models on static test data  

### 🔄 Phase 2 – Real-time Stream Processing with Kafka

- **Streaming Setup**: Kafka handles real-time review ingestion  
- **Live Feature Extraction**: Extracts features on-the-fly  
- **Online Prediction**: Applies ensemble models to streamed reviews  
- **Live Feedback**: Stores prediction confidence and metrics  

---

## 🔍 Key Capabilities

### ⚙️ Smart Preprocessing

- Custom n-gram TF-IDF vectorizer  
- Filler and repetitive term filtering  
- Vocabulary richness, writing complexity, sentiment shift detection  
- Dataset balancing and augmentation  

### 🤖 Modular Model Design

- Deep neural network: 512 → 256 → 128 → 64  
- Batch norm + dropout for stability  
- Class-weighted loss  
- Early stopping and model checkpointing  
- Optional model ensembling  

### ⚡ Real-time Review Monitoring

- Kafka-powered streaming pipeline  
- Live review scoring and confidence tracking  
- JSON-ready output for dashboards  
- Configurable consumer-producer architecture  

---

## 🚀 How to Use

### 1. Setup the Environment

```bash
pip install -r requirements.txt
docker-compose up -d
```

### 2. Collect & Train (Phase 1)

```bash
python main.py --mode train --run-scraper
python main.py --mode train
```

### 3. Evaluate Model Performance

```bash
python main.py --mode test
python main.py --mode test --run-scraper
```

### 4. Real-time Pipeline (Phase 2)

```bash
python main.py --mode serve
python -m src.kafka.producer --simulate 10 5
```

### 5. Full Pipeline Trigger

```bash
python main.py --mode kafka-pipeline
python amazon_scraper.py --send-to-kafka
```

---

## ⚙️ Configuration Overview

All key parameters are stored in `config/config.yaml`:

```yaml
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
```

---

## 🗂️ Folder Layout

```
├── Data/
│   ├── Scraped_data/
│   ├── Training_data/
│   ├── Testing_data/
│
├── config/
│   └── config.yaml
│
├── models/
├── results/
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
```

---

## 📈 Evaluation Dashboard

After training, the system can generate performance metrics and visualizations:

- Accuracy, Precision, Recall, F1-score  
- Class-level metrics for Original (OR) vs AI-generated (CG) reviews  
- Visuals:
  - Confusion matrix  
  - Probability distribution histograms  
  - Class pie charts  
  - ROC curves  
  - Feature importance graphs  

### Run Evaluation

```bash
python src/evaluation.py
python src/evaluation.py --threshold 0.9 --set-labels --output results/custom_eval
```

---

## 📊 Output Samples

- `results/testing/test_results.csv`: Final prediction results  
- `results/training/roc_curves_ieee.png`: ROC curves  
- `results/training/feature_importance_ieee.png`: Feature importance  
- `results/evaluation/report.html`: Full HTML summary  

---

## 📌 Highlights

- Ensemble logic boosts prediction reliability  
- Dropout and batch norm help stable training  
- Apache Kafka enables real-time detection  
- Data and results folders kept clean and organized  
- Feature naming consistency handled for training/inference compatibility  

---

## 🛠️ Troubleshooting

### Docker Port Conflicts

If you see something like:

```
Bind for 0.0.0.0:32181 failed: port is already allocated
```

Fix it by either:

```bash
lsof -i :32181
kill -9 <PID>
docker-compose up -d --remove-orphans
```

Or update ports in `docker-compose.yml`.

---

## ⚠️ Disclaimer

This project is meant purely for educational and research purposes. Please follow Amazon’s terms of service and applicable laws when using any scraping tools or automated methods.
