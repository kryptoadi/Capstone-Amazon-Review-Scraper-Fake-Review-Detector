# 📦 Amazon Review Scraper & Fake Review Detector

This project provides a complete pipeline to **scrape product reviews from Amazon**, process the data, and **detect fake or AI-generated reviews** using advanced machine learning and deep learning techniques.

---

## 🚀 Features

- ✅ Scrapes Amazon reviews using product ASINs  
- ✅ Sends real-time review data to **Apache Kafka**  
- ✅ Converts JSON to structured CSV  
- ✅ Trains ensemble models (Neural Network, XGBoost, Random Forest, and others)  
- ✅ Supports pseudo-labeling and class balancing  
- ✅ Visualizes results with IEEE-style plots  
- ✅ Detects suspicious review patterns using NLP and linguistic analysis  

---

## 📂 Project Structure

├── amazon_scraper.py # Review scraping and Kafka integration
├── combine_datasets.py # Merges labeled & unlabeled data
├── main.py # Central control (train, test, serve)
├── deploy.sh # Docker-based deployment script
├── docker-compose.yml # Kafka and Zookeeper containers
├── product_asins.txt # List of ASINs to scrape
├── requirements.txt # Python dependencies
├── .env.example # Sample environment file
└── Data/
└── Scraped_data/ # Raw and converted review data
└── Training_data/ # Combined datasets for model training

yaml
Copy
Edit

---

## 🧠 Model Architecture

- **Text Processing**: TF-IDF, sentence complexity, vocabulary diversity, etc.  
- **Models Used**:
  - Multilayer Neural Network
  - XGBoost Classifier
  - Random Forest
  - SVM, KNN, Logistic Regression, Naive Bayes (optional)
- **Ensemble Voting** for final prediction

---

## 🔎 Linguistic Features Analyzed

- Vocabulary uniqueness  
- Repetitive patterns  
- Sentiment consistency  
- Use of filler/extreme words  
- Formulaic phrases (e.g. "highly recommend", "excellent quality")

---

## 🛠️ How to Run

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
Step 2: Set up environment variables
Create a .env file using .env.example and add:

env
Copy
Edit
APIFY_API_KEY=your_apify_token_here
KAFKA_BOOTSTRAP_SERVERS=localhost:39092
Step 3: Deploy Kafka services
bash
Copy
Edit
./deploy.sh up
Step 4: Run the scraper and send reviews to Kafka
bash
Copy
Edit
python main.py --mode test --run-scraper
Step 5: Train the model
bash
Copy
Edit
python main.py --mode train --dataset Data/Training_data/combined_dataset.csv
Step 6: Evaluate or serve predictions
bash
Copy
Edit
python main.py --mode test
python main.py --mode serve
📊 Output Examples
results/training/feature_importance_ieee.png – Top review features

results/training/roc_curves_ieee.png – ROC curves for all models

results/testing/test_results.csv – Label predictions and confidence

🧪 Fake Review Detection Logic
The final prediction combines:

Statistical review features

NLP signals (sentiment, repetition, caps ratio)

Ensemble of ML classifiers

Threshold optimization to reduce false positives

💡 Use Cases
E-commerce fraud detection

Marketplace moderation tools

Academic research on generative content
