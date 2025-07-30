📦 Amazon Review Scraper & Fake Review Detector
This project provides a complete pipeline to scrape product reviews from Amazon, process the data, and detect fake or AI-generated reviews using advanced machine learning and deep learning techniques.

🚀 Features
Scrapes Amazon reviews using product ASINs

Sends real-time review data to Apache Kafka

Converts JSON to structured CSV

Trains ensemble models (Neural Network, XGBoost, Random Forest, and others)

Supports pseudo-labeling and class balancing

Visualizes results with IEEE-style plots

Detects suspicious review patterns using NLP and linguistic analysis

📂 Project Structure
amazon_scraper.py - Review scraping and Kafka integration
combine_datasets.py - Merges labeled & unlabeled data
main.py - Central control (train, test, serve)
deploy.sh - Docker-based deployment script
docker-compose.yml - Kafka and Zookeeper containers
product_asins.txt - List of ASINs to scrape
requirements.txt - Python dependencies
.env.example - Sample environment file
Data/
Scraped_data/ - Raw and converted review data
Training_data/ - Combined datasets for model training

🧠 Model Architecture
Text processing includes TF-IDF, sentence complexity, and vocabulary diversity.
Models used include:

Multilayer Neural Network

XGBoost Classifier

Random Forest

SVM, KNN, Logistic Regression, Naive Bayes (optional)
Final output is calculated using ensemble voting.

🔎 Linguistic Features Analyzed
Vocabulary uniqueness

Repetitive patterns

Sentiment consistency

Use of filler or extreme words

Formulaic phrases like "highly recommend", "excellent quality"

🛠️ How to Run
Step 1: Install dependencies using pip install -r requirements.txt

Step 2: Set up environment variables.
Create a .env file using .env.example and add:
APIFY_API_KEY=your_apify_token_here
KAFKA_BOOTSTRAP_SERVERS=localhost:39092

Step 3: Deploy Kafka services by running ./deploy.sh up

Step 4: Run the scraper and send reviews to Kafka using
python main.py --mode test --run-scraper

Step 5: Train the model with
python main.py --mode train --dataset Data/Training_data/combined_dataset.csv

Step 6: Evaluate the model or serve predictions
python main.py --mode test
python main.py --mode serve

📊 Output Examples
results/training/feature_importance_ieee.png - Top review features
results/training/roc_curves_ieee.png - ROC curves for all models
results/testing/test_results.csv - Label predictions and confidence

🧪 Fake Review Detection Logic
Final predictions are based on a combination of statistical review features, NLP-based signals like sentiment and repetition, and voting from multiple ML classifiers. Threshold optimization helps reduce false positives.

💡 Use Cases
E-commerce fraud detection

Marketplace moderation tools

Academic research on generative content
