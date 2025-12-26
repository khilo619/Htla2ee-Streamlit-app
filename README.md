🚗 Htla2ee – Egyptian Used Car Market Explorer

Htla2ee is an end-to-end data science & machine learning project that analyzes and predicts used car prices in the Egyptian market, combining ethical data collection, advanced ML modeling, and an interactive Streamlit web application deployed on Hugging Face Spaces.

📌 Project Overview

This project demonstrates a full real-world data science pipeline:

Ethical web scraping of public car listings

Data cleaning & feature engineering

Exploratory data analysis with Arabic support

Machine learning price prediction models

Interactive Streamlit application

Cloud deployment using Hugging Face Spaces

The application allows users to:

Explore the Egyptian used car market

Analyze pricing trends by brand, city, and year

Predict car prices using trained ML models

Compare different car configurations

⚠️ Ethical & Legal Disclaimer

This project uses publicly available car listing data collected strictly for educational and research purposes.

✅ Respects robots.txt and website terms

✅ Applies rate limiting during scraping

✅ Collects no personal or private data

❌ Not intended for commercial usage

Users are responsible for ensuring compliance with local laws and website policies.

🚀 Key Features
🔍 Data Collection

Ethical web scraping using Selenium & BeautifulSoup

Rate limiting and respectful crawling

Structured storage for further analysis

🧹 Data Processing

Missing value handling & duplicate removal

Outlier filtering (price, mileage, year)

Feature engineering (car age, log mileage)

Target encoding for categorical variables

📊 Exploratory Data Analysis

Price trends & depreciation analysis

Brand, city, and transmission comparisons

Arabic text rendering support

Interactive and static visualizations

🤖 Machine Learning

Ridge Regression (baseline, stable)

XGBoost Regressor (high accuracy)

Cross-validation & performance comparison

Feature importance analysis

🌐 Streamlit Web Application

Market dashboard

Dataset exploration

AI-based price prediction

Car-to-car comparison

Arabic-friendly UI

🗂️ Project Structure
Htla2ee-Streamlit-app/
├── app.py                         # Streamlit application
├── requirements.txt               # Python dependencies
├── README.md                      # Documentation
├── data/
│   ├── processed/
│   │   ├── cleaned_data.csv
│   │   └── cleaned_data.xls
│   └── exports/
│       └── Used_Car_Analysis_Egypt.mydatabase.json
├── notebooks/
│   ├── 01_data_extraction.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_data_visualization.ipynb
│   ├── 04_price_prediction.ipynb
│   └── 05_mongodb_integration.ipynb

🔄 Data Science Pipeline
1️⃣ Data Extraction

Automated browsing via Selenium

Extraction of brand, model, year, mileage, price, city, transmission

2️⃣ Data Cleaning & Feature Engineering

Data type correction

Outlier removal

Car age calculation

Encoding categorical features

Mileage transformation

3️⃣ Exploratory Data Analysis

Price vs age & mileage

Brand and city price distribution

Transmission impact

Arabic visualization support

4️⃣ Machine Learning Models
Model	R² Score	MAE	Notes
Ridge Regression	~0.85	~15%	Stable baseline
XGBoost	~0.91	~9.7%	Best performance
5️⃣ Deployment

Models integrated into Streamlit app

Hosted on Hugging Face Spaces

🌐 Streamlit Application

The app provides an intuitive interface for both technical and non-technical users.

Main Sections

📊 Market Overview Dashboard

🔍 Dataset Exploration

💰 Price Prediction

⚖️ Car Comparison

Predictions are generated instantly using trained ML models.

🤗 Hugging Face Deployment

This project is fully compatible with Hugging Face Spaces and uses the following configuration (already included at the top of this README):

sdk: streamlit
app_file: app.py
sdk_version: 1.28.0

🚀 Deploy to Hugging Face Spaces

Create a new Streamlit Space on Hugging Face

Connect your GitHub repository or upload files manually

Ensure the following files exist:

app.py

requirements.txt

README.md (with HF metadata header)

Hugging Face will automatically:

Install dependencies

Launch the Streamlit app

Host it publicly

✅ No Docker required
✅ Automatic rebuild on updates

▶️ Run Locally
Prerequisites

Python 3.8+

pip

Installation
git clone https://github.com/khilo619/Htla2ee-Streamlit-app.git
cd Htla2ee-Streamlit-app
pip install -r requirements.txt

Launch App
streamlit run app.py


Open: http://localhost:8501

🛠️ Tech Stack

Core

Python

Streamlit

Pandas, NumPy

Scikit-learn

XGBoost

Visualization

Plotly

Matplotlib

Seaborn

Scraping (Educational)

Selenium

BeautifulSoup

Requests

Arabic Support

arabic-reshaper

python-bidi

🎓 Educational Value

This repository is ideal for:

Data science portfolios

Machine learning practice

End-to-end project demonstrations

Streamlit & Hugging Face deployment learning

🤝 Contributing

Contributions are welcome for educational improvements.

Fork the repo

Create a feature branch

Commit changes

Open a Pull Request

📄 License & Legal

Educational Use Only

No commercial usage

No redistribution of scraped data

Respect data source policies

📬 Contact

GitHub: https://github.com/khilo619/Htla2ee-Streamlit-app

Issues: Bug reports & feature requests welcome

<div align="center">

⭐ If you found this project useful, consider giving it a star! ⭐

Built with ❤️ for learning, data science, and real-world ML applications.

</div> ```
