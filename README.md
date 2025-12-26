---
title: Htla2ee Streamlit App
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.28.0"
app_file: app.py
pinned: false
---

# Htla2ee Streamlit App 🚗

## Overview
Htla2ee Streamlit App is a professional data analytics and price prediction platform for Egypt's used car market. Leveraging real data scraped from Hatla2ee, the app provides:
- Interactive market exploration
- Advanced data visualizations
- AI-powered price prediction
- Car comparison tools

## Features
- **Market Explorer:** Filter and explore thousands of car listings by brand, model, year, price, and city.
- **Data Visualization:** Dynamic charts for price trends, brand/city analysis, transmission and fuel type distributions, and more.
- **Price Prediction:** Machine learning models (Ridge Regression & XGBoost) estimate car prices based on user input.
- **Car Comparison:** Side-by-side comparison of two cars across key metrics.
- **MongoDB Integration:** Store and manage cleaned data for scalable analytics.

## Data Pipeline
The project follows a robust data science workflow:

### 1. Data Extraction ([notebooks/01_data_extraction.ipynb](notebooks/01_data_extraction.ipynb))
- Scrapes car listings from Hatla2ee using Selenium and BeautifulSoup
- Extracts brand, model, price, year, mileage, city, transmission, color, fuel, and date

### 2. Data Cleaning ([notebooks/02_data_cleaning.ipynb](notebooks/02_data_cleaning.ipynb))
- Handles missing values, removes duplicates, and corrects data types
- Cleans and standardizes price, mileage, and categorical fields
- Removes outliers and unrealistic entries
- Outputs a clean dataset: [data/processed/cleaned_data.csv](data/processed/cleaned_data.csv)

### 3. Data Visualization ([notebooks/03_data_visulaization.ipynb](notebooks/03_data_visulaization.ipynb))
- Explores trends in price, year, brand, city, transmission, and fuel type
- Uses Matplotlib, Seaborn, and Plotly for interactive and publication-quality charts

### 4. Price Prediction ([notebooks/04_price_prediction.ipynb](notebooks/04_price_prediction.ipynb))
- Feature engineering: target encoding, polynomial features, log transforms
- Trains and evaluates Ridge Regression and XGBoost models
- Reports R² and MAE for model performance

### 5. MongoDB Integration ([notebooks/05_mongodb_integration.ipynb](notebooks/05_mongodb_integration.ipynb))
- Loads cleaned data into a MongoDB database for scalable querying and analytics

## App Structure
- **app.py:** Main Streamlit app with navigation, visualizations, prediction, and comparison modules
- **data/processed/cleaned_data.csv:** Final cleaned dataset used by the app
- **notebooks/**: Jupyter notebooks for each pipeline stage

## Usage
1. **Install requirements:**
	```bash
	pip install -r requirements.txt
	```
2. **Run the app locally:**
	```bash
	streamlit run app.py
	```
3. **Explore the app:**
	- Use the sidebar to navigate between Market Explorer, Data Visualization, Price Prediction, and Car Comparison

## Requirements
- Python 3.11 (see `runtime.txt`)
- All dependencies listed in `requirements.txt` or `req.txt`

## Hugging Face Deployment
This app is ready for deployment on [Hugging Face Spaces](https://huggingface.co/spaces):

1. **Prepare your repository:**
	- Ensure `app.py`, `requirements.txt`, and `runtime.txt` are in the root directory
	- Add your data files (or use a public data source)
2. **Create a new Space:**
	- Go to [Hugging Face Spaces](https://huggingface.co/spaces)
	- Select "Streamlit" as the SDK
	- Link your repository or upload your files
3. **Configure:**
	- The app will automatically launch using `app.py`
	- Make sure your requirements and runtime files are correct

For more details, see the [Hugging Face Streamlit deployment guide](https://huggingface.co/docs/hub/spaces-sdks-streamlit).

---
**Author:** [KhaLood]  
**Data Source:** [Hatla2ee](https://eg.hatla2ee.com/)  
**License:** MIT
