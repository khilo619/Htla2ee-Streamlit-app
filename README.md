# Used Car Market Explorer – Egypt

## Overview

**Used Car Market Explorer** is a comprehensive data science project focused on analyzing and predicting used car prices in the Egyptian market. Leveraging real-world data scraped from Hatla2ee, this project provides deep insights into market trends, price determinants, and offers an interactive web application for price prediction and car comparison.

---

## Features

- **Automated Data Extraction:**  
  Scrapes thousands of used car listings from Hatla2ee using Selenium and BeautifulSoup.

- **Robust Data Cleaning & Preprocessing:**  
  Handles missing values, removes duplicates, corrects data types, and filters outliers for reliable analysis.

- **Exploratory Data Analysis & Visualization:**  
  Visualizes price trends, brand and city averages, and the impact of features like mileage and transmission type.

- **Advanced Machine Learning Models:**  
  Implements Ridge Regression and XGBoost for price prediction, with feature engineering and polynomial interactions.

- **Interactive Streamlit Web App:**  
  - **Market Dashboard:** Visualizes trends and key metrics.
  - **Data Exploration:** Filter and browse the dataset.
  - **Price Prediction:** Predicts car prices using AI models.
  - **Car Comparison:** Compares two car models side by side.

---

## Project Structure

```
.
├── app.py                  # Streamlit web application
├── Used_Car_Analysis.ipynb # Data extraction, cleaning, EDA, and modeling notebook
├── cleaned_data.xls        # Cleaned dataset used for analysis and app
├── requirements.txt        # Python dependencies
```

---

## Data Pipeline

### 1. Data Extraction ([Used_Car_Analysis.ipynb](Used_Car_Analysis.ipynb))
- **Web Scraping:**  
  Uses Selenium to automate browsing and extract car details (brand, model, year, mileage, price, transmission, city, etc.) from Hatla2ee.
- **Data Storage:**  
  Saves raw and cleaned data to CSV/Excel for further processing.

### 2. Data Cleaning & Preprocessing
- **Null & Duplicate Handling:**  
  Drops irrelevant columns, removes duplicates, and fills or drops missing values.
- **Type Conversion:**  
  Converts price, year, and mileage to numeric types.
- **Outlier Filtering:**  
  Removes unrealistic values (e.g., mileage = 0, future years, extremely low prices).
- **Feature Engineering:**  
  - Extracts car age.
  - Encodes categorical variables (brand, model, city) using target encoding.
  - Log-transforms mileage for normalization.
  - Handles rare models and transmission types.

### 3. Exploratory Data Analysis (EDA)
- **Visualizations:**  
  - Price vs. Year (Depreciation)
  - Average price per brand and city
  - Transmission and fuel type distributions
  - Mileage impact on price

### 4. Machine Learning Modeling
- **Feature Selection & Scaling:**  
  Selects relevant features, applies standard scaling, and generates polynomial interaction terms.
- **Model Training:**  
  - **Ridge Regression:** Hyperparameter tuning and cross-validation.
  - **XGBoost Regressor:** For comparison and improved accuracy.
- **Evaluation:**  
  Reports R² and MAE, and compares model performance.

---

## Streamlit App ([app.py](app.py))

### Key Modules

- **Main Page:**  
  Overview of the dataset and platform features.

- **Data Exploration:**  
  Interactive filters to browse and analyze the dataset.

- **Data Visualization:**  
  Dynamic charts and dashboards for market insights.

- **Price Prediction:**  
  User inputs car details to get instant price predictions from both Ridge and XGBoost models, with confidence intervals and market comparisons.

- **Car Comparison:**  
  Side-by-side comparison of two car models, including average price, mileage, transmission, and city.

---

## How to Run

1. **Install Requirements**
    ```sh
    pip install -r requirements.txt
    ```

2. **Launch the Streamlit App**
    ```sh
    streamlit run app.py
    ```

3. **Explore the Notebook**  
   Open [Used_Car_Analysis.ipynb](Used_Car_Analysis.ipynb) in Jupyter or VS Code for full data pipeline and modeling details.

---

## Model Performance

- **Ridge Regression:**  
  - R² ≈ 0.85  
  - MAE ≈ 15%

- **XGBoost:**  
  - R² ≈ 0.91  
  - MAE ≈ 9.7%

---

## Acknowledgements

- Data sourced from [Hatla2ee](https://eg.hatla2ee.com/ar/car).
- Built with Python, Pandas, Scikit-learn, XGBoost, Plotly, Seaborn, and Streamlit.

---

## License

This project is for educational and research purposes only. Please respect the terms of use of the data source.

---

## Contact

For questions or collaboration, please open an issue or contact
