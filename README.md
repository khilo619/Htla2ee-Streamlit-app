# 🚗 Used Car Market Explorer – Egypt

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36.0-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-Educational%20Use-green.svg)](#license)

## 📋 Overview

**Used Car Market Explorer** is a comprehensive data science project focused on analyzing and predicting used car prices in the Egyptian market. This project demonstrates the complete data science pipeline from web scraping to deployment, providing deep insights into market trends, price determinants, and offering an interactive web application for price prediction and car comparison.

### ⚠️ Important Disclaimer

This project uses publicly available car listings data from a popular automotive marketplace in Egypt that has been scraped **strictly for research and educational purposes only**. The web scraping component of this project:

- **Follows ethical scraping practices** with appropriate rate limiting to avoid overwhelming the server
- **Respects robots.txt** directives and website terms of service
- **Is intended solely for educational demonstration** of data science techniques
- **Does not store or redistribute personal information** of sellers or dealers
- **Uses only publicly visible listing information**

The scraping methodology adheres to responsible data collection practices and is designed to minimize any impact on the source website's performance.

---

## 🚀 Features

- **🔍 Ethical Web Scraping:**  
  Responsibly extracts thousands of used car listings from automotive marketplace websites using Selenium and BeautifulSoup with proper rate limiting and robots.txt compliance.

- **🧹 Robust Data Cleaning & Preprocessing:**  
  Handles missing values, removes duplicates, corrects data types, and filters outliers for reliable analysis.

- **📊 Exploratory Data Analysis & Visualization:**  
  Visualizes price trends, brand and city averages, and the impact of features like mileage and transmission type with Arabic language support.

- **🤖 Advanced Machine Learning Models:**  
  Implements Ridge Regression and XGBoost for price prediction, with feature engineering and polynomial interactions.

- **🌐 Interactive Streamlit Web App:**  
  - **📈 Market Dashboard:** Visualizes trends and key metrics
  - **🔎 Data Exploration:** Filter and browse the dataset
  - **💰 Price Prediction:** Predicts car prices using AI models
  - **⚖️ Car Comparison:** Compares two car models side by side
  - **🌍 Bilingual Support:** Handles Arabic text for Egyptian market data

---

## 📁 Project Structure

```
Htla2ee-Streamlit-app/
├── 📱 app.py                           # Main Streamlit web application
├── 📋 requirements.txt                 # Python dependencies
├── 📖 README.md                        # Project documentation
├── 📂 data/
│   ├── 📂 processed/
│   │   ├── cleaned_data.csv           # Cleaned dataset (CSV format)
│   │   └── cleaned_data.xls           # Cleaned dataset (Excel format)
│   └── 📂 exports/
│       └── Used_Car_Analysis_Egypt.mydatabase.json  # MongoDB export
├── 📂 notebooks/
│   ├── 01_data_extraction.ipynb       # Web scraping implementation
│   ├── 02_data_cleaning.ipynb         # Data preprocessing pipeline
│   ├── 03_data_visulaization.ipynb    # Exploratory data analysis
│   ├── 04_price_prediction.ipynb      # Machine learning models
│   └── 05_mongodb_integration.ipynb   # Database integration
```

---

## 🔄 Data Pipeline

### 1. 🔍 Data Extraction (`01_data_extraction.ipynb`)
- **Ethical Web Scraping:**  
  Uses Selenium WebDriver to responsibly automate browsing and extract car details (brand, model, year, mileage, price, transmission, city, etc.) from automotive marketplace websites
- **Rate Limiting:**  
  Implements appropriate delays between requests to avoid overwhelming the server
- **Robots.txt Compliance:**  
  Respects website guidelines and scraping policies
- **Data Storage:**  
  Saves raw data to CSV/Excel formats for further processing

### 2. 🧹 Data Cleaning & Preprocessing (`02_data_cleaning.ipynb`)
- **Null & Duplicate Handling:**  
  Drops irrelevant columns, removes duplicates, and handles missing values appropriately
- **Type Conversion:**  
  Converts price, year, and mileage to appropriate numeric types
- **Outlier Filtering:**  
  Removes unrealistic values (e.g., mileage = 0, future years, extremely low prices)
- **Feature Engineering:**  
  - Calculates car age from manufacturing year
  - Encodes categorical variables (brand, model, city) using target encoding
  - Log-transforms mileage for better distribution
  - Handles rare models and transmission types

### 3. 📊 Exploratory Data Analysis (`03_data_visulaization.ipynb`)
- **Market Insights:**  
  - Price depreciation analysis over years
  - Average price comparison across brands and cities
  - Transmission and fuel type distribution analysis
  - Mileage impact on pricing
- **Arabic Language Support:**  
  - Proper rendering of Arabic city names and labels
  - Bilingual visualization for Egyptian market context

### 4. 🤖 Machine Learning Modeling (`04_price_prediction.ipynb`)
- **Feature Selection & Scaling:**  
  Selects relevant features, applies standard scaling, and generates polynomial interaction terms
- **Model Implementation:**  
  - **Ridge Regression:** Linear model with L2 regularization and hyperparameter tuning
  - **XGBoost Regressor:** Gradient boosting for improved accuracy and feature importance
- **Model Evaluation:**  
  Cross-validation, R² scoring, Mean Absolute Error (MAE), and performance comparison

### 5. 🗃️ Database Integration (`05_mongodb_integration.ipynb`)
- **MongoDB Setup:**  
  Stores processed data in MongoDB for scalable data management
- **Data Export:**  
  Creates JSON exports for data sharing and backup

---

## 🌐 Streamlit Web Application (`app.py`)

The interactive web application provides a user-friendly interface for exploring the used car market data and making price predictions.

### 🎯 Key Features

- **🏠 Main Dashboard:**  
  Overview of the dataset, market statistics, and platform features

- **🔍 Data Exploration:**  
  Interactive filters to browse and analyze the dataset with real-time updates

- **📊 Data Visualization:**  
  Dynamic charts and dashboards for market insights with Arabic language support

- **💰 Price Prediction:**  
  - User-friendly input form for car specifications
  - Instant price predictions from both Ridge and XGBoost models
  - Confidence intervals and market comparisons
  - Model performance metrics display

- **⚖️ Car Comparison:**  
  Side-by-side comparison of two car models including:
  - Average price analysis
  - Mileage statistics
  - Transmission type distribution
  - Geographic availability

---

## 🚀 How to Run

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/khilo619/Htla2ee-Streamlit-app.git
   cd Htla2ee-Streamlit-app
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit App**
   ```bash
   streamlit run app.py
   ```

4. **Access the Application**
   - Open your web browser and navigate to `http://localhost:8501`
   - The application will load with the main dashboard

### 📓 Jupyter Notebooks

To explore the complete data science pipeline:

1. **Install Jupyter**
   ```bash
   pip install jupyter
   ```

2. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

3. **Open and Run Notebooks**
   - Navigate to the `notebooks/` directory
   - Run notebooks in sequence: `01_data_extraction.ipynb` → `02_data_cleaning.ipynb` → `03_data_visualization.ipynb` → `04_price_prediction.ipynb` → `05_mongodb_integration.ipynb`

---

## 📈 Model Performance

Our machine learning models demonstrate strong predictive capabilities for the Egyptian used car market:

- **🔵 Ridge Regression:**  
  - **R² Score:** ~0.85 (Explains 85% of price variance)
  - **Mean Absolute Error:** ~15%
  - **Strengths:** Stable predictions, good generalization

- **🟢 XGBoost Regressor:**  
  - **R² Score:** ~0.91 (Explains 91% of price variance)  
  - **Mean Absolute Error:** ~9.7%
  - **Strengths:** Higher accuracy, captures complex feature interactions

### 🎯 Key Insights from Analysis

- **Brand Impact:** Luxury brands (Mercedes, BMW, Audi) command premium prices
- **Depreciation:** Cars lose approximately 15-20% value per year
- **Geographic Variation:** Cairo and Alexandria show higher average prices
- **Mileage Effect:** Exponential price decrease with higher mileage
- **Transmission Preference:** Automatic transmission preferred in premium segment

---

## 🛠️ Technical Stack

### Core Technologies
- **Python 3.8+** - Primary programming language
- **Streamlit 1.36.0** - Web application framework
- **Pandas 2.2.2** - Data manipulation and analysis
- **NumPy 1.26.4** - Numerical computing
- **Scikit-learn 1.5.0** - Machine learning library
- **XGBoost 2.0.3** - Gradient boosting framework

### Visualization & UI
- **Plotly 5.22.0** - Interactive visualizations
- **Matplotlib 3.8.4** - Static plotting
- **Seaborn 0.13.2** - Statistical visualizations

### Web Scraping (Educational)
- **Selenium** - Browser automation
- **BeautifulSoup** - HTML parsing
- **Requests** - HTTP library

### Data Processing
- **OpenPyXL 3.1.2** - Excel file handling
- **Arabic-reshaper & python-bidi** - Arabic text processing

---

## 🎓 Educational Objectives

This project serves as a comprehensive educational resource demonstrating:

1. **Ethical Web Scraping Practices**
   - Responsible data collection methodologies
   - Rate limiting and server respect
   - Robots.txt compliance

2. **Complete Data Science Pipeline**
   - Data extraction and cleaning
   - Exploratory data analysis
   - Feature engineering
   - Model development and evaluation

3. **Real-world Application Development**
   - Interactive web application design
   - User experience considerations
   - Deployment-ready code structure

4. **Market Analysis Techniques**
   - Price prediction modeling
   - Market trend analysis
   - Comparative analytics

---

## 🤝 Contributing

This is an educational project, but contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License & Legal

### Educational Use License

This project is developed **strictly for educational and research purposes**. The code and methodologies are provided as learning resources for:

- Data science students and professionals
- Web scraping education (with ethical considerations)
- Machine learning model development
- Interactive application design

### Data Usage Terms

- **Source Data:** Publicly available listings from Egyptian automotive marketplace websites
- **Purpose:** Educational demonstration and research only
- **Compliance:** Adheres to ethical scraping practices and robots.txt guidelines
- **Privacy:** No personal information of sellers/dealers is collected or stored
- **Commercial Use:** Not intended for commercial purposes

### Disclaimer

Users of this code are responsible for:
- Ensuring compliance with applicable laws and website terms of service
- Implementing appropriate rate limiting and respectful scraping practices
- Using the data responsibly and ethically
- Understanding that web scraping policies may change over time

---

## 🙏 Acknowledgements

- **Data Source:** Egyptian automotive marketplace - publicly available car listings
- **Arabic Text Processing:** Arabic-reshaper and python-bidi libraries for proper text rendering
- **Machine Learning:** Scikit-learn and XGBoost communities for excellent ML libraries
- **Visualization:** Plotly and Seaborn for powerful data visualization capabilities
- **Web Framework:** Streamlit team for the amazing web app framework

---

## 📞 Contact & Support

- **Repository:** [GitHub - Htla2ee-Streamlit-app](https://github.com/khilo619/Htla2ee-Streamlit-app)
- **Issues:** Please open an issue for bug reports or feature requests
- **Educational Inquiries:** Welcome for learning and academic purposes

---

<div align="center">

**⭐ If this project helped you learn something new, please give it a star! ⭐**

Made with ❤️ for the data science and web scraping education community

</div>
