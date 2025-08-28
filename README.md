# House Price Prediction Tool

A Python-based web application that predicts house prices using machine learning models, powered by Streamlit. This application uses the Boston Housing Dataset features to provide accurate price predictions through an intuitive web interface.

![House Price Prediction App](https://github.com/user-attachments/assets/b873651f-754e-4776-b1d0-de3763167cdc)

## 🏠 Overview

This project implements a comprehensive house price prediction system that:
- Uses multiple machine learning algorithms to ensure accuracy
- Provides a user-friendly web interface built with Streamlit
- Accepts 13 key housing features for prediction
- Automatically selects the best performing model from 8 different algorithms
- Displays predictions in a professional, easy-to-use format

## ✨ Features

- **Interactive Web Interface**: Clean, professional UI with the GAIL logo
- **Multiple ML Models**: Tests 8 different algorithms and uses the best performer
- **Real-time Predictions**: Instant house price predictions based on input features
- **Feature Validation**: Input validation with helpful tooltips for each feature
- **Responsive Design**: Card-based layout with proper styling
- **Error Handling**: Graceful error handling for missing models or invalid inputs

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Required Python packages (see installation section)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RITESH2127/HOUSE-PRICE-PREDICTION-TOOL.git
   cd HOUSE-PRICE-PREDICTION-TOOL
   ```

2. **Install required packages:**
   ```bash
   pip install streamlit pandas scikit-learn joblib numpy
   ```

### Running the Application

1. **First, train the machine learning models:**
   ```bash
   python GAIL_PROJECT.py
   ```
   This will:
   - Load or generate the housing dataset
   - Train 8 different ML models
   - Select the best performing model
   - Save it as `best_model.pkl`

2. **Launch the Streamlit web application:**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access the application:**
   - The app will automatically open in your default web browser
   - If not, navigate to `http://localhost:8501`

## 📊 Dataset & Features

The application uses 13 housing features based on the Boston Housing Dataset:

| Feature | Description | Range |
|---------|-------------|--------|
| **CRIM** | Per capita crime rate by town | 0.0 - 90.0 |
| **ZN** | Proportion of residential land zoned for lots over 25,000 sq.ft | 0.0 - 100.0 |
| **INDUS** | Proportion of non-retail business acres per town | 0.5 - 27.0 |
| **CHAS** | Charles River dummy variable (1 if tract bounds river; 0 otherwise) | 0 or 1 |
| **NOX** | Nitric oxides concentration (parts per 10 million) | 0.3 - 0.9 |
| **RM** | Average number of rooms per dwelling | 3.0 - 9.0 |
| **AGE** | Proportion of owner-occupied units built prior to 1940 | 2.0 - 100.0 |
| **DIS** | Weighted distances to five Boston employment centres | 1.0 - 13.0 |
| **RAD** | Index of accessibility to radial highways | 1.0 - 24.0 |
| **TAX** | Full-value property-tax rate per $10,000 | 187.0 - 711.0 |
| **PTRATIO** | Pupil-teacher ratio by town | 12.0 - 22.0 |
| **B** | 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town | 0.0 - 400.0 |
| **LSTAT** | % lower status of the population | 1.0 - 38.0 |

## 🤖 Machine Learning Models

The application tests multiple algorithms and automatically selects the best performer:

- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **Elastic Net**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **Support Vector Regression (SVR)**

Each model uses standardized features through scikit-learn pipelines for optimal performance.

## 📁 Project Structure

```
HOUSE-PRICE-PREDICTION-TOOL/
│
├── streamlit_app.py           # Main Streamlit web application
├── GAIL_PROJECT.py           # Model training script with offline support
├── GAIL_PROJECT_ORIGINAL.py  # Original script (requires internet connection)
├── best_model.pkl            # Trained model file (generated after running training)
├── GAIL.svg.png             # GAIL logo for the application
├── house_prices.csv         # Additional dataset file
└── README.md                # This documentation file
```

## 💡 Usage Instructions

1. **Input Features**: Enter values for all 13 housing features using the input fields
2. **Tooltips**: Hover over the info icons (ℹ️) to see detailed descriptions and valid ranges
3. **Predict**: Click the "Predict House Price" button to generate a prediction
4. **Result**: The predicted house price will be displayed in dollars

### Example Usage:
- Set Crime Rate to 0.1 (low crime area)
- Set Rooms per Dwelling to 7.0 (spacious homes)
- Adjust other features based on the property characteristics
- Click "Predict House Price" to see the estimated value

## 🔧 Troubleshooting

### Common Issues:

**"'best_model.pkl' not found" error:**
- Run `python GAIL_PROJECT.py` first to train and save the model

**Import errors:**
- Install missing packages: `pip install streamlit pandas scikit-learn joblib numpy`

**Port already in use:**
- Use a different port: `streamlit run streamlit_app.py --server.port 8502`

**Network connectivity issues (original script):**
- The current `GAIL_PROJECT.py` works offline with synthetic data
- If you have the original version that requires internet, it's saved as `GAIL_PROJECT_ORIGINAL.py`

## 🔄 Development

### To modify the application:

1. **Update features**: Modify the input fields in `streamlit_app.py`
2. **Add new models**: Update the `model_dict` in the training script
3. **Change styling**: Modify the CSS in the `st.markdown` sections
4. **Update validation**: Adjust min/max values for input fields

### Model Performance:
The application automatically selects the best model based on Mean Squared Error (MSE). Typical performance metrics:
- **Best Model**: Usually Linear Regression or Ridge Regression
- **R² Score**: ~0.97 (97% variance explained)
- **MSE**: ~4.6 (low prediction error)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

- **RITESH2127** - [GitHub Profile](https://github.com/RITESH2127)

## 🙏 Acknowledgments

- Boston Housing Dataset from the UCI Machine Learning Repository
- Streamlit for the excellent web framework
- Scikit-learn for machine learning algorithms
- GAIL (Gas Authority of India Limited) for project sponsorship