# SkyCast AI: Weather Intelligence Dashboard

A professional machine learning application built with React, TypeScript, and TensorFlow.js for historical weather analysis and predictive forecasting.

## Features
- **LSTM Time-Series Forecasting**: Predicts next-day temperature based on historical sequences.
- **Classification Pipeline**: Predicts weather labels (clear, cloudy, rain, etc.) using a deep neural network.
- **Interactive Dashboard**: Real-time training visualization, data inspection, and evaluation metrics.
- **Feature Engineering**: Automated processing of day-of-year, rolling averages, and normalization.

## Project Structure
- `src/services/weatherModel.ts`: Core ML logic using TensorFlow.js.
- `src/App.tsx`: Main dashboard UI and state management.
- `weather_data.csv`: Sample historical dataset.
- `evaluation_report.txt`: Detailed performance analysis.

## Getting Started
1. **Upload Data**: Use the "Dataset" tab to upload your own `weather_data.csv`.
2. **Train**: Navigate to the "Training" tab and click "Start Training".
3. **Evaluate**: Review the MAE, RMSE, and Accuracy metrics in the "Evaluation" tab.
4. **Predict**: Use the "Prediction" tab to input current conditions and get a forecast.

## Dataset Schema
The input CSV must contain:
- `date`: YYYY-MM-DD
- `temperature_c`: Numeric
- `humidity`: Numeric (%)
- `pressure_hpa`: Numeric
- `wind_speed_mps`: Numeric
- `precipitation_mm`: Numeric
- `weather_label`: Category (clear, cloudy, rain, storm, snow)
