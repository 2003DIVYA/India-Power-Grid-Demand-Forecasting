# India-Power-Grid-Demand-Forecasting
This project presents an end-to-end machine learning solution for forecasting India’s national electricity demand using historical hourly load data. The objective is to build a robust and interpretable predictive system that can assist in energy planning, demand management, and grid stability analysis.

The dataset consists of time-series electricity consumption data, which is preprocessed and enriched through feature engineering. Key temporal features such as hour, day, month, year, and day of the week are extracted to capture seasonal and cyclical demand patterns. Additionally, lag-based features (lag-1 and lag-24) are incorporated to model short-term dependencies and daily consumption trends.

A Ridge Regression model with cross-validation (RidgeCV) is implemented within a Scikit-learn pipeline, combined with feature scaling to ensure optimal performance. The model is trained on 80% of the data and evaluated on the remaining 20%, achieving strong performance with a high R² score (~0.95) and low error metrics (RMSE, MAE, MAPE), indicating reliable predictive capability.

The project is deployed as an interactive web application using Streamlit, providing a user-friendly interface for real-time analysis and forecasting. Key features include 24-hour ahead predictions, multi-day forecasting, actual vs predicted comparisons, and detailed data exploration across different time dimensions. The application also integrates grid load classification (low, normal, high, critical) and generates actionable insights to support operational decision-making.

Overall, this project demonstrates the practical application of time-series forecasting, feature engineering, and machine learning deployment in the energy domain, delivering both analytical depth and real-world usability.

give a 300 characters description

Here’s a ~300 character GitHub/LinkedIn short description:

Built an end-to-end electricity demand forecasting system using Ridge Regression on India’s hourly load data. Achieved ~0.95 R². Includes feature engineering, time-series modeling, and an interactive Streamlit dashboard for real-time prediction, insights, and grid analysis.
