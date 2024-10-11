import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Title of the app
st.title('Car Model Analysis: Dummies and VIF')

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    raw_data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(raw_data.head())

    # Preprocessing
    data = raw_data.drop(['Model'], axis=1)
    data_cleaned = data.dropna().reset_index(drop=True)

    # Visualizations
    st.subheader('Price Distribution')
    fig, ax = plt.subplots()
    sns.histplot(data_cleaned['Price'], kde=True, ax=ax)
    st.pyplot(fig)

    # Correlation Matrix Heatmap
    st.subheader('Correlation Heatmap')
    numeric_data = data_cleaned.select_dtypes(include=[np.number])  # Select only numeric columns
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(plt)

    # Box plot for Price vs Brand
    st.subheader('Price by Brand')
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Brand', y='Price', data=raw_data)
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Log price transformation
    log_price = np.log(data_cleaned['Price'])
    data_cleaned['log_price'] = log_price

    # VIF Calculation
    st.subheader('Variance Inflation Factor (VIF) Calculation')

    # Select columns for VIF calculation
    variables = data_cleaned[['Mileage', 'Year', 'EngineV']]
    vif = pd.DataFrame()
    vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
    vif["features"] = variables.columns

    # Show VIF values
    st.write("VIF values before removing multicollinearity:")
    st.write(vif)

    # Remove 'Year' and recalculate VIF
    data_no_multicollinearity = data_cleaned.drop(['Year'], axis=1)
    variables = data_no_multicollinearity[['Mileage', 'EngineV']]
    vif = pd.DataFrame()
    vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
    vif["features"] = variables.columns

    st.write("VIF values after removing 'Year':")
    st.write(vif)

    # Create dummy variables
    data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)

    st.write("Data with dummy variables:")
    st.write(data_with_dummies.head())

    # Rearrange columns and further analysis
    cols = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW',
            'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
            'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
            'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
            'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']
    data_preprocessed = data_with_dummies[cols]
    st.write("Preprocessed Data:")
    st.write(data_preprocessed.head())

    # Model Building
    st.subheader('Linear Regression Model')
    X = data_preprocessed.drop('log_price', axis=1)
    y = data_preprocessed['log_price']
    
    model = LinearRegression()
    model.fit(X, y)

    # Predictions
    predictions = model.predict(X)

    # Calculate metrics
    st.write("Model Coefficients:")
    st.write(model.coef_)
    
    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)

    st.write(f"R² Score: {r2:.2f}")
    st.write(f"Mean Absolute Error: {mae:.2f}")

    # Add user interaction for scatter plot
    st.subheader('Interactive Scatter Plot')
    selected_brand = st.selectbox("Select Brand", raw_data['Brand'].unique())
    filtered_data = raw_data[raw_data['Brand'] == selected_brand]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Mileage', y='Price', data=filtered_data)
    st.pyplot(plt)
