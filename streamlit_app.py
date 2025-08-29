import streamlit as st
import joblib
import pandas as pd



# Display the image at the top (make sure the file path is correct)
st.image(r"GAIL.svg.png", width=150)
# Define the Streamlit application title and description
st.title("Boston House Price Prediction")
st.write("Enter the details of the house to predict its price.")

# Center the input form using columns with extra space between
st.markdown("## House Features")
outer_col1, outer_col2, outer_col3 = st.columns([1, 2, 1])

# Load the saved best model pipeline
try:
    pipeline = joblib.load(r'best_model.pkl')
except FileNotFoundError:
    st.error("Error: 'best_model.pkl' not found. Please ensure the best model is saved.")
    st.stop()

with outer_col2:
    left, spacer, right = st.columns([1, 0.3, 1])
    with left:
        crim = st.number_input(
            "CRIM",
            value=0.1,
            format="%f",
            help="Per capita crime rate by town. Typical range: 0.0 - 90.0"
        )
        zn = st.number_input(
            "ZN",
            value=0.0,
            format="%f",
            help="Proportion of residential land zoned for lots over 25,000 sq.ft. Typical range: 0.0 - 100.0"
        )
        indus = st.number_input(
            "INDUS",
            value=7.0,
            format="%f",
            help="Proportion of non-retail business acres per town. Typical range: 0.5 - 27.0"
        )
        chas = st.selectbox(
            "CHAS",
            options=[0, 1],
            help="Charles River dummy variable: 1 if tract bounds river; 0 otherwise"
        )
        nox = st.number_input(
            "NOX",
            value=0.5,
            format="%f",
            help="Nitric oxides concentration (parts per 10 million). Typical range: 0.3 - 0.9"
        )
        rm = st.number_input(
            "RM",
            value=6.0,
            format="%f",
            help="Average number of rooms per dwelling. Typical range: 3.0 - 9.0"
        )
        age = st.number_input(
            "AGE",
            value=60.0,
            format="%f",
            help="Proportion of owner-occupied units built prior to 1940. Typical range: 2.0 - 100.0"
        )
    with right:
        dis = st.number_input(
            "DIS",
            value=4.0,
            format="%f",
            help="Weighted distances to five Boston employment centres. Typical range: 1.0 - 13.0"
        )
        rad = st.number_input(
            "RAD",
            value=1.0,
            format="%f",
            help="Index of accessibility to radial highways. Typical range: 1.0 - 24.0"
        )
        tax = st.number_input(
            "TAX",
            value=300.0,
            format="%f",
            help="Full-value property-tax rate per $10,000. Typical range: 187.0 - 711.0"
        )
        ptratio = st.number_input(
            "PTRATIO",
            value=18.0,
            format="%f",
            help="Pupil-teacher ratio by town. Typical range: 12.0 - 22.0"
        )
        b = st.number_input(
            "B",
            value=390.0,
            format="%f",
            help="1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town. Typical range: 0.0 - 400.0"
        )
        lstat = st.number_input(
            "LSTAT",
            value=12.0,
            format="%f",
            help="% lower status of the population. Typical range: 1.0 - 38.0"
        )

    # Add a button to trigger prediction
    if st.button("Predict House Price"):
        input_data = pd.DataFrame([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]],
                                  columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
        try:
            predicted_price = pipeline.predict(input_data)[0]
            final_price = predicted_price*1000
            st.success(f"The predicted house price is: ${final_price:.6f}")
        except Exception as e:

            st.error(f"An error occurred during prediction: {e}")
