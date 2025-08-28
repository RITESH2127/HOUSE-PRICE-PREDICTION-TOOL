import streamlit as st
import joblib
import pandas as pd

# Show GAIL logo centered and large at the top
try:
    st.image("GAIL.svg.png", width=250)
except:
    # If logo not found, just show the title
    pass

# Title and subtitle with styling
st.markdown(
    "<h1 style='text-align: center; color: #0072C6;'>House Price Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center; color: #444;'>Enter the details below to predict the house price</h4>",
    unsafe_allow_html=True
)

# Load the saved best model pipeline
try:
    pipeline = joblib.load('best_model.pkl')
except FileNotFoundError:
    st.error("Error: 'best_model.pkl' not found. Please ensure the best model is saved by running GAIL_PROJECT.py first.")
    st.stop()

# Card-like input area (only around the input fields)
with st.container():
    st.markdown(
        """
        <div style="background-color: #f7f7f7; border-radius: 12px; padding: 30px; margin-top: 20px; box-shadow: 0 2px 8px #ccc;">
        """,
        unsafe_allow_html=True
    )

    # Two columns with space between
    col1, col_space, col2 = st.columns([1, 0.2, 1])

    with col1:
        crim = st.number_input(
            "CRIM (Crime Rate)",
            min_value=0.0, max_value=90.0, value=0.1, format="%.2f",
            help="Per capita crime rate by town. Range: 0.0 - 90.0"
        )
        zn = st.number_input(
            "ZN (Residential Land %)",
            min_value=0.0, max_value=100.0, value=0.0, format="%.2f",
            help="Proportion of residential land zoned for lots over 25,000 sq.ft. Range: 0.0 - 100.0"
        )
        indus = st.number_input(
            "INDUS (Business Acres)",
            min_value=0.5, max_value=27.0, value=7.0, format="%.2f",
            help="Proportion of non-retail business acres per town. Range: 0.5 - 27.0"
        )
        chas = st.selectbox(
            "CHAS (Bounds River)",
            options=[0, 1],
            help="Charles River dummy variable: 1 if tract bounds river; 0 otherwise"
        )
        nox = st.number_input(
            "NOX (Nitric Oxides)",
            min_value=0.3, max_value=0.9, value=0.5, format="%.2f",
            help="Nitric oxides concentration (parts per 10 million). Range: 0.3 - 0.9"
        )
        rm = st.number_input(
            "RM (Rooms per Dwelling)",
            min_value=3.0, max_value=9.0, value=6.0, format="%.2f",
            help="Average number of rooms per dwelling. Range: 3.0 - 9.0"
        )
        age = st.number_input(
            "AGE (Older Units %)",
            min_value=2.0, max_value=100.0, value=60.0, format="%.2f",
            help="Proportion of owner-occupied units built prior to 1940. Range: 2.0 - 100.0"
        )

    with col2:
        dis = st.number_input(
            "DIS (Distance to Employment Centers)",
            min_value=1.0, max_value=13.0, value=4.0, format="%.2f",
            help="Weighted distances to five Boston employment centres. Range: 1.0 - 13.0"
        )
        rad = st.number_input(
            "RAD (Highway Accessibility)",
            min_value=1.0, max_value=24.0, value=1.0, format="%.2f",
            help="Index of accessibility to radial highways. Range: 1.0 - 24.0"
        )
        tax = st.number_input(
            "TAX (Property Tax Rate)",
            min_value=187.0, max_value=711.0, value=300.0, format="%.2f",
            help="Full-value property-tax rate per $10,000. Range: 187.0 - 711.0"
        )
        ptratio = st.number_input(
            "PTRATIO (Pupil-Teacher Ratio)",
            min_value=12.0, max_value=22.0, value=18.0, format="%.2f",
            help="Pupil-teacher ratio by town. Range: 12.0 - 22.0"
        )
        b = st.number_input(
            "B (Black Population Index)",
            min_value=0.0, max_value=400.0, value=390.0, format="%.2f",
            help="1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town. Range: 0.0 - 400.0"
        )
        lstat = st.number_input(
            "LSTAT (Lower Status %)",
            min_value=1.0, max_value=38.0, value=12.0, format="%.2f",
            help="% lower status of the population. Range: 1.0 - 38.0"
        )

    st.markdown("</div>", unsafe_allow_html=True)

# Predict button centered
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
if st.button("Predict House Price"):
    input_data = pd.DataFrame([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]],
                              columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
    try:
        predicted_price = pipeline.predict(input_data)[0]
        st.success(f"The predicted house price is: ${predicted_price:.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
st.markdown("</div>", unsafe_allow_html=True)