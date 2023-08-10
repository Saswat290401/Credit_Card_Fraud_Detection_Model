import warnings
import streamlit as st
import pickle
import numpy as np

with open('logistic_regression_model.pkl', 'rb') as file:
    logreg = pickle.load(file)

with open('random_forest_model.pkl', 'rb') as file:
    rf = pickle.load(file)

with open('mlp_model.pkl', 'rb') as file:
    mlp = pickle.load(file)

with open('decision_tree_model.pkl', 'rb') as file:
    dt = pickle.load(file)

with open('gradient_boosting_model.pkl', 'rb') as file:
    gb = pickle.load(file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('pca_model.pkl', 'rb') as file:
    pca = pickle.load(file)

def predict_fraud_probabilities(amount, time):
    v_features = np.random.uniform(-113.7433067, 120.5894939, size=28)
    input_data = np.array([time, amount] + list(v_features)).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    
    predictions = {
        'Logistic Regression': logreg.predict_proba(input_data_scaled)[:, 1],
        'Random Forest': rf.predict_proba(input_data_scaled)[:, 1],
        'Neural Network': mlp.predict_proba(input_data_scaled)[:, 1],
        'Decision Tree': dt.predict_proba(input_data_scaled)[:, 1],
        'Gradient Boosting': gb.predict_proba(input_data_scaled)[:, 1]
    }
    
    combined_probability = np.mean(list(predictions.values())) * 100
    
    return combined_probability

# Streamlit app
def main():
    st.set_page_config(
        page_title="Credit Card Fraud Detection App",
        page_icon=":credit_card:",
        layout="wide"
    )

    st.title('Credit Card Fraud Detection')
    st.sidebar.header("About")
    st.sidebar.write("This app predicts the chance of a transaction being fraud.")

    st.write("## Adjust the transaction details using sliders and see the combined chance of a transaction being fraud. ##")

    col1, col2 = st.columns(2)

    with col1:
        time = st.slider("Time (In Seconds)", 0, 200000, value=0)

    with col2:
        amount = st.slider("Amount (In Rupees)", 0.0, 10000000.0, value=0.0)

    if st.button('Calculate Fraud Probability'):
        combined_probability = predict_fraud_probabilities(amount, time)

        st.write("---")
        st.write("## Result:")
        st.write("### Chance of Transaction Being Fraud (Combined): {:.2f}%".format(combined_probability))

if __name__ == '__main__':
    main()