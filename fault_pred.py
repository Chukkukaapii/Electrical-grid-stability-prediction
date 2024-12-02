import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import kagglehub

st.title("Electrical Grid Stability Prediction")
st.write("""
This app uses a **Random Forest Classifier** to predict the stability of the electrical grid based on the provided dataset. 
Upload a dataset or use the KaggleHub integration to fetch the data.
""")

st.sidebar.header("Model Configuration")
n_estimators = st.sidebar.slider("Number of Trees in Random Forest", 10, 200, 100)
test_size = st.sidebar.slider("Test Data Size (%)", 10, 40, 20)

st.header("1. Upload Dataset or Use KaggleHub")
dataset_source = st.radio("Select Dataset Source", ("Upload Dataset", "Use KaggleHub"))

if dataset_source == "Upload Dataset":
    uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(data.head())
elif dataset_source == "Use KaggleHub":
    st.write("Downloading dataset from KaggleHub...")
    path = kagglehub.dataset_download("ishadss/electrical-grid-stability-simulated-data-data-set")
    file_path = f"{path}/Data_for_UCI_named.csv"
    data = pd.read_csv(file_path)
    st.write("Dataset Preview:")
    st.write(data.head())

if 'data' in locals() or 'data' in globals():
    data.drop(columns=['stabf'], inplace=True)

    X = data.drop(columns=['stab'])  
    y = (data['stab'] > 0).astype(int)  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False

    if not st.session_state.model_trained:
        st.header("2. Train the Random Forest Classifier")
        if st.button("Train Model"):
            rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            rf_model.fit(X_train, y_train)
            y_pred = rf_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Model Accuracy: **{accuracy:.2f}**")
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            st.session_state.rf_model = rf_model
            st.session_state.model_trained = True

    if st.session_state.model_trained:
        st.header("3. Make Predictions")
        st.write("Use the form below to test the model with custom inputs.")

        input_data = {}
        for feature in X.columns:
            input_data[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

        if st.button("Predict Stability"):
            rf_model = st.session_state.rf_model
            
            new_data = pd.DataFrame([input_data])
            prediction = rf_model.predict(new_data)
            stability = "Stable" if prediction[0] == 1 else "Unstable"
            st.write(f"**Prediction: {stability}**")
