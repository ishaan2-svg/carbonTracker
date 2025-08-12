import streamlit as st
import requests

st.set_page_config(page_title="Carbon-Aware ML Training", layout="wide")

st.title("🌱 Carbon-Aware ML Training Frontend")

st.markdown("""
This application allows you to train a machine learning model and track its carbon emissions.
The backend is powered by FastAPI, and the model training uses Hugging Face Transformers.
""")

st.subheader("Training Configuration")

# User input for epochs
epochs = st.slider(
    "Select the number of training epochs:",
    min_value=1,
    max_value=10,
    value=2
)

# User input for dataset size
dataset_size = st.number_input(
    "Select the dataset size for training:",
    min_value=10,
    max_value=1000,
    value=100
)

# User input for model selection
model_name = st.selectbox(
    "Select a pre-trained model:",
    ("distilbert-base-uncased", "bert-base-uncased")
)

# Button to start training
if st.button("Start Training"):
    with st.spinner("Training in progress... This may take a while."):
        try:
            # Prepare the request body
            data = {
                "epochs": epochs,
                "dataset_size": dataset_size,
                "model_name": model_name
            }
            
            # Send a POST request to the FastAPI backend
            response = requests.post("http://127.0.0.1:8000/train", json=data)
            
            # Check for a successful response
            if response.status_code == 200:
                report = response.json()
                st.success("Training completed successfully! 🎉")
                
                # Display the carbon report
                st.subheader("Carbon Emissions Report")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(label="Total Emissions", value=f"{report['emissions_kg']:.4f} kg CO2")
                
                with col2:
                    st.metric(label="Training Duration", value=f"{report['duration_sec']:.2f} seconds")
                
                with col3:
                    st.metric(label="Epochs Trained", value=f"{report['epochs_trained']}")
                
                st.balloons()
            else:
                # Handle errors from the backend
                st.error(f"Error during training: {response.json().get('detail', 'Unknown error')}")
        
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend server. Please ensure the FastAPI server is running.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

st.markdown("---")

st.info("To run this application, you must first start the FastAPI backend server by running `uvicorn app:app --reload` in your terminal.")