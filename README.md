# Carbon-Aware Machine Learning Training

A machine learning training application that monitors and reports **carbon emissions** during model training using **CarbonTracker**.  
The system is split into a **FastAPI backend** for model training and a **Streamlit frontend** for user interaction.

---

## Features

- Train Hugging Face Transformer models with custom configurations.
- Monitor real-time **CO₂ emissions** and training duration.
- Adjustable training parameters:
  - Number of epochs
  - Dataset size
  - Pre-trained model selection
- **Frontend (Streamlit)** for easy interaction.
- **Backend (FastAPI)** for training orchestration.
- Carbon emission tracking with **CarbonTracker**.

---

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **ML Framework**: Hugging Face Transformers, Datasets
- **Carbon Tracking**: CarbonTracker
- **Language**: Python

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/carbon-aware-ml.git
cd carbon-aware-ml
```
### 2. Install Requirements
```bash
pip install -r requirements.txt
```
### 3. Start the FastAPI Backend
```bash
uvicorn app:app --reload
```
- This starts the training API server on http://127.0.0.1:8000.
### 4. Start the Streamlit Frontend
```bash
streamlit run frontend.py
```
- This launches the web UI for configuring and starting training.

## Future Enhancements
- Support for multiple datasets.
- GPU/TPU energy consumption reporting.
- Automated CSV/JSON export of carbon reports.
- Integration with cloud-based training environments.
