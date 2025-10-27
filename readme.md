# Customer Churn Prediction Project

## 🚀 Overview
This project is an end-to-end machine learning application that predicts whether a telecom customer will churn. It includes data analysis, model training, hyperparameter tuning, and a deployed web interface using Streamlit.

The trained model achieves **~81% accuracy** on the test set.

## 📂 Project Structure

customer-churn-prediction/ ├── data/ │ └── Telco-Customer-Churn.csv ├── models/ │ └── churn_prediction_model.joblib ├── notebooks/ │ └── 1.0-data-exploration-and-model-training.ipynb ├── .gitignore ├── app.py ├── README.md └── requirements.txt

## 🛠️ Technologies Used
- Python
- Pandas & NumPy for data manipulation
- Scikit-learn for modeling and preprocessing
- Streamlit for the web application
- Joblib for model serialization

## ⚙️ How to Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd customer-churn-prediction
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Jupyter Notebook (Optional):**
    To see the training process, run the notebook in the `notebooks/` directory. This will regenerate the model in the `models/` folder.
    ```bash
    jupyter notebook
    ```
5.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```