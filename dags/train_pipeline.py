from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import random

# detailed DAG with NO external libraries
def load_data():
    print("Loading data...")
    return "Data Loaded"

def train_model(**context):
    print("Training model...")
    accuracy = random.uniform(0.8, 0.95)
    print(f"Model trained with accuracy: {accuracy}")
    return accuracy

def save_model():
    print("Saving model...")
    return "Model Saved"

with DAG(
    dag_id='mlops_pipeline_fixed',  # Renamed to force fresh load
    schedule=None,                  # <--- THE FIX IS HERE
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:

    task_load = PythonOperator(
        task_id='load_data',
        python_callable=load_data
    )

    task_train = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )

    task_save = PythonOperator(
        task_id='save_model',
        python_callable=save_model
    )

    task_load >> task_train >> task_save