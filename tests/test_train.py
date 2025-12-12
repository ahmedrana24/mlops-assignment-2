import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def test_data_exists():
    # Test if data exists
    assert os.path.exists("data/dataset.csv")

def test_training_flow():
    # Test if we can load data and fit a model
    data = pd.read_csv('data/dataset.csv')
    assert not data.empty
    
    X = data[['id', 'value']]
    y = data['label']
    
    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X, y)
    
    # Check if model has classes_
    assert hasattr(clf, "classes_")