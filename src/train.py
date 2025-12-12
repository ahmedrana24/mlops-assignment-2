import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# 1. Load dataset
data = pd.read_csv('data/dataset.csv')
X = data[['id', 'value']]
y = data['label']

# 2. Train model
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X, y)

# 3. Save model
os.makedirs('models', exist_ok=True) # Ensure folder exists
with open('models/model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Model trained and saved to models/model.pkl")