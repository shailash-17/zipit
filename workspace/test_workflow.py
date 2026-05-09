
import numpy as np
import joblib

# Load model
model = joblib.load('user_model.joblib')
print(f"Model loaded successfully")

# Make prediction
test_data = np.random.rand(1, 10)
prediction = model.predict(test_data)
print(f"Prediction: {prediction[0]}")
print("Workspace execution successful!")
