"""
DATA SCIENCE QUIZ 3 — FEATURE ENGINEERING & ML INTRO
TechLabs Aachen | Digital Shaper Program
---------------------------------------------------------
Instructions:
1. Replace your_id with your full name or email.
2. Complete all functions.
3. Run the final check to get your unique key.
"""

from techlabs_checker import check_quiz
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# === 1. Personal Info ===
your_id = "your_email_here"  # <-- change this!

# === 2. TASK 1 ===
# Generate a reproducible dataset based on your name.
# Use np.random.seed(sum of ASCII codes of your name).
# Create a DataFrame with columns:
#  - 'x': 10 random integers (1–20)
#  - 'y': 3*x + 5 + random noise from np.random.randint(-2, 3)
def generate_dataset(name):
    # TODO
    df = pd.DataFrame()
    return df


# === 3. TASK 2 ===
# Add a new column 'x_squared' to the DataFrame.
# Return the modified DataFrame.
def add_feature(df):
    # TODO
    new_df = df.copy()
    return new_df


# === 4. TASK 3 ===
# Train a LinearRegression model using 'x' and 'x_squared' to predict 'y'.
# Return the coefficients as a tuple (coef_x, coef_x2, intercept)
def train_model(df):
    # TODO
    return (0.0, 0.0, 0.0)


# === 5. TASK 4 ===
# Predict y for x = 10 using your trained model.
# Return the predicted value as a float rounded to 2 decimals.
def predict_value(df):
    # TODO
    y_pred = 0.0
    return y_pred


# === 6. FINAL CHECK ===
solutions = {
    "generate_dataset": generate_dataset,
    "add_feature": add_feature,
    "train_model": train_model,
    "predict_value": predict_value
}

check_quiz(quiz_id=3, name=your_id, solutions=solutions)
