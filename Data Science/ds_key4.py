"""
DATA SCIENCE QUIZ 4 — MODEL EVALUATION & INSIGHTS
TechLabs Aachen | Digital Shaper Program
---------------------------------------------------------
Instructions:
1. Replace your_id with your name or email.
2. Complete all tasks.
3. Use check_quiz() to verify and generate your key.
"""

from techlabs_checker import check_quiz
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === 1. Personal Info ===
your_id = "your_email_here"  # <-- change this!

# === 2. TASK 1 ===
# Generate data using your name as a seed:
#   y = 4*x + 10 + random noise (-3, +3)
# Create DataFrame with 'x' and 'y' (15 samples, x=0..14)
def generate_eval_data(name):
    # TODO
    df = pd.DataFrame()
    return df


# === 3. TASK 2 ===
# Train a LinearRegression model on the generated data.
# Return the trained model object.
def train_eval_model(df):
    # TODO
    model = None
    return model


# === 4. TASK 3 ===
# Compute MAE, MSE, and R² between predicted and actual y.
# Return a dict: {"MAE": value, "MSE": value, "R2": value}
# Round each to 3 decimals.
def evaluate_model(df, model):
    # TODO
    metrics = {}
    return metrics


# === 5. TASK 4 ===
# Create an insight function: return True if R² > 0.9, else False.
def strong_model(metrics):
    # TODO
    return False


# === 6. FINAL CHECK ===
solutions = {
    "generate_eval_data": generate_eval_data,
    "train_eval_model": train_eval_model,
    "evaluate_model": evaluate_model,
    "strong_model": strong_model
}

check_quiz(quiz_id=4, name=your_id, solutions=solutions)
