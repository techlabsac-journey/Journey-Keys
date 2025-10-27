"""
DATA SCIENCE QUIZ 2 — DATA CLEANING & VISUALIZATION
TechLabs Aachen | Digital Shaper Program
---------------------------------------------------------
Instructions:
1. Fill in your name or email.
2. Complete all tasks.
3. Use the shared 'techlabs_checker.py' to verify and get your key.
"""

from ds_checker import check_quiz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === 1. Personal Info ===
your_id = "your_email_here"  # <-- change this!

# === 2. TASK 1 ===
# Create a DataFrame using your name as the seed:
#   - Column 'A': 5 random integers between 1 and 10
#   - Column 'B': same numbers + their index (0–4)
#   - Introduce one missing value (NaN) in column 'A' at position index 2
# HINT: use np.random.seed(sum of ASCII codes of your name)

def create_noisy_df(name):
    # TODO: your code here
    df = pd.DataFrame()
    return df


# === 3. TASK 2 ===
# Clean the DataFrame by filling the NaN in column 'A' with the column mean (rounded to 2 decimals).
# Return the cleaned DataFrame.

def clean_df(df):
    # TODO: your code here
    cleaned = df.copy()
    return cleaned


# === 4. TASK 3 ===
# Calculate and return the correlation coefficient between columns 'A' and 'B' (rounded to 3 decimals).

def correlation(df):
    # TODO: your code here
    corr = 0.0
    return corr


# === 5. TASK 4 ===
# Create a plot (Matplotlib) showing 'A' vs 'B' (scatter plot) with a title containing your name.
# Return the matplotlib Figure object.

def plot_data(df, name):
    # TODO: your code here
    fig = plt.figure()
    return fig


# === 6. FINAL CHECK ===
solutions = {
    "create_noisy_df": create_noisy_df,
    "clean_df": clean_df,
    "correlation": correlation,
    "plot_data": plot_data
}

check_quiz(quiz_id=2, name=your_id, solutions=solutions)
