# === Data Science Quiz 1: Foundations ===
from ds_checker import check_quiz

"""
DATA SCIENCE QUIZ 1 — FOUNDATIONS (PERSONALIZED)
TechLabs Aachen | Digital Shaper Program
---------------------------------------------------------
Instructions:
1. Replace the string below with your full name or email.
2. Complete all the coding tasks.
3. Run the last cell to generate your unique key.
"""

# === Personal Info ===
# Use the same email you signed up to techlabs.
your_id = "alice@example.com"

# --- Tasks ---

import numpy as np
import pandas as pd

# === TASK 1 ===
# Create a NumPy array of the ASCII codes of all characters in your name.
# For example, if your_id = "bob", output should be array([98, 111, 98])

def name_to_ascii(name):
    return np.array([ord(c) for c in name])

# === TASK 2 ===
# Create a DataFrame with two columns: "Char" and "Code"
# - "Char" should be the characters of your name (from Task 1)
# - "Code" should be the ASCII values (from Task 1)
# Example for "bob":
#    Char  Code
# 0    b     98
# 1    o    111
# 2    b     98

def create_char_df(name):
    return pd.DataFrame({"Char": list(name), "Code": [ord(c) for c in name]})

# === TASK 3 ===
# Compute the mean ASCII value of all characters in your name.
# Return it as a float rounded to 2 decimal places.

def mean_ascii(name):
    return round(np.mean([ord(c) for c in name]), 2)

# === TASK 4 ===
# Create a reproducible "hash number" from your name:
# Steps:
#   - Get ASCII codes from Task 1
#   - Multiply each by its index position (0-based)
#   - Return the sum of those products
# Example for "bob" -> [98, 111, 98]
# Calculation: 98*0 + 111*1 + 98*2 = 307

def name_hash_number(name):
    codes = [ord(c) for c in name]
    return sum([i * codes[i] for i in range(len(codes))])

# --- Final check ---
solutions = {
    "name_to_ascii": name_to_ascii,
    "create_char_df": create_char_df,
    "mean_ascii": mean_ascii,
    "name_hash_number": name_hash_number
}

check_quiz(quiz_id=1, name=your_id, solutions=solutions)
