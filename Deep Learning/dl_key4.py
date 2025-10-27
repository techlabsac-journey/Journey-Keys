"""
DEEP LEARNING QUIZ 4 — EVALUATION & DEPLOYMENT
TechLabs Aachen | Digital Shaper Program
---------------------------------------------------------
"""

from dl_checker import check_quiz
import torch
import torch.nn as nn
import torch.nn.functional as F

your_id = "your_email_here"

# === TASK 1 ===
# Create dummy logits tensor of size (1,3) with values based on your name length.
# First value = len(name), second = len(name)//2, third = 0.
def create_logits(name):
    # TODO
    return torch.zeros(1,3)

# === TASK 2 ===
# Apply softmax and return predicted class index.
def predict_class(logits):
    # TODO
    return 0

# === TASK 3 ===
# Save and reload a model using torch.save/load.
def save_reload_model(model):
    # TODO
    return model

# === TASK 4 ===
# Given true=[1,0,2], pred=[1,0,1], compute accuracy.
def compute_accuracy(true, pred):
    # TODO
    return 0.0

solutions = {
    "create_logits": create_logits,
    "predict_class": predict_class,
    "save_reload_model": save_reload_model,
    "compute_accuracy": compute_accuracy
}

check_quiz(quiz_id=104, name=your_id, solutions=solutions)
