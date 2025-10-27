"""
DEEP LEARNING QUIZ 2 — MODEL CREATION & TRAINING
TechLabs Aachen | Digital Shaper Program
---------------------------------------------------------
Instructions:
1. Fill in your name/email.
2. Complete all tasks.
3. Run final cell to check.
"""

from dl_checker import check_quiz
import torch
import torch.nn as nn
import torch.optim as optim

your_id = "your_email_here"  # <-- change this

# === TASK 1 ===
# Create a PyTorch nn.Module named "SimpleModel" with:
# - one Linear layer (1 input, 1 output)
# - forward: y = linear(x)
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO
    def forward(self, x):
        # TODO
        return x


# === TASK 2 ===
# Train the model for 100 epochs on synthetic data:
# y = 2*x + 1, x in [0,1,2,3]
# Use MSELoss and SGD (lr=0.01). Return final loss (rounded to 3 decimals).
def train_model(name):
    # TODO
    final_loss = 0.0
    return final_loss


# === TASK 3 ===
# Return model weight and bias as tuple (w,b) rounded to 1 decimal.
def get_params(model):
    # TODO
    return (0.0, 0.0)


# === TASK 4 ===
# Predict y for x=5 using the trained model.
def predict(model):
    # TODO
    return 0.0


solutions = {
    "SimpleModel": SimpleModel,
    "train_model": train_model,
    "get_params": get_params,
    "predict": predict
}

check_quiz(quiz_id=2, name=your_id, solutions=solutions)
