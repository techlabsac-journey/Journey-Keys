"""
DEEP LEARNING QUIZ 3 — CNNs & IMAGE CLASSIFICATION
TechLabs Aachen | Digital Shaper Program
---------------------------------------------------------
"""

from dl_checker import check_quiz
import torch
import torch.nn as nn

your_id = "your_email_here"

# === TASK 1 ===
# Build a CNN model with:
# Conv2d(1, 4, 3), ReLU, Flatten, Linear(4*26*26, 2)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO
    def forward(self, x):
        # TODO
        return x

# === TASK 2 ===
# Create a dummy image tensor (1,1,28,28) filled with
# ASCII sum of your name modulo 255 divided by 255.
def generate_image(name):
    # TODO
    return torch.zeros((1,1,28,28))

# === TASK 3 ===
# Forward pass through model, return output tensor shape.
def forward_shape(model, img):
    # TODO
    return (0,)

# === TASK 4 ===
# Return number of parameters in model.
def count_params(model):
    # TODO
    return 0

solutions = {
    "SimpleCNN": SimpleCNN,
    "generate_image": generate_image,
    "forward_shape": forward_shape,
    "count_params": count_params
}

check_quiz(quiz_id=3, name=your_id, solutions=solutions)
