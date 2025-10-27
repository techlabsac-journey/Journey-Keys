"""
DEEP LEARNING QUIZ 1 — FUNDAMENTALS
TechLabs Aachen | Digital Shaper Program
---------------------------------------------------------
Instructions:
1. Enter your name or email.
2. Complete the coding tasks.
3. Use `check_quiz(quiz_id=101, ...)` to get your unique key.
"""

from dl_checker import check_quiz
import torch
import torch.nn.functional as F

# === 1. Personal Info ===
your_id = "your_email_here"  # <-- change this

# === 2. TASK 1 ===
# Create a 1D tensor of size 5 using ASCII codes of your name modulo 10.
# Example: for "bob" -> [8,1,8,0,1] (depending on your name length)
def create_name_tensor(name):
    # TODO
    t = torch.tensor([])
    return t


# === 3. TASK 2 ===
# Apply ReLU activation to the tensor and return the result.
def apply_relu(tensor):
    # TODO
    return tensor


# === 4. TASK 3 ===
# Compute the softmax of the tensor (dim=0), rounded to 3 decimals.
def apply_softmax(tensor):
    # TODO
    return tensor


# === 5. TASK 4 ===
# Compute manual forward pass of y = 2*x + 3, using your tensor.
def forward_pass(tensor):
    # TODO
    return tensor


# === 6. FINAL CHECK ===
solutions = {
    "create_name_tensor": create_name_tensor,
    "apply_relu": apply_relu,
    "apply_softmax": apply_softmax,
    "forward_pass": forward_pass
}

check_quiz(quiz_id=1, name=your_id, solutions=solutions)
