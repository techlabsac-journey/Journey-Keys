"""
TECHLABS CHECKER (Shared)
-------------------------------------------------
This file handles:
- Task validation for all Data Science quizzes
- Secure key generation per user
-------------------------------------------------
Usage:
1. Import this file in your quiz notebook:
   from techlabs_checker import check_quiz
2. Call:
   check_quiz(quiz_id=<number>, name=your_id, solutions=<dict>)
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import hashlib

# ========== TASK CHECKERS ==========

def check_dl_key1(name, solutions):
    """Check answers for DL Quiz 1"""
    score = 0
    torch.manual_seed(sum([ord(c) for c in name]))

    # expected tensor
    ascii_vals = [ord(c) % 10 for c in name]
    t_true = torch.tensor(ascii_vals, dtype=torch.float32)

    relu_true = F.relu(t_true)
    soft_true = torch.round(F.softmax(t_true, dim=0) * 1000) / 1000
    fwd_true = 2 * t_true + 3

    # Task 1
    try:
        if torch.equal(solutions["create_name_tensor"](name), t_true):
            score += 1
    except:
        pass

    # Task 2
    try:
        if torch.equal(solutions["apply_relu"](t_true), relu_true):
            score += 1
    except:
        pass

    # Task 3
    try:
        if torch.allclose(solutions["apply_softmax"](t_true), soft_true, atol=1e-3):
            score += 1
    except:
        pass

    # Task 4
    try:
        if torch.equal(solutions["forward_pass"](t_true), fwd_true):
            score += 1
    except:
        pass

    return score, 4

def check_dl_key2(name, solutions):
    score = 0
    torch.manual_seed(sum([ord(c) for c in name]))

    # Reference model
    x = torch.tensor([[0.0],[1.0],[2.0],[3.0]])
    y = 2*x + 1
    model_ref = nn.Linear(1,1)
    opt = optim.SGD(model_ref.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for _ in range(100):
        opt.zero_grad()
        loss = loss_fn(model_ref(x), y)
        loss.backward()
        opt.step()
    final_loss_true = round(loss.item(), 3)
    w_true = round(model_ref.weight.item(),1)
    b_true = round(model_ref.bias.item(),1)
    y_pred_true = round(model_ref(torch.tensor([[5.0]])).item(),1)

    # --- Task 1 ---
    try:
        model = solutions["SimpleModel"]()
        if hasattr(model, "forward"):
            score += 1
    except:
        pass

    # --- Task 2 ---
    try:
        if abs(solutions["train_model"](name) - final_loss_true) < 0.1:
            score += 1
    except:
        pass

    # --- Task 3 ---
    try:
        if all(abs(a - b) < 0.2 for a,b in zip(solutions["get_params"](model_ref),(w_true,b_true))):
            score += 1
    except:
        pass

    # --- Task 4 ---
    try:
        if abs(solutions["predict"](model_ref) - y_pred_true) < 0.2:
            score += 1
    except:
        pass

    return score, 4

def check_dl_key3(name, solutions):
    score = 0
    ascii_sum = sum([ord(c) for c in name])
    val = (ascii_sum % 255) / 255
    img_true = torch.ones((1,1,28,28)) * val

    class Ref(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1,4,3)
            self.relu = nn.ReLU()
            self.flat = nn.Flatten()
            self.fc = nn.Linear(4*26*26,2)
        def forward(self,x):
            x = self.relu(self.conv(x))
            x = self.flat(x)
            return self.fc(x)

    ref = Ref()
    out_true = ref(img_true)
    shape_true = tuple(out_true.shape)
    nparams_true = sum(p.numel() for p in ref.parameters())

    try:
        model = solutions["SimpleCNN"]()
        if isinstance(model, nn.Module):
            score += 1
    except:
        pass

    try:
        img = solutions["generate_image"](name)
        if torch.allclose(img, img_true, atol=1e-3):
            score += 1
    except:
        pass

    try:
        if solutions["forward_shape"](ref, img_true) == shape_true:
            score += 1
    except:
        pass

    try:
        if abs(solutions["count_params"](ref) - nparams_true) < 2:
            score += 1
    except:
        pass

    return score, 4

def check_dl_key4(name, solutions):
    score = 0
    n = len(name)
    logits_true = torch.tensor([[float(n), float(n//2), 0.0]])
    pred_true = torch.argmax(F.softmax(logits_true, dim=1)).item()
    acc_true = 2/3

    class Dummy(nn.Module): pass
    dummy = Dummy()

    try:
        if torch.equal(solutions["create_logits"](name), logits_true):
            score += 1
    except:
        pass

    try:
        if solutions["predict_class"](logits_true) == pred_true:
            score += 1
    except:
        pass

    try:
        reloaded = solutions["save_reload_model"](dummy)
        if isinstance(reloaded, nn.Module):
            score += 1
    except:
        pass

    try:
        if abs(solutions["compute_accuracy"]([1,0,2],[1,0,1]) - acc_true) < 1e-3:
            score += 1
    except:
        pass

    return score, 4

# ========== QUIZ MAPPING ==========

QUIZ_FUNCTIONS = {
    1: check_dl_key1,
    2: check_dl_key2,
    3: check_dl_key3,
    4: check_dl_key4
}


# ========== UNIVERSAL CHECKER ==========

def check_quiz(quiz_id: int, name: str, solutions: dict):
    """
    Main entry point for all quizzes.
    quiz_id: which quiz (1–4)
    name: user name or email
    solutions: dictionary of implemented functions
    """
    if quiz_id not in QUIZ_FUNCTIONS:
        print(f"❌ Quiz {quiz_id} not found.")
        return

    try:
        checker = QUIZ_FUNCTIONS[quiz_id]
        score, total = checker(name, solutions)
    except Exception as e:
        print("⚠️ Error during checking:", e)
        return

    # Generate unique key
    data = f"{name.lower()}_{score}_dataquiz{quiz_id}_secret2025"
    key = hashlib.sha256(data.encode()).hexdigest()[:12]

    print(f"✅ Quiz {quiz_id} complete! Score: {score}/{total}")
    print(f"🔑 Your unique key: {key}")
