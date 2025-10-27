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

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import hashlib

# ========== TASK CHECKERS ==========

def check_ds_key1(name, solutions):
    """Check answers for Data Science Quiz 1"""
    score = 0

    arr_true = np.array([ord(c) for c in name])
    df_true = pd.DataFrame({"Char": list(name), "Code": arr_true})
    mean_true = round(np.mean(arr_true), 2)
    hash_true = int(sum([i * arr_true[i] for i in range(len(arr_true))]))

    # Task 1
    try:
        if np.array_equal(solutions["name_to_ascii"](name), arr_true):
            score += 1
    except:
        pass

    # Task 2
    try:
        df = solutions["create_char_df"](name)
        if df.equals(df_true):
            score += 1
    except:
        pass

    # Task 3
    try:
        if abs(solutions["mean_ascii"](name) - mean_true) < 1e-6:
            score += 1
    except:
        pass

    # Task 4
    try:
        if solutions["name_hash_number"](name) == hash_true:
            score += 1
    except:
        pass

    return score, 4

def check_ds_key2(name, solutions):
    """Check answers for Data Science Quiz 2"""
    score = 0

    # deterministic seed
    seed = sum([ord(c) for c in name])
    np.random.seed(seed)

    # expected DataFrame
    A = np.random.randint(1, 11, 5).astype(float)
    B = A + np.arange(5)
    A[2] = np.nan
    df_true = pd.DataFrame({"A": A, "B": B})

    # expected cleaned DataFrame
    A_clean = A.copy()
    A_clean[np.isnan(A_clean)] = round(np.nanmean(A_clean), 2)
    df_clean_true = pd.DataFrame({"A": A_clean, "B": B})

    corr_true = round(df_clean_true["A"].corr(df_clean_true["B"]), 3)

    # --- check Task 1 ---
    try:
        df = solutions["create_noisy_df"](name)
        if df.equals(df_true):
            score += 1
    except:
        pass

    # --- check Task 2 ---
    try:
        cleaned = solutions["clean_df"](df_true)
        if cleaned.equals(df_clean_true):
            score += 1
    except:
        pass

    # --- check Task 3 ---
    try:
        if abs(solutions["correlation"](df_clean_true) - corr_true) < 1e-6:
            score += 1
    except:
        pass

    # --- check Task 4 ---
    try:
        fig = solutions["plot_data"](df_clean_true, name)
        if fig and hasattr(fig, "axes"):
            title = fig.axes[0].get_title().lower()
            if name.lower() in title:
                score += 1
    except:
        pass

    return score, 4

def check_ds_key3(name, solutions):
    """Check answers for Data Science Quiz 3"""
    score = 0

    seed = sum([ord(c) for c in name])
    np.random.seed(seed)

    x = np.random.randint(1, 21, 10)
    noise = np.random.randint(-2, 3, 10)
    y = 3 * x + 5 + noise
    df_true = pd.DataFrame({"x": x, "y": y})

    # True extended df
    df_feat = df_true.copy()
    df_feat["x_squared"] = df_feat["x"] ** 2

    # True model
    X = df_feat[["x", "x_squared"]]
    model = LinearRegression().fit(X, y)
    coef_x, coef_x2 = model.coef_
    intercept = model.intercept_
    y_pred_true = round(model.predict([[10, 10**2]])[0], 2)

    # --- Task 1 ---
    try:
        df = solutions["generate_dataset"](name)
        if df.equals(df_true):
            score += 1
    except:
        pass

    # --- Task 2 ---
    try:
        df2 = solutions["add_feature"](df_true)
        if df2.equals(df_feat):
            score += 1
    except:
        pass

    # --- Task 3 ---
    try:
        result = solutions["train_model"](df_feat)
        if (
            abs(result[0] - coef_x) < 0.05
            and abs(result[1] - coef_x2) < 0.05
            and abs(result[2] - intercept) < 0.05
        ):
            score += 1
    except:
        pass

    # --- Task 4 ---
    try:
        if abs(solutions["predict_value"](df_feat) - y_pred_true) < 0.05:
            score += 1
    except:
        pass

    return score, 4

def check_ds_key4(name, solutions):
    """Check answers for Data Science Quiz 4"""
    score = 0

    seed = sum([ord(c) for c in name])
    np.random.seed(seed)

    x = np.arange(15)
    noise = np.random.randint(-3, 4, 15)
    y = 4 * x + 10 + noise
    df_true = pd.DataFrame({"x": x, "y": y})

    # Train true model
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    mae_true = round(mean_absolute_error(y, y_pred), 3)
    mse_true = round(mean_squared_error(y, y_pred), 3)
    r2_true = round(r2_score(y, y_pred), 3)
    metrics_true = {"MAE": mae_true, "MSE": mse_true, "R2": r2_true}
    strong_true = r2_true > 0.9

    # --- Task 1 ---
    try:
        df = solutions["generate_eval_data"](name)
        if df.equals(df_true):
            score += 1
    except:
        pass

    # --- Task 2 ---
    try:
        model2 = solutions["train_eval_model"](df_true)
        if hasattr(model2, "predict"):
            score += 1
    except:
        pass

    # --- Task 3 ---
    try:
        metrics = solutions["evaluate_model"](df_true, model)
        ok = all(abs(metrics[k] - metrics_true[k]) < 0.05 for k in metrics_true)
        if ok:
            score += 1
    except:
        pass

    # --- Task 4 ---
    try:
        if solutions["strong_model"](metrics_true) == strong_true:
            score += 1
    except:
        pass

    return score, 4

# ========== QUIZ MAPPING ==========

QUIZ_FUNCTIONS = {
    1: check_ds_key1,
    2: check_ds_key2,
    3: check_ds_key3,
    4: check_ds_key4
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
