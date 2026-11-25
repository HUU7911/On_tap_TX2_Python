
import shlex
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


PATH = Path(__file__).parent / "data" / "auto_mpg" / "auto-mpg.data"


def load_data(path):
    cols = [
        "mpg", "cylinders", "displacement", "horsepower",
        "weight", "acceleration", "model_year", "origin", "car_name"
    ]

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")

    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                values = shlex.split(line)
            except ValueError as e:
                raise ValueError(f"Lỗi tại dòng {line_number}: {e}")

            if len(values) != len(cols):
                raise ValueError(
                    f"Dòng {line_number} có {len(values)} cột, "
                    f"yêu cầu {len(cols)} cột."
                )

            rows.append(values)

    df = pd.DataFrame(rows, columns=cols)

    for col in cols[:-1]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["horsepower", "mpg", "weight"])


def closed_form(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        raise ValueError("X không có độ biến thiên.")

    w = numerator / denominator
    b = y_mean - w * x_mean

    return w, b


def gradient_descent(x, y, lr=0.1, epochs=1000):
    mu = x.mean()
    sigma = x.std()

    if sigma == 0:
        raise ValueError("Độ lệch chuẩn của X bằng 0.")

    z = (x - mu) / sigma
    w, b = 0.0, 0.0

    for _ in range(epochs):
        pred = w * z + b
        dw = np.mean((pred - y) * z)
        db = np.mean(pred - y)

        w -= lr * dw
        b -= lr * db

    return w / sigma, b - w * mu / sigma


if __name__ == "__main__":
    df = load_data(PATH)

    print("=" * 60)
    print("THÔNG TIN DATASET AUTO MPG")
    print("=" * 60)

    print("Số lượng bản ghi:", len(df))
    print("\n5 dòng dữ liệu đầu tiên:")
    print(df.head())

    print("\nThông tin thống kê:")
    print(df[["mpg", "weight"]].describe())

    X = df["weight"].to_numpy(dtype=float)
    y = df["mpg"].to_numpy(dtype=float)

    w, b = closed_form(X, y)

    print("\n" + "=" * 60)
    print("1. CLOSED-FORM SOLUTION")
    print("=" * 60)

    print(f"Hệ số w: {w:.6f}")
    print(f"Hệ số b: {b:.6f}")
    print(f"Phương trình: MPG = {w:.6f} * Weight + {b:.6f}")

    wg, bg = gradient_descent(X, y, lr=0.1, epochs=1000)

    print("\n" + "=" * 60)
    print("2. GRADIENT DESCENT")
    print("=" * 60)

    print(f"Hệ số w: {wg:.6f}")
    print(f"Hệ số b: {bg:.6f}")
    print(f"Phương trình: MPG = {wg:.6f} * Weight + {bg:.6f}")

    Xtr, Xte, ytr, yte = train_test_split(
        X.reshape(-1, 1),
        y,
        test_size=0.2,
        random_state=42
    )

    print("\n" + "=" * 60)
    print("3. TRAIN / TEST SPLIT")
    print("=" * 60)

    print("Số lượng Train:", len(Xtr))
    print("Số lượng Test:", len(Xte))

    wt, bt = closed_form(Xtr[:, 0], ytr)
    pred = wt * Xte[:, 0] + bt

    mse = mean_squared_error(yte, pred)
    r2 = r2_score(yte, pred)

    print("\n" + "=" * 60)
    print("4. MODEL EVALUATION")
    print("=" * 60)

    print(f"Test MSE: {mse:.6f}")
    print(f"Test RMSE: {np.sqrt(mse):.6f}")
    print(f"Test R2 Score: {r2:.6f}")

    print("\n" + "=" * 60)
    print("5. PREDICTION")
    print("=" * 60)

    mpg_2500 = wt * 2500 + bt
    mpg_4000 = wt * 4000 + bt

    print(f"Dự đoán xe có trọng lượng 2500 lbs: {mpg_2500:.2f} MPG")
    print(f"Dự đoán xe có trọng lượng 4000 lbs: {mpg_4000:.2f} MPG")

    xx = np.linspace(X.min(), X.max(), 100)
    yy = wt * xx + bt

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, s=20, alpha=0.5, label="Actual data")
    plt.plot(xx, yy, linewidth=2, label="Regression line")

    plt.xlabel("Weight (lbs)")
    plt.ylabel("Miles per Gallon (MPG)")
    plt.title("Simple Linear Regression: MPG vs Weight")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    sk = LinearRegression()
    sk.fit(Xtr, ytr)

    print("\n" + "=" * 60)
    print("6. SCIKIT-LEARN LINEAR REGRESSION")
    print("=" * 60)

    print(f"Sklearn w: {sk.coef_[0]:.6f}")
    print(f"Sklearn b: {sk.intercept_:.6f}")

    print("\nSo sánh hệ số trên tập Train:")
    print(f"Closed-form w: {wt:.6f}")
    print(f"Sklearn w:     {sk.coef_[0]:.6f}")
    print(f"Closed-form b: {bt:.6f}")
    print(f"Sklearn b:     {sk.intercept_:.6f}")

    print("\nHoàn thành bài 3 - Linear Regression!")
