"""Bài 2 - Perceptron tự cài đặt. Đặt data_banknote_authentication.txt trong data/."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron as SkPerceptron

DATA = Path(__file__).parent/"data"/"banknote+authentication/data_banknote_authentication.txt"

def load_data(path):
    a = np.loadtxt(path, delimiter=",")
    X, y = a[:, :4], a[:, 4].astype(int)
    # Theo quy ước đề: 0 thật -> +1; 1 giả -> -1.
    return X, np.where(y == 0, 1, -1)

class Perceptron:
    def __init__(self, eta=0.1, epochs=50):
        self.eta, self.epochs = eta, epochs
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1]); self.b = 0.0; self.errors = []
        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                pred = 1 if np.dot(xi, self.w)+self.b >= 0 else -1
                if pred != target:
                    self.w += self.eta*target*xi
                    self.b += self.eta*target
                    errors += 1
            self.errors.append(errors)
        return self
    def predict(self, X):
        return np.where(X@self.w+self.b >= 0, 1, -1)

if __name__ == "__main__":
    X, y = load_data(DATA)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
    # Perceptron nhạy với thang đo; chuẩn hóa giúp các đặc trưng cân bằng.
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    model = Perceptron().fit(Xtr_s, ytr)
    print("Accuracy 4 features:", accuracy_score(yte, model.predict(Xte_s)))
    plt.plot(range(1, 51), model.errors, marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Số mẫu dự đoán sai"); plt.title("Perceptron - 4 features"); plt.grid(); plt.show()

    X2 = X[:, :2]
    X2tr, X2te, y2tr, y2te = train_test_split(X2, y, test_size=.2, random_state=42)
    sc2 = StandardScaler().fit(X2tr)
    a, b = sc2.transform(X2tr), sc2.transform(X2te)
    m2 = Perceptron().fit(a, y2tr)
    print("Accuracy 2 features:", accuracy_score(y2te, m2.predict(b)))
    # Vẽ đường phân tách trong không gian đã chuẩn hóa.
    plt.figure()
    for label, marker in ((1, "o"), (-1, "x")):
        plt.scatter(a[y2tr==label,0], a[y2tr==label,1], marker=marker, label=str(label))
    xx = np.linspace(a[:,0].min(), a[:,0].max(), 100)
    if abs(m2.w[1]) > 1e-12:
        plt.plot(xx, -(m2.w[0]*xx+m2.b)/m2.w[1], label="Decision boundary")
    plt.legend(); plt.xlabel("variance (scaled)"); plt.ylabel("skewness (scaled)"); plt.show()

    sk = SkPerceptron(max_iter=50, eta0=.1, tol=None, random_state=42).fit(Xtr_s, ytr)
    print("Sklearn accuracy:", accuracy_score(yte, sk.predict(Xte_s)))
