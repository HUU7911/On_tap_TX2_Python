"""Bài 4 - Hồi quy tuyến tính đa biến trên Abalone."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

PATH = Path(__file__).parent/"data"/"abalone"/"abalone.data"
COLS = ["sex","length","diameter","height","whole_weight","shucked_weight","viscera_weight","shell_weight","rings"]

def load_data():
    return pd.read_csv(PATH, names=COLS)

def fit_normal_equation(X,y):
    A = np.c_[X, np.ones(len(X))]
    theta = np.linalg.solve(A.T@A, A.T@y)
    return theta[:-1], theta[-1]

def fit_gd(X,y,lr=.05,epochs=3000):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd==0] = 1
    Z=(X-mu)/sd
    ym, ys = y.mean(), y.std()
    yn=(y-ym)/ys
    A=np.c_[Z,np.ones(len(Z))]
    theta=np.zeros(A.shape[1]); losses=[]
    for _ in range(epochs):
        err=A@theta-yn
        theta-=lr*(A.T@err)/len(y)
        losses.append(np.mean(err**2))
    w=theta[:-1]*ys/sd
    b=ym+ys*theta[-1]-np.dot(w,mu)
    return w,b,losses

if __name__ == "__main__":
    df=load_data()
    y=df["whole_weight"].to_numpy(float)
    X=df[["length","diameter","height"]].to_numpy(float)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    w,b=fit_normal_equation(Xtr,ytr)
    p=Xte@w+b
    print("3 features MSE/R2:",mean_squared_error(yte,p),r2_score(yte,p),"theta:",w,b)
    wg,bg,loss=fit_gd(Xtr,ytr)
    print("Gradient Descent MSE:",mean_squared_error(yte,Xte@wg+bg))
    plt.plot(loss); plt.xlabel("Epoch"); plt.ylabel("MSE (normalized)"); plt.title("Gradient Descent"); plt.show()
    # So sánh chỉ dùng length trên cùng train/test.
    wl,bl=fit_normal_equation(Xtr[:,[0]],ytr)
    pl=Xte[:,0]*wl[0]+bl
    print("Length-only MSE/R2:",mean_squared_error(yte,pl),r2_score(yte,pl))
    # One-hot Sex (M/F/I), fit cùng tập chia.
    sex=pd.get_dummies(df["sex"],dtype=float).to_numpy()
    Xall=np.c_[X,sex]
    xa,xb,ya,yb=train_test_split(Xall,y,test_size=.2,random_state=42)
    ws,bs=fit_normal_equation(xa,ya)
    print("Thêm one-hot Sex MSE/R2:",mean_squared_error(yb,xb@ws+bs),r2_score(yb,xb@ws+bs))
    from sklearn.linear_model import LinearRegression
    sk=LinearRegression().fit(Xtr,ytr)
    print("Sklearn MSE:",mean_squared_error(yte,sk.predict(Xte)))
