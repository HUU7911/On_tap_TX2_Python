"""Bài 7 - KNN hồi quy trên Abalone."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

PATH=Path(__file__).parent/"data"/"abalone"/"abalone.data"
COLS=["sex","length","diameter","height","whole_weight","shucked_weight","viscera_weight","shell_weight","rings"]

def knn_reg_predict(Xtr,ytr,Xte,k):
    result=[]
    for x in Xte:
        d=np.sqrt(np.sum((Xtr-x)**2,axis=1))
        result.append(np.mean(ytr[np.argsort(d)[:k]]))
    return np.array(result)

if __name__=="__main__":
    df=pd.read_csv(PATH,names=COLS)
    X=df.iloc[:,1:8].to_numpy(float); y=df["rings"].to_numpy(float)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    mn=Xtr.min(axis=0); mx=Xtr.max(axis=0); den=mx-mn; den[den==0]=1
    a=(Xtr-mn)/den; b=(Xte-mn)/den
    ks=range(1,31); maes=[]
    for k in ks: maes.append(mean_absolute_error(yte,knn_reg_predict(a,ytr,b,k)))
    plt.plot(list(ks),maes,marker="."); plt.xlabel("k"); plt.ylabel("Test MAE"); plt.title("KNN Regression"); plt.show()
    best=int(np.argmin(maes)+1)
    print("k có MAE thấp nhất:",best,"MAE:",min(maes))
    pred=knn_reg_predict(a,ytr,b,best)
    lr=LinearRegression().fit(a,ytr)
    print("Linear regression MAE:",mean_absolute_error(yte,lr.predict(b)))
    print("5 mẫu đầu: rings thật | rings dự đoán | tuổi ước lượng")
    for actual,p in zip(yte[:5],pred[:5]): print(actual,p,p+1.5)
