"""Bài 6 - KNN phân lớp trên Iris và Wine."""
import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def distance(a,b,metric="euclidean"):
    if metric=="euclidean": return np.sqrt(np.sum((a-b)**2))
    if metric=="manhattan": return np.sum(np.abs(a-b))
    raise ValueError("metric phải là euclidean hoặc manhattan")

def knn_predict(Xtr,ytr,Xte,k,weighted=False,metric="euclidean"):
    out=[]
    for x in Xte:
        ds=np.array([distance(x,z,metric) for z in Xtr])
        ids=np.argsort(ds)[:k]
        scores={}
        for i in ids:
            label=ytr[i]; weight=(1/(ds[i]+1e-12)) if weighted else 1
            scores[label]=scores.get(label,0)+weight
        # Nếu hòa, lấy nhãn của láng giềng gần nhất trong nhóm hòa.
        best=max(scores.values())
        winners={c for c,v in scores.items() if np.isclose(v,best)}
        out.append(next(ytr[i] for i in ids if ytr[i] in winners))
    return np.asarray(out)

if __name__=="__main__":
    iris=load_iris(); X,y=iris.data,iris.target
    k_values=[1,3,5,7,9]; scores=[]
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    for k in k_values:
        fold=[]
        for tr,te in cv.split(X,y): fold.append(accuracy_score(y[te],knn_predict(X[tr],y[tr],X[te],k)))
        scores.append(np.mean(fold))
    print("Iris CV accuracy:",dict(zip(k_values,scores)))
    plt.plot(k_values,scores,marker="o"); plt.xlabel("k"); plt.ylabel("Mean accuracy"); plt.show()
    wine=load_wine(); X,y=wine.data,wine.target
    Xt,Xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    print("Wine unscaled:",accuracy_score(yv,knn_predict(Xt,yt,Xv,5)))
    scaler=StandardScaler().fit(Xt); a,b=scaler.transform(Xt),scaler.transform(Xv)
    print("Wine z-score:",accuracy_score(yv,knn_predict(a,yt,b,5)))
    print("Weighted KNN:",accuracy_score(yv,knn_predict(a,yt,b,5,weighted=True)))
    sk=KNeighborsClassifier(n_neighbors=5).fit(a,yt)
    print("Sklearn:",accuracy_score(yv,sk.predict(b)))
