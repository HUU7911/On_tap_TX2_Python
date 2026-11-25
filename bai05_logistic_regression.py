"""Bài 5 - Logistic Regression tự cài đặt trên WDBC."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

PATH=Path(__file__).parent/"data"/"diagnostic"/"wdbc.data"
def load_data():
    df=pd.read_csv(PATH,header=None)
    y=(df.iloc[:,1]=="M").astype(int).to_numpy() # M=1 ác tính, B=0 lành tính
    return df.iloc[:,2:].to_numpy(float),y

class LogisticGD:
    def __init__(self,lr=.5,epochs=1000): self.lr,self.epochs=lr,epochs
    @staticmethod
    def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))
    def fit(self,X,y):
        self.w=np.zeros(X.shape[1]); self.b=0.; self.losses=[]
        n=len(y)
        for _ in range(self.epochs):
            p=self.sigmoid(X@self.w+self.b)
            eps=1e-12
            self.losses.append(-np.mean(y*np.log(p+eps)+(1-y)*np.log(1-p+eps)))
            self.w-=self.lr*(X.T@(p-y))/n
            self.b-=self.lr*np.mean(p-y)
        return self
    def predict_proba(self,X): return self.sigmoid(X@self.w+self.b)
    def predict(self,X,threshold=.5): return (self.predict_proba(X)>=threshold).astype(int)

def minmax_train(Xtr,Xte):
    mn=Xtr.min(axis=0); mx=Xtr.max(axis=0); den=mx-mn; den[den==0]=1
    return (Xtr-mn)/den,(Xte-mn)/den

def report(y,p):
    print("Accuracy:",accuracy_score(y,p),"Precision:",precision_score(y,p,zero_division=0),
          "Recall:",recall_score(y,p,zero_division=0),"\nConfusion matrix:\n",confusion_matrix(y,p))

if __name__=="__main__":
    X,y=load_data()
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    for label,cols in [("2 features",[0,1]),("30 features",list(range(30)))]:
        a,b=minmax_train(Xtr[:,cols],Xte[:,cols])
        model=LogisticGD().fit(a,ytr)
        print("\\n",label); report(yte,model.predict(b,.5))
        if label=="30 features":
            report(yte,model.predict(b,.3))
            import matplotlib.pyplot as plt
            plt.plot(model.losses); plt.xlabel("Epoch"); plt.ylabel("Cross entropy"); plt.show()
    sk=LogisticRegression(max_iter=3000).fit(*minmax_train(Xtr,Xte)[:1],ytr)
    print("Sklearn accuracy:",accuracy_score(yte,sk.predict(minmax_train(Xtr,Xte)[1])))
