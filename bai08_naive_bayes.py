"""Bài 8 - Categorical Naive Bayes với Laplace smoothing."""
import csv, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

DATA=Path(__file__).parent/"data"
def read_csv_rows(path):
    with open(path,encoding="utf-8") as f: return list(csv.reader(f))
class CategoricalNBManual:
    def __init__(self,alpha=1): self.alpha=alpha
    def fit(self,rows,y):
        self.classes=sorted(set(y)); self.class_count=Counter(y)
        self.values=[set(r[j] for r in rows if r[j]!="?") for j in range(len(rows[0]))]
        self.counts=defaultdict(lambda:defaultdict(Counter))
        for row,label in zip(rows,y):
            for j,v in enumerate(row):
                if v!="?": self.counts[label][j][v]+=1
        return self
    def predict_one(self,row):
        scores={}
        total=len(row)
        for c in self.classes:
            score=math.log(self.class_count[c]/sum(self.class_count.values()))
            for j,v in enumerate(row):
                if v=="?": continue
                # Mẫu số dùng số mẫu lớp + số giá trị đã biết của thuộc tính.
                V=max(1,len(self.values[j]))
                count=self.counts[c][j][v]
                score+=math.log((count+self.alpha)/(self.class_count[c]+self.alpha*V))
            scores[c]=score
        return max(scores,key=scores.get)
    def predict(self,rows): return [self.predict_one(r) for r in rows]

def evaluate(rows,y,indices=None):
    Xtr,Xte,ytr,yte=train_test_split(rows,y,test_size=.2,random_state=42,stratify=y)
    model=CategoricalNBManual().fit(Xtr,ytr)
    p=model.predict(Xte)
    print("Manual accuracy:",accuracy_score(yte,p))
    # sklearn: coi '?' là một category riêng theo hướng dẫn đề.
    enc=OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1)
    a=enc.fit_transform(Xtr); b=enc.transform(Xte)
    sk=CategoricalNB().fit(a,np.asarray(ytr))
    print("Sklearn CategoricalNB:",accuracy_score(yte,sk.predict(b)))

if __name__=="__main__":
    # Mushroom: agaricus-lepiota.data, nhãn cột đầu; 22 thuộc tính còn lại.
    mush=read_csv_rows(DATA/"mushroom/agaricus-lepiota.data")
    ym=[r[0] for r in mush]; Xm=[r[1:] for r in mush]
    for label,cols in [("3 attributes",[3,7,17]),("odor only",[4]),("all 22",list(range(22)))]:
        Xsel=[[r[i] for i in cols] for r in Xm]
        print("\\nMushroom",label); evaluate(Xsel,ym)
    # Congressional Voting Records: 16 votes + class label at final column.
    vote=read_csv_rows(DATA/"congressional/house-votes-84.data")
    yv=[r[0] for r in vote]; Xv=[r[1:] for r in vote]
    print("\\nCongressional Voting"); evaluate(Xv,yv)
