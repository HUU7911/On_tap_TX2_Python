"""Bài 9 - Cây quyết định ID3 trên Car Evaluation."""
import csv, math
from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder

PATH=Path(__file__).parent/"data"/"car/car.data"
ATTRS=["buying","maint","doors","persons","lug_boot","safety"]
def load():
    with open(PATH,encoding="utf-8") as f:
        rows=list(csv.reader(f))
    return [dict(zip(ATTRS+["label"],r)) for r in rows]

def entropy(labels):
    n=len(labels); c=Counter(labels)
    return -sum((v/n)*math.log2(v/n) for v in c.values() if v)

def gain(rows,attr):
    base=entropy([r["label"] for r in rows])
    groups={}
    for r in rows: groups.setdefault(r[attr],[]).append(r)
    return base-sum(len(g)/len(rows)*entropy([r["label"] for r in g]) for g in groups.values())

def majority(rows): return Counter(r["label"] for r in rows).most_common(1)[0][0]

def id3(rows,attrs,max_depth=None,depth=0):
    labels=[r["label"] for r in rows]; default=majority(rows)
    if len(set(labels))==1: return labels[0],default
    if not attrs or (max_depth is not None and depth>=max_depth): return default,default
    best=max(attrs,key=lambda a:gain(rows,a))
    tree={}; remaining=[a for a in attrs if a!=best]
    for value in sorted(set(r[best] for r in rows)):
        subset=[r for r in rows if r[best]==value]
        tree[value]=id3(subset,remaining,max_depth,depth+1)[0]
    return {best:tree},default

def predict_one(tree,row,default):
    if not isinstance(tree,dict): return tree
    attr=next(iter(tree)); branch=tree[attr]
    value=row[attr]
    if value not in branch: return default
    return predict_one(branch[value],row,default)

def count_leaves(tree):
    if not isinstance(tree,dict): return 1
    return sum(count_leaves(v) for v in next(iter(tree.values())).values())

def print_tree(tree,indent=0):
    if not isinstance(tree,dict): print(" "*indent+"-> "+str(tree)); return
    attr=next(iter(tree))
    for value,sub in tree[attr].items():
        print(" "*indent+f"{attr} = {value}")
        print_tree(sub,indent+4)

if __name__=="__main__":
    rows=load(); tr,te=train_test_split(rows,test_size=.2,random_state=42,stratify=[r["label"] for r in rows])
    print("Gain tại gốc:")
    for a in ATTRS: print(a, gain(tr,a))
    for depth in (None,2,3,4):
        tree,default=id3(tr,ATTRS,depth)
        pred=[predict_one(tree,r,default) for r in te]
        print("\\nDepth:",depth,"Accuracy:",accuracy_score([r["label"] for r in te],pred),"Leaves:",count_leaves(tree))
        if depth is None: print_tree(tree)
    # Đối chiếu sklearn: encode từng cột phân loại.
    enc=OrdinalEncoder()
    xtr=enc.fit_transform([[r[a] for a in ATTRS] for r in tr])
    xte=enc.transform([[r[a] for a in ATTRS] for r in te])
    sk=DecisionTreeClassifier(criterion="entropy",random_state=42).fit(xtr,[r["label"] for r in tr])
    print("Sklearn accuracy:",accuracy_score([r["label"] for r in te],sk.predict(xte)))
