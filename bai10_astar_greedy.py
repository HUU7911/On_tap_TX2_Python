"""Bài 10 - A* và Greedy Best-First trên đồ thị k-NN Seeds."""
import csv, heapq, math
from pathlib import Path
import numpy as np
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

PATH=Path(__file__).parent/"data"/"seeds/seeds_dataset.txt"
def load():
    # UCI Seeds: 7 đặc trưng và class; file thường phân tách whitespace.
    return np.loadtxt(PATH)

def build_graph(X,k=6):
    n=len(X); graph={i:[] for i in range(n)}
    d=np.sqrt(((X[:,None,:]-X[None,:,:])**2).sum(axis=2))
    for i in range(n):
        nearest=np.argsort(d[i])[1:k+1]
        for j in nearest:
            w=float(d[i,j])
            graph[i].append((int(j),w))
            graph[j].append((i,w))
    # Loại cạnh trùng do thêm vô hướng.
    for i in graph:
        graph[i]=list({j:w for j,w in graph[i]}.items())
    return graph

def connected(graph,start,goal):
    seen={start}; q=deque([start])
    while q:
        u=q.popleft()
        for v,_ in graph[u]:
            if v not in seen: seen.add(v); q.append(v)
    return goal in seen,len(seen)

def search(graph,start,goal,h,greedy=False):
    pq=[(h[start] if greedy else h[start],0,start)]
    parent={}; best_g={start:0.}; closed=set(); expanded=0
    while pq:
        _,g,u=heapq.heappop(pq)
        if u in closed: continue
        closed.add(u); expanded+=1
        if u==goal:
            path=[u]
            while u in parent: u=parent[u]; path.append(u)
            path.reverse()
            return path,best_g[goal],expanded
        for v,cost in graph[u]:
            ng=best_g[u]+cost
            if ng<best_g.get(v,float("inf")):
                best_g[v]=ng; parent[v]=u
                priority=h[v] if greedy else ng+h[v]
                heapq.heappush(pq,(priority,ng,v))
    return [],float("inf"),expanded

if __name__=="__main__":
    data=load(); X=data[:,:2] # tọa độ (area, perimeter)
    start=int(np.argmin(X[:,0])); goal=int(np.argmax(X[:,0]))
    graph=build_graph(X,6)
    ok,visited=connected(graph,start,goal)
    print("Connected:",ok,"BFS visited:",visited,"start:",start,"goal:",goal)
    if not ok: raise RuntimeError("Đồ thị chưa liên thông; kiểm tra cách dựng k-NN.")
    h={i:float(np.linalg.norm(X[i]-X[goal])) for i in range(len(X))}
    for name,greedy in (("A*",False),("Greedy Best-First",True)):
        path,cost,expanded=search(graph,start,goal,h,greedy)
        print(name,"path:",path,"cost:",cost,"expanded:",expanded)
    # Dijkstra là A* với h(n)=0.
    path,cost,expanded=search(graph,start,goal,{i:0. for i in range(len(X))})
    print("Dijkstra:",path,"cost:",cost,"expanded:",expanded)
