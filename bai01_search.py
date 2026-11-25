"""Bài 1 - Linear Search và Binary Search. Đặt zoo.data, auto-mpg.data trong thư mục data/."""
import csv, re
from pathlib import Path

DATA = Path(__file__).parent / "data"

def load_zoo(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.reader(f):
            if r:
                rows.append({"name": r[0], "record": r[1:]})
    return rows

def linear_search_all(records, key):
    found, comparisons = [], 0
    for i, row in enumerate(records):
        comparisons += 1
        if row["name"].lower() == key.lower():
            found.append((i, row))
    return found, comparisons

def load_auto_mpg(path):
    rows = []
    # auto-mpg.data có 9 trường, tên xe ở cuối trong dấu ngoặc kép.
    pattern = re.compile(r'^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+"(.*)"\s*$')
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line)
            if not m:
                continue
            v = m.groups()
            rows.append({"mpg": v[0], "name": v[8], "raw": v})
    # Giữ bản ghi đầu tiên của mỗi tên xe.
    unique = {}
    for r in rows:
        unique.setdefault(r["name"], r)
    return sorted(unique.values(), key=lambda x: x["name"].lower())

def binary_iterative(a, name, trace=False):
    left, right, comparisons = 0, len(a)-1, 0
    while left <= right:
        mid = (left + right)//2
        comparisons += 1
        if trace: print(f"[left={left}, right={right}, mid={mid}] -> {a[mid]['name']}")
        k, mk = name.lower(), a[mid]["name"].lower()
        if mk == k: return mid, comparisons
        if mk < k: left = mid+1
        else: right = mid-1
    return -1, comparisons

def binary_recursive(a, name, left=0, right=None, trace=False, count=None):
    if count is None: count = [0]
    if right is None: right = len(a)-1
    if left > right: return -1, count[0]
    mid = (left+right)//2
    count[0] += 1
    if trace: print(f"[left={left}, right={right}, mid={mid}] -> {a[mid]['name']}")
    if a[mid]["name"].lower() == name.lower(): return mid, count[0]
    if a[mid]["name"].lower() < name.lower():
        return binary_recursive(a, name, mid+1, right, trace, count)
    return binary_recursive(a, name, left, mid-1, trace, count)

def lower_bound(a, value):
    l, r = 0, len(a)
    while l < r:
        m = (l+r)//2
        if float(a[m]["mpg"]) < value: l = m+1
        else: r = m
    return l

def upper_bound(a, value):
    l, r = 0, len(a)
    while l < r:
        m = (l+r)//2
        if float(a[m]["mpg"]) <= value: l = m+1
        else: r = m
    return l

if __name__ == "__main__":
    zoo = load_zoo(DATA/"zoo/zoo.data")
    for q in ("frog", "not_an_animal"):
        hits, c = linear_search_all(zoo, q)
        print(f"Linear search {q!r}: comparisons={c}, positions={[i for i,_ in hits]}")
        for i, row in hits: print(i, row)
    cars = load_auto_mpg(DATA/"auto_mpg/auto-mpg.data")
    print("Số tên xe duy nhất:", len(cars))
    query = cars[len(cars)//2]["name"]
    print("\\nBinary iterative:")
    print(binary_iterative(cars, query, True))
    print("\\nBinary recursive:")
    print(binary_recursive(cars, query, trace=True))
    ci, cr, n = 0, 0, len(cars)
    for car in cars:
        ci += binary_iterative(cars, car["name"])[1]
        cr += binary_recursive(cars, car["name"])[1]
    print("So sánh trung bình iterative/recursive:", ci/n, cr/n)
    by_mpg = sorted(cars, key=lambda x: float(x["mpg"]) if x["mpg"] != "?" else float("inf"))
    lo, hi = lower_bound(by_mpg, 25), upper_bound(by_mpg, 30)
    print("Xe có mpg trong [25,30]:")
    for car in by_mpg[lo:hi]: print(car["mpg"], car["name"])
