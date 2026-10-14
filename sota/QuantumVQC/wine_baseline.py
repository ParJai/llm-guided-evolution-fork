import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.datasets import load_wine
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from network import load_data, SEED

X_tr, X_va, X_te, y_tr, y_va, y_te = load_data()


def run(name, model, tr, va, te):
    model.fit(*tr)
    print(f"{name:34s} val {model.score(*va):.1%}   test {model.score(*te):.1%}")


# 1) Majority class
run("majority class", DummyClassifier(strategy="most_frequent"),
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te))

# 2) Logistic regression on the same 8 PCA features the circuit sees
run("logistic regression (8 PCA feats)", LogisticRegression(max_iter=1000),
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te))

# 3) Logistic regression on all 13 standardized features (same split)
X, y = load_wine(return_X_y=True)
X_a, X_t, y_a, y_t = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
X_r, X_v, y_r, y_v = train_test_split(X_a, y_a, test_size=0.25, random_state=SEED, stratify=y_a)
assert (y_r == y_tr).all() and (y_v == y_va).all() and (y_t == y_te).all(), "split mismatch"
sc = StandardScaler().fit(X_r)
run("logistic regression (13 feats)", LogisticRegression(max_iter=1000),
    (sc.transform(X_r), y_r), (sc.transform(X_v), y_v), (sc.transform(X_t), y_t))