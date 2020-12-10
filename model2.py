import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score

wine = load_wine()

data = pd.DataFrame(data= np.c_[wine['data'], wine['target']],
                     columns= wine['feature_names'] + ['target'])
data.head()

X_train = data[:-20]
X_test = data[-20:]

y_train = X_train.target
y_test = X_test.target

X_train = X_train.drop('target',1)
X_test = X_test.drop('target',1)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("accuracy_score: %.2f"
      % accuracy_score(y_test, y_pred))



import json
data = [[14.34, 1.68, 2.7, 25.0, 98.0, 2.8, 1.31, 0.53, 2.7, 13.0, 0.57, 1.96, 660.0]]
j_data = json.dumps(data)
print(j_data)