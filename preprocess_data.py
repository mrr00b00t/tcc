import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import numpy as np

df = pd.read_csv('breast_cancer_coimbra.csv')

y = df['Classification'].values
df = df.drop(columns=['Classification'])
X = df.values

X_out = ','.join([f"[{','.join([str(e) for e in x])}]" for x in X])
print(f"X = np.array([{X_out}])")
print(f"y = np.array([{','.join([str(e) for e in y])}])")

quit()
skf = StratifiedKFold(n_splits=5, random_state=51, shuffle=True)

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    
    X_traini = ','.join([f"[{','.join([str(e) for e in x])}]" for x in X_train])
    print(f"X_train{i} = np.array([{X_traini}])")
    print(f"y_train{i} = np.array([{','.join([str(e) for e in y_train])}])")
    
    X_testi = ','.join([f"[{','.join([str(e) for e in x])}]" for x in X_test])
    print(f"X_test{i} = np.array([{X_testi}])")
    print(f"y_test{i} = np.array([{','.join([str(e) for e in y_test])}])")