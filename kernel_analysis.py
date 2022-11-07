import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split


df = pd.read_csv('breast_cancer_coimbra.csv')

_y = df['Classification'].values
df = df.drop(columns=['Classification'])
_X = df.values

x = np.array([0.91707988, 0.56762625, 0.722352, 0.47838231, 0.17699625, 0.51083091, 0.13446851])

z = (x.copy() - 0.5) * 50.
z[-1] += 250
f = lambda M: (z[1] + z[2]*(M-z[0]) + z[3]*((M-z[0])**2)) / (1 + z[4]*(M-z[0]) + z[5]*((M-z[0])**2))

X_train, X_valid, y_train, y_valid = train_test_split(
    _X, _y,
    test_size=0.25,
    stratify=_y, 
    random_state=13
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid  = scaler.transform(X_valid)

svc = SVC(C=z[-1], kernel='precomputed')
kernel_train = f(np.dot(X_train, X_train.T))
svc.fit(kernel_train, y_train)
kernel_test = f(np.dot(X_valid, X_train.T))
y_pred = svc.predict(kernel_test)

print(balanced_accuracy_score(y_valid, y_pred))