import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score


df = pd.read_csv('breast_cancer_coimbra.csv')

y = df['Classification'].values
df = df.drop(columns=['Classification'])
X = df.values

x = np.array([0.61017942, 0.63894261, 0.66357388, 0.64408081, 0.2479763,0.04069385,0.85769182, 0.48909921, 0.41017339, 0.12919023, 0.50397636, 0.1429224, 0.80208766])

scaler  = StandardScaler()
X = scaler.fit_transform(X)

x = (x - 0.5) * 25.
x[12] += 225
f = lambda y: (x[0] + x[1]*(y-x[11]) + x[2]*((y-x[11])**2) + x[3]*((y-x[11])**3) + x[4]*((y-x[11])**4) + x[5]*((y-x[11])**5)) / (1 + x[6]*(y-x[11]) + x[7]*((y-x[11])**2) + x[8]*((y-x[11])**3) + x[9]*((y-x[11])**4) + x[10]*((y-x[11])**5))

pca = PCA(n_components=2)
X = pca.fit_transform(X)


svc = SVC(C=x[12], kernel='precomputed', probability=True)
kernel_train = f(np.dot(X, X.T))
svc.fit(kernel_train, y)

h = .02  # step size in the mesh

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


kernel_test = f(np.dot(X, X.T))
y_pred = svc.predict(kernel_test)
print(accuracy_score(y, y_pred))

y_pred = SVC().fit(X, y).predict(X)
print(accuracy_score(y, y_pred))

kernel_test = f(np.dot(np.c_[xx.ravel(), yy.ravel()], X.T))

Z = svc.predict(kernel_test)

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.xlabel('PCA dim 1')
plt.ylabel('PCA dim 2')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title('Decision function')

plt.savefig(f"decision_function.png")
