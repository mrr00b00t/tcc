import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv('breast_cancer_coimbra.csv')

y = df['Classification'].values
df = df.drop(columns=['Classification'])
X = df.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

clf = LinearSVC(tol=1e-5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(acc)

from sklearn.model_selection import GridSearchCV
  
# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']} 
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
  
# fitting the model for grid search
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(acc)
print(classification_report(y_test, y_pred))

from sklearn.manifold import TSNE
import seaborn as sns
tsne = TSNE(n_components=2, verbose=1, random_state=123)
sc = StandardScaler()
z = tsne.fit_transform(sc.fit_transform(X))
df = pd.DataFrame()
df["y"] = y
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]
fig = sns.scatterplot(x="comp-1", y="comp-2", hue=y,
                palette=sns.color_palette("hls", 2),
                data=df)

fig.get_figure().savefig('out.png')