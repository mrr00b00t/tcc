import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

import locale


locale.setlocale(locale.LC_ALL, "pt_BR.utf8")

plt.rcParams.update({
    'axes.formatter.use_locale' : True,
})

plt.style.use('classic')

def get_figsize(
    columnwidth=4, wf=1.0, hf_rel=(5.0 ** 0.5 - 1.0) / 2.0, hf_abs=None, unit="inch"
):

    unit = unit.lower()
    conversion = dict(inch=1.0, mm=25.4, cm=2.54, pt=72.0,)

    if unit in conversion.keys():
        fig_width = columnwidth / conversion[unit]
        if hf_abs is not None:
            fig_height = hf_abs / conversion[unit]
    else:
        raise ValueError(f"unit deve ser: {conversion.keys()}")

    fig_width *= wf

    if hf_abs is None:
        fig_height = fig_width * hf_rel

    return (fig_width, fig_height)

plt.rcParams.update({
    'figure.figsize' : get_figsize(columnwidth=455.0, unit='pt'),
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# Generate dataset with 2 blobs
X, y = datasets.make_blobs(n_samples=100, centers=2, random_state=6, cluster_std=1.05)

# Train a linear SVM
svm_clf = SVC(kernel="linear", C=1)
svm_clf.fit(X, y)

# Function to plot decision boundary
def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]
    
    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]
    
    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin
    
    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)

# Plotting the decision boundary
#plt.figure(figsize=(12, 4))
plot_svc_decision_boundary(svm_clf, X[:, 0].min()-1, X[:, 0].max()+1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.tight_layout()
plt.savefig('svm-margem-rigida.pdf', format='pdf', bbox_inches='tight')
plt.clf()
