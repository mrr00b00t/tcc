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

# Generate non-linearly separable data
X, y = datasets.make_moons(n_samples=100, noise=0.1, random_state=1)

# Train a non-linear SVM with RBF kernel
svm_clf = SVC(kernel="rbf", C=1, gamma='auto')  # Setting the kernel to 'rbf' and gamma to 'auto' for demonstration
svm_clf.fit(X, y)

# Function to plot decision boundary and highlight support vectors
def plot_svc_decision_boundary(svm_clf):
    h = .02  # Step size in the mesh
    # Create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = svm_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    
    # Highlight the support vectors
    plt.scatter(svm_clf.support_vectors_[:, 0], svm_clf.support_vectors_[:, 1], s=100, 
               linewidth=1, facecolors='none', edgecolors='k')
    
    # Plot the dataset
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', marker='o')
    #plt.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    #plt.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)

# Plotting the decision boundary and support vectors
plot_svc_decision_boundary(svm_clf)
plt.xlabel('Atributo 1')
plt.ylabel('Atributo 2')
plt.tight_layout()
plt.savefig('svm-nao-linear.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
plt.clf()