"""Make blobs machine learning using SVM
Date: 20.10.2021
Part of Simplearn Machine Learning Program
Done By: Sofien Abidi"""

#Import standards library
import numpy as np
import matplotlib.pyplot as plt
#Import datest and model selection
from sklearn import svm
from sklearn.datasets import make_blobs
#Creation 40 sepearable points
X, y = make_blobs(n_samples=40, centers=2, random_state=20)

#Fit the model
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X,y)
plt.scatter(X[:,0],X[:,1], c=y, s=30, cmap=plt.cm.Paired)
#plt.show()
#Predict unknown data
newData = [[3,4],[5,6]]
print(clf.predict(newData))

#Plot decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

#Create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy,xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

#Plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1,0,1], alpha=0.5, linestyles=['--','-','--'])

#Plot supporrt vector
ax.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=100, linewidth=1, facecolors='none')
plt.show()

