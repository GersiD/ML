from sklearn import svm
import pandas as pd
import os

# Load the iris dataset
assert os.getcwd().endswith('svm') # NOTE: Need to run file from the svm directory
df = pd.read_csv('../data/iris.csv')
X = df[['sepal length', 'sepal width']]
Y = df['class']
Y = Y.map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

clf = svm.SVC()
clf.fit(X, Y)

# Pretty plot
import matplotlib.pyplot as plt
import numpy as np
plt.scatter(X['sepal length'], X['sepal width'], c=Y, cmap=plt.cm.coolwarm)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title("SVM Classifier on Iris Dataset")
# Plot the decision boundary
x_min, x_max = X['sepal length'].min() - 1, X['sepal length'].max() + 1
y_min, y_max = X['sepal width'].min() - 1, X['sepal width'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.savefig('plots/svm.pdf')
plt.show()
