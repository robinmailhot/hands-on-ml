from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as  np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score



mnist = fetch_openml('mnist_784', version=1)
Data, Target = mnist['data'], mnist['target']
Target = Target.astype(np.uint8)


first_number = Data[0]
first_number_reshaped = first_number.reshape(28,28)
plt.imshow(first_number_reshaped)
plt.axis('off')
plt.show()

X_train, X_test, Y_train, Y_test = Data[:60000], Data[60000:], Target[:60000], Target[60000:]

knn = KNeighborsClassifier(n_jobs=10)
knn.fit(X_train,Y_train)
result = cross_val_score(knn, X_train, Y_train,cv=3,scoring='accuracy')
#  This will do it to for a small test of how the API works for model fitting and data loading with sklearn