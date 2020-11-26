from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as  np



mnist = fetch_openml('mnist_784', version=1)
Data, Target = mnist['data'], mnist['target']
Target = Target.astype(np.uint8)


first_number = Data[0]
first_number_reshaped = first_number.reshape(28,28)
plt.imshow(first_number_reshaped)
plt.axis('off')
plt.show()

X_train, X_test, Y_train, Y_test = Data[:60000], Data[60000:], Target[:60000], Target[60000:]
print(mnist.keys())
