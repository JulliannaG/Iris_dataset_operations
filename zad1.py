import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('iris.csv', delimiter=',', dtype=object)
print(data.shape)
column_names = data[0]
data = data[1:]
print(data.shape)
labels = data[:, -1]
features = data[:, :-1]

labels[labels=='setosa'] = 0
labels[labels=='versicolor'] = 1
labels[labels=='virginica'] = 2

features = features.astype(float)

fig, ax = plt.subplots(1,1,figsize=(7,7))
ax.scatter(features[:, 0], features[:, 1], c=labels)
plt.savefig('zad1.png')
plt.plot()

