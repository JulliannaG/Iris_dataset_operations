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

fig, ax = plt.subplots(4,4 ,figsize=(14,14))

for i in range(4):
    for j in range(4):
        ax[i, j].scatter(features[:,i ], features[:, j], c=labels)
        ax[i, j].set_xlabel(column_names[i])
        ax[i, j].set_ylabel(column_names[j])
plt.tight_layout()
plt.savefig('zad2.png')
#plt.show()