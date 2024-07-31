import numpy as np
import matplotlib.pyplot as plt
import scipy

data = np.loadtxt('iris.csv', delimiter=',', dtype=object)
print(data.shape)
column_names = data[0]
data = data[1:]
print(data.shape)
labels = data[:, -1]
petal_length = data[:, 2]
petal_width = data[:, 3]
features = np.column_stack((petal_length, petal_width))

labels[labels=='setosa'] = 0
labels[labels=='versicolor'] = 1
labels[labels=='virginica'] = 2

features = features.astype(float)

new_petal = [3.1, 1.2]

centroids = [np.mean(features[labels==i], axis=0) for i in range(3)]
centroids = np.array(centroids)

distances = scipy.spatial.distance.cdist([new_petal], centroids)
print('Distances: ' + str(distances))
prediction = np.argmin(distances)

if prediction==0:
    name='setosa'
elif prediction==1:
    name="versicolor"
else:
    name="virginica"
print('Prediction: ' + name)

fig, ax = plt.subplots(1,1,figsize=(7,7))
ax.scatter(centroids[:, 0], centroids[:, 1], alpha=1, c=[0, 1, 2])
ax.scatter(features[:,0], features[:,1], alpha=0.15, c=labels)
ax.scatter(new_petal[0], new_petal[1], marker='x', c='black')
ax.set_xlabel('petal_length')
ax.set_ylabel('petal_width')
plt.savefig('zad3.png')
plt.plot()

