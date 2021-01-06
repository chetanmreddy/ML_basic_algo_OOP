# -*- coding: utf-8 -*-
# @Author: chetan
# @Date:   2021-01-06 01:27:22
# @Last Modified by:   chetan
# @Last Modified time: 2021-01-06 01:34:13

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from kmeans import K_Means

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11]])
colors = 10*["g","r","c","b","k"]


clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker="o", color="k", s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)
        
plt.show()