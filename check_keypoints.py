import numpy as np
X = np.load('keypoints_output/X_json.npy')
y = np.load('keypoints_output/y_json.npy')
print('X shape:', X.shape)
print('y:', y)
print('max:', X.max())
print('min:', X.min())
print('sample[0] frame0:', X[0][0])
print('sample[5] frame0:', X[5][0])
