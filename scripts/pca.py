from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

name = 'all_states.npy'
arr = np.load(name)
pca = PCA(n_components=2)
brr = pca.fit_transform(arr)

best = brr[:100]
others = brr[100:-100]
worst = brr[-100:]

def prt(a, file):
    with open(f'{file}.dat', 'w') as f:
        for x, y in a:
            print(x, y, file=f)

prt(best, 'best')
prt(worst, 'worst')
prt(others, 'others')

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)

# def plot(dt, **kwargs):
    # X = dt[:, 0]
    # Y = dt[:, 1]
    # plt.scatter(X, Y, **kwargs)

# plot(brr[100:-100], c='black', label='others')
# plot(brr[:100], c='blue', label='best')
# plot(brr[-100:], c='red', label='worst')
# plt.show()
