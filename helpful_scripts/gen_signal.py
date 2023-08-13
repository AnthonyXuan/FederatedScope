import numpy as np

def gen_sin(delta, h, w):
    arr = np.empty((h, w))
    for i in range(h):
        for j in range(w):
            arr[i, j] = delta * np.sin(2 * np.pi * j / w)
    np.save('array.npy', arr)
    return arr

gen_sin(10, 32, 32)