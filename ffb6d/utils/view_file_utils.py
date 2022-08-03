import pickle
from matplotlib import pyplot as plt
img = '/media/nachiket/Windows/Users/All Users/Documents/Datasets/LineMod/Linemod_preprocessed/renders/ape/13.pkl'

with open(img) as f:
    data = pickle.load(f)

print('Keys in loaded file: ',data.keys())

for k in data.keys():
    plt.imshow(data[k])
    plt.show()

