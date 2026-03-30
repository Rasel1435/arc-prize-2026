import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def load_arc_data(data_path):
    with open(data_path) as f:
        return json.load(f)

def plot_grid(grid, title="Grid"):
    cmap = colors.ListedColormap(['#000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
                                  '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.BoundaryNorm(np.arange(-0.5, 10, 1), cmap.N)
    plt.imshow(grid, cmap=cmap, norm=norm)
    plt.title(title)
    plt.axis('off')