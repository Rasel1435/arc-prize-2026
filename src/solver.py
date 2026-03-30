import numpy as np

def repeat_pattern(input_grid, n_times):
    arr = np.array(input_grid)
    output = np.tile(arr, (n_times, n_times))
    return output.tolist()