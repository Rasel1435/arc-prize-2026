import numpy as np
from scipy.ndimage import label


# --- Transformation Functions ---
def repeat_pattern(input_grid, n_times):
    """Tiles the input grid n_times in both dimensions."""
    arr = np.array(input_grid)
    output = np.tile(arr, (n_times, n_times))
    return output.tolist()

def translate_object(grid, obj_coords, dr, dc):
    """
    Moves an object by dr (rows) and dc (columns).
    dr: positive = down, negative = up
    dc: positive = right, negative = left
    """
    grid = np.array(grid)
    new_grid = np.zeros_like(grid)
    
    for r, c in obj_coords:
        new_r, new_c = r + dr, c + dc
        if 0 <= new_r < grid.shape[0] and 0 <= new_c < grid.shape[1]:
            new_grid[new_r, new_c] = grid[r, c]
            
    return new_grid.tolist()

def rotate_grid(grid, k=1):
    """Rotates grid 90 degrees k times."""
    return np.rot90(np.array(grid), k=k).tolist()

def flip_grid(grid, axis=0):
    """Flips grid: axis=0 is vertical (up-down), axis=1 is horizontal (left-right)."""
    if axis == 0:
        return np.flipud(np.array(grid)).tolist()
    else:
        return np.fliplr(np.array(grid)).tolist()

def kronecker_solve(grid):
    """Applies Kronecker product - essential for task 007bbfb7."""
    arr = np.array(grid)
    res = np.kron(arr > 0, arr)
    return res.tolist()


def apply_gravity(grid):
    """Moves all non-zero pixels to the bottom of the grid."""
    arr = np.array(grid)
    rows, cols = arr.shape
    new_grid = np.zeros_like(arr)
    for c in range(cols):
        col_values = arr[:, c]
        non_zeros = col_values[col_values != 0]
        for i, val in enumerate(reversed(non_zeros)):
            new_grid[rows - 1 - i, c] = val
    return new_grid.tolist()


def color_remapping(grid, train_tasks):
    """Checks for a consistent 1-to-1 color swap between input and output."""
    mapping = {}
    for task in train_tasks:
        inp = np.array(task['input']).flatten()
        out = np.array(task['output']).flatten()
        if len(inp) != len(out): return None
        for i, o in zip(inp, out):
            if i in mapping and mapping[i] != o: return None
            mapping[i] = o
    arr = np.array(grid)
    new_grid = np.zeros_like(arr)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            new_grid[r, c] = mapping.get(arr[r, c], 0)
    return new_grid.tolist()


# --- The Master Solver ---
def master_solver(train_tasks, test_input):
    """
    The Brain: Tests various logical transformations against training data
    to find a consistent rule, then applies it to the test data.
    """
    
    # Define candidate logics with descriptive labels to avoid lambda __name__ errors
    simple_logics = [
        ("Rotate 90", lambda x: rotate_grid(x, k=1)),
        ("Rotate 180", lambda x: rotate_grid(x, k=2)),
        ("Rotate 270", lambda x: rotate_grid(x, k=3)),
        ("Flip Vertical", lambda x: flip_grid(x, axis=0)),
        ("Flip Horizontal", lambda x: flip_grid(x, axis=1)),
        ("Repeat 2x", lambda x: repeat_pattern(x, 2)),
        ("Repeat 3x", lambda x: repeat_pattern(x, 3)),
        ("Kronecker Product", lambda x: kronecker_solve(x)),
        ("Gravity", lambda x: apply_gravity(x))
    ]
    
    # Check simple geometric/physical logics
    for label_name, logic_func in simple_logics:
        is_correct = True
        for example in train_tasks:
            try:
                if logic_func(example['input']) != example['output']:
                    is_correct = False
                    break
            except:
                is_correct = False
                break
        if is_correct:
            print(f"Success! Found winning logic: {label_name}")
            return logic_func(test_input)

    # Check Context-Heavy Logic (like Color Remapping)
    # We check this separately because it needs the training data to build its rule
    remapped = color_remapping(test_input, train_tasks)
    if remapped is not None:
        # Verify the remapping works on all training examples before returning
        is_correct = True
        for example in train_tasks:
            if color_remapping(example['input'], train_tasks) != example['output']:
                is_correct = False
                break
        if is_correct:
            print("Success! Found winning logic: Color Remapping")
            return remapped
            
    return "No winning logic found."