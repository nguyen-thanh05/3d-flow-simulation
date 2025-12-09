import numpy as np

def harmonic_mean(array):
    """
    Helper fn to calculate the harmonic mean of a 1D numpy array.
    """
    n = len(array)
    if type(array) is not np.ndarray:
        array = np.array(array)
    return n / np.sum(1.0 / array)

def generate_active_mask(nx, ny, nz, null_blocks=None):
    """
    Generate a 3D boolean mask indicating active (True) and inactive (False) grid blocks.
    
    Parameters:
    -----------
    nx, ny, nz : int, number of grid blocks in x, y, z directions
    null_blocks : list of tuples, each tuple is (i, j, k) index of an inactive block
    
    Returns:
    --------
    active : 3D numpy array of shape (nx, ny, nz) with boolean values
    """
    active = np.ones((nx, ny, nz), dtype=bool)
    if null_blocks is not None and len(null_blocks) > 0:
        for null_block in null_blocks:
            i, j, k = null_block
            active[i, j, k] = False
    return active


def xyz_to_flat(x, y, z, nx, ny, nz):
    """
    Convert 3D coordinates (x, y, z) to flattened index.
    Assumes C-style (row-major) ordering: z varies fastest, then y, then x.
    
    Parameters:
    -----------
    x, y, z : int
        3D coordinates in the grid
    nx, ny, nz : int
        Grid dimensions
    
    Returns:
    --------
    int : Flattened index
    """
    return x * (ny * nz) + y * nz + z


def flat_to_xyz(idx, nx, ny, nz):
    """
    Convert flattened index to 3D coordinates (x, y, z).
    Assumes C-style (row-major) ordering: z varies fastest, then y, then x.
    
    Parameters:
    -----------
    idx : int
        Flattened index
    nx, ny, nz : int
        Grid dimensions
    
    Returns:
    --------
    tuple : (x, y, z) coordinates
    """
    z = idx % nz
    y = (idx // nz) % ny
    x = idx // (ny * nz)
    return x, y, z