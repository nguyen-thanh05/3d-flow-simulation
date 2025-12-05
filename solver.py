import numpy as np
from utils import harmonic_mean


def calculate_transmissibility(k_x, k_y, k_z, nx, ny, nz, dx, dy, dz, null_blocks=None):
    """
    Calculate transmissibility in x, y and z directions.
    Tx includes Tx_right (i+1/2, jk) and Tx_left (i-1/2, jk)
    Ty includes Ty_front (i, j+1/2, k) and Ty_back (i, j-1/2, k)
    Tz includes Tz_up (i, j, k+1/2) and Tz_down (i, j, k-1/2)
    
    dx, dy, dz are vector of size (nx,), (ny,), (nz,) respectively.
    null_blocks: array of shape (# null, 3) with [i, j, k] coordinates of inactive cells
    """
    Tx_right = np.zeros((nx, ny, nz))
    Tx_left = np.zeros((nx, ny, nz))
    Ty_front = np.zeros((nx, ny, nz))
    Ty_back = np.zeros((nx, ny, nz))
    Tz_up = np.zeros((nx, ny, nz))
    Tz_down = np.zeros((nx, ny, nz))
    
    # Create active block mask (True = active, False = null/inactive)
    active = np.ones((nx, ny, nz), dtype=bool)
    if null_blocks is not None and len(null_blocks) > 0:
        for null_block in null_blocks:
            i, j, k = null_block
            active[i, j, k] = False
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Skip if current block is inactive
                if not active[i, j, k]:
                    continue
                
                # Tx_right
                if i < nx - 1 and active[i + 1, j, k]:
                    Tx_right[i, j, k] = harmonic_mean([
                        k_x[i, j, k] * dy[j] * dz[k] / dx[i],
                        k_x[i + 1, j, k] * dy[j] * dz[k] / dx[i + 1]
                    ])
                
                # Tx_left
                if i > 0 and active[i - 1, j, k]:
                    Tx_left[i, j, k] = harmonic_mean([
                        k_x[i, j, k] * dy[j] * dz[k] / dx[i],
                        k_x[i - 1, j, k] * dy[j] * dz[k] / dx[i - 1]
                    ])
                
                # Ty_front
                if j < ny - 1 and active[i, j + 1, k]:
                    Ty_front[i, j, k] = harmonic_mean([
                        k_y[i, j, k] * dx[i] * dz[k] / dy[j],
                        k_y[i, j + 1, k] * dx[i] * dz[k] / dy[j + 1]
                    ])
                
                # Ty_back
                if j > 0 and active[i, j - 1, k]:
                    Ty_back[i, j, k] = harmonic_mean([
                        k_y[i, j, k] * dx[i] * dz[k] / dy[j],
                        k_y[i, j - 1, k] * dx[i] * dz[k] / dy[j - 1]
                    ])
                
                # Tz_up
                if k < nz - 1 and active[i, j, k + 1]:
                    Tz_up[i, j, k] = harmonic_mean([
                        k_z[i, j, k] * dx[i] * dy[j] / dz[k],
                        k_z[i, j, k + 1] * dx[i] * dy[j] / dz[k + 1]
                    ])
                
                # Tz_down
                if k > 0 and active[i, j, k - 1]:
                    Tz_down[i, j, k] = harmonic_mean([
                        k_z[i, j, k] * dx[i] * dy[j] / dz[k],
                        k_z[i, j, k - 1] * dx[i] * dy[j] / dz[k - 1]
                    ])
    
    return Tx_right, Tx_left, Ty_front, Ty_back, Tz_up, Tz_down, active
    

def calculate_pore_volume(nx, ny, nz, dx, dy, dz, porosity, null_blocks=None):
    """
    Calculate pore volume for each cell in the grid.
    
    dx, dy, dz are vector of size (nx,), (ny,), (nz,) respectively.
    porosity: 3D array of shape (nx, ny, nz)
    null_blocks: array of shape (# null, 3) with [i, j, k] coordinates of inactive cells
    """
    pore_volume = np.zeros((nx, ny, nz))
    
    # Create active block mask (True = active, False = null/inactive)
    active = np.ones((nx, ny, nz), dtype=bool)
    if null_blocks is not None and len(null_blocks) > 0:
        for null_block in null_blocks:
            i, j, k = null_block
            active[i, j, k] = False
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if active[i, j, k]:
                    cell_volume = dx[i] * dy[j] * dz[k]
                    pore_volume[i, j, k] = porosity[i, j, k] * cell_volume
    
    return pore_volume