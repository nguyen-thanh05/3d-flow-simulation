import numpy as np
from utils import harmonic_mean, generate_active_mask, xyz_to_flat, flat_to_xyz


def calculate_transmissibility(permeability, grid_config, null_blocks=None):
    """
    Calculate transmissibility at cell interfaces using harmonic mean.
    Returns 6 arrays for interfaces: right, left, front, back, up, down.
    """
    k_x, k_y, k_z = permeability["k_x"], permeability["k_y"], permeability["k_z"]
    nx, ny, nz = grid_config["nx"], grid_config["ny"], grid_config["nz"]
    dx, dy, dz = grid_config["dx"], grid_config["dy"], grid_config["dz"]
    
    # Initialize transmissibility arrays
    T = {
        'right': np.zeros((nx, ny, nz)),
        'left': np.zeros((nx, ny, nz)),
        'front': np.zeros((nx, ny, nz)),
        'back': np.zeros((nx, ny, nz)),
        'up': np.zeros((nx, ny, nz)),
        'down': np.zeros((nx, ny, nz))
    }
    
    active = generate_active_mask(nx, ny, nz, null_blocks)
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not active[i, j, k]:
                    continue
                
                # X-direction transmissibilities: T = k * (A / L)
                if i < nx - 1 and active[i + 1, j, k]:
                    A_yz = dy[j] * dz[k]
                    T_current = k_x[i, j, k] * A_yz / dx[i]
                    T_neighbor = k_x[i + 1, j, k] * A_yz / dx[i + 1]
                    T['right'][i, j, k] = harmonic_mean([T_current, T_neighbor])
                
                if i > 0 and active[i - 1, j, k]:
                    A_yz = dy[j] * dz[k]
                    T_current = k_x[i, j, k] * A_yz / dx[i]
                    T_neighbor = k_x[i - 1, j, k] * A_yz / dx[i - 1]
                    T['left'][i, j, k] = harmonic_mean([T_current, T_neighbor])
                
                # Y-direction transmissibilities
                if j < ny - 1 and active[i, j + 1, k]:
                    A_xz = dx[i] * dz[k]
                    T_current = k_y[i, j, k] * A_xz / dy[j]
                    T_neighbor = k_y[i, j + 1, k] * A_xz / dy[j + 1]
                    T['front'][i, j, k] = harmonic_mean([T_current, T_neighbor])
                
                if j > 0 and active[i, j - 1, k]:
                    A_xz = dx[i] * dz[k]
                    T_current = k_y[i, j, k] * A_xz / dy[j]
                    T_neighbor = k_y[i, j - 1, k] * A_xz / dy[j - 1]
                    T['back'][i, j, k] = harmonic_mean([T_current, T_neighbor])
                
                # Z-direction transmissibilities
                if k < nz - 1 and active[i, j, k + 1]:
                    A_xy = dx[i] * dy[j]
                    T_current = k_z[i, j, k] * A_xy / dz[k]
                    T_neighbor = k_z[i, j, k + 1] * A_xy / dz[k + 1]
                    T['up'][i, j, k] = harmonic_mean([T_current, T_neighbor])
                
                if k > 0 and active[i, j, k - 1]:
                    A_xy = dx[i] * dy[j]
                    T_current = k_z[i, j, k] * A_xy / dz[k]
                    T_neighbor = k_z[i, j, k - 1] * A_xy / dz[k - 1]
                    T['down'][i, j, k] = harmonic_mean([T_current, T_neighbor])
    
    return T
    

def calculate_pore_volume(nx, ny, nz, dx, dy, dz, porosity, null_blocks=None):
    """
    Calculate pore volume: PV = phi * V_bulk for each active cell.
    """
    active = generate_active_mask(nx, ny, nz, null_blocks)
    
    # Compute bulk volume for each cell: V_bulk = dx × dy × dz
    dx_3d = dx[:, None, None]  # (nx, 1, 1)
    dy_3d = dy[None, :, None]  # (1, ny, 1)
    dz_3d = dz[None, None, :]  # (1, 1, nz)
    bulk_volume = dx_3d * dy_3d * dz_3d  # (nx, ny, nz)
    
    # Apply porosity and mask inactive cells
    pore_volume = porosity * bulk_volume
    pore_volume[~active] = 0.0
    
    return pore_volume


def calculate_mobility(rel_perm, pvt_table, P, P_c, S_w, depth_top, dz, null_blocks=None):
    """
    Calculate phase mobility at cell interfaces using upstream weighting.
    Mobility: lambda = kr / (mu * B)
    Upstream weighting: use properties from cell with higher potential.
    """
    nx, ny, nz = P.shape
    active = generate_active_mask(nx, ny, nz, null_blocks)
    
    # Get fluid properties at current pressure and saturation
    B_w, B_o = pvt_table.get_B_w(P), pvt_table.get_B_o(P)
    mu_w, mu_o = pvt_table.get_mu_w(P), pvt_table.get_mu_o(P)
    rho_w, rho_o = pvt_table.get_density_w(P), pvt_table.get_density_o(P)
    kr_w, kr_o = rel_perm.kr1(S_w), rel_perm.kr2(S_w)
    
    # Precompute cell-center depths
    depth_center = depth_top + np.cumsum(dz) - dz / 2
    
    # Initialize mobility arrays
    lambda_w = {dir: np.zeros((nx, ny, nz)) for dir in ['right', 'left', 'front', 'back', 'up', 'down']}
    lambda_o = {dir: np.zeros((nx, ny, nz)) for dir in ['right', 'left', 'front', 'back', 'up', 'down']}
    
    g = 9.81  # gravitational constant
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not active[i, j, k]:
                    continue
                
                # X-direction interfaces
                if i < nx - 1 and active[i + 1, j, k]:
                    dPhi_w = P[i + 1, j, k] - P[i, j, k]
                    dPhi_o = (P[i + 1, j, k] + P_c[i + 1, j, k]) - (P[i, j, k] + P_c[i, j, k])
                    
                    i_up_w = (i + 1) if dPhi_w >= 0 else i
                    i_up_o = (i + 1) if dPhi_o >= 0 else i
                    lambda_w['right'][i, j, k] = kr_w[i_up_w, j, k] / (mu_w[i_up_w, j, k] * B_w[i_up_w, j, k])
                    lambda_o['right'][i, j, k] = kr_o[i_up_o, j, k] / (mu_o[i_up_o, j, k] * B_o[i_up_o, j, k])
                
                if i > 0 and active[i - 1, j, k]:
                    dPhi_w = P[i - 1, j, k] - P[i, j, k]
                    dPhi_o = (P[i - 1, j, k] + P_c[i - 1, j, k]) - (P[i, j, k] + P_c[i, j, k])
                    
                    i_up_w = (i - 1) if dPhi_w >= 0 else i
                    i_up_o = (i - 1) if dPhi_o >= 0 else i
                    lambda_w['left'][i, j, k] = kr_w[i_up_w, j, k] / (mu_w[i_up_w, j, k] * B_w[i_up_w, j, k])
                    lambda_o['left'][i, j, k] = kr_o[i_up_o, j, k] / (mu_o[i_up_o, j, k] * B_o[i_up_o, j, k])
                
                # Y-direction interfaces
                if j < ny - 1 and active[i, j + 1, k]:
                    dPhi_w = P[i, j + 1, k] - P[i, j, k]
                    dPhi_o = (P[i, j + 1, k] + P_c[i, j + 1, k]) - (P[i, j, k] + P_c[i, j, k])
                    
                    j_up_w = (j + 1) if dPhi_w >= 0 else j
                    j_up_o = (j + 1) if dPhi_o >= 0 else j
                    lambda_w['front'][i, j, k] = kr_w[i, j_up_w, k] / (mu_w[i, j_up_w, k] * B_w[i, j_up_w, k])
                    lambda_o['front'][i, j, k] = kr_o[i, j_up_o, k] / (mu_o[i, j_up_o, k] * B_o[i, j_up_o, k])
                
                if j > 0 and active[i, j - 1, k]:
                    dPhi_w = P[i, j - 1, k] - P[i, j, k]
                    dPhi_o = (P[i, j - 1, k] + P_c[i, j - 1, k]) - (P[i, j, k] + P_c[i, j, k])
                    
                    j_up_w = (j - 1) if dPhi_w >= 0 else j
                    j_up_o = (j - 1) if dPhi_o >= 0 else j
                    lambda_w['back'][i, j, k] = kr_w[i, j_up_w, k] / (mu_w[i, j_up_w, k] * B_w[i, j_up_w, k])
                    lambda_o['back'][i, j, k] = kr_o[i, j_up_o, k] / (mu_o[i, j_up_o, k] * B_o[i, j_up_o, k])
                
                # Z-direction interfaces (with gravity)
                if k < nz - 1 and active[i, j, k + 1]:
                    # Potential: P - rho * g * depth
                    Phi_w_up = P[i, j, k + 1] - rho_w[i, j, k + 1] * g * depth_center[k + 1]
                    Phi_w_curr = P[i, j, k] - rho_w[i, j, k] * g * depth_center[k]
                    Phi_o_up = (P[i, j, k + 1] + P_c[i, j, k + 1]) - rho_o[i, j, k + 1] * g * depth_center[k + 1]
                    Phi_o_curr = (P[i, j, k] + P_c[i, j, k]) - rho_o[i, j, k] * g * depth_center[k]
                    
                    dPhi_w = Phi_w_up - Phi_w_curr
                    dPhi_o = Phi_o_up - Phi_o_curr
                    
                    k_up_w = (k + 1) if dPhi_w >= 0 else k
                    k_up_o = (k + 1) if dPhi_o >= 0 else k
                    lambda_w['up'][i, j, k] = kr_w[i, j, k_up_w] / (mu_w[i, j, k_up_w] * B_w[i, j, k_up_w])
                    lambda_o['up'][i, j, k] = kr_o[i, j, k_up_o] / (mu_o[i, j, k_up_o] * B_o[i, j, k_up_o])
                
                if k > 0 and active[i, j, k - 1]:
                    Phi_w_down = P[i, j, k - 1] - rho_w[i, j, k - 1] * g * depth_center[k - 1]
                    Phi_w_curr = P[i, j, k] - rho_w[i, j, k] * g * depth_center[k]
                    Phi_o_down = (P[i, j, k - 1] + P_c[i, j, k - 1]) - rho_o[i, j, k - 1] * g * depth_center[k - 1]
                    Phi_o_curr = (P[i, j, k] + P_c[i, j, k]) - rho_o[i, j, k] * g * depth_center[k]
                    
                    dPhi_w = Phi_w_down - Phi_w_curr
                    dPhi_o = Phi_o_down - Phi_o_curr
                    
                    k_up_w = (k - 1) if dPhi_w >= 0 else k
                    k_up_o = (k - 1) if dPhi_o >= 0 else k
                    lambda_w['down'][i, j, k] = kr_w[i, j, k_up_w] / (mu_w[i, j, k_up_w] * B_w[i, j, k_up_w])
                    lambda_o['down'][i, j, k] = kr_o[i, j, k_up_o] / (mu_o[i, j, k_up_o] * B_o[i, j, k_up_o])
    
    return {
        'lambda_w': lambda_w,
        'lambda_o': lambda_o
    }           
                    
                

def add_well_model(rate_map, bottomhole_pressure_map, J, lhs_matrix, rhs_vector, 
                      pvt_table, P_map, P_c_matrix,
                      rel_perm, S1_matrix,
                      nx, ny, nz, dt):
    if rate_map is not None:
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if rate_map[i, j, k] != 0:
                        B_w = pvt_table.get_B_w(P_map[i, j, k])
                        idx = xyz_to_flat(i, j, k, nx, ny, nz)
                        rhs_vector[idx] += dt * B_w * rate_map[i, j, k]
    
    if bottomhole_pressure_map is not None:
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if bottomhole_pressure_map[i, j, k] != 0:
                        B_w = pvt_table.get_B_w(P_map[i, j, k])
                        B_o = pvt_table.get_B_o(P_map[i, j, k])
                        krw = rel_perm.kr1(S1_matrix[i, j, k])
                        kro = rel_perm.kr2(S1_matrix[i, j, k])
                        idx = xyz_to_flat(i, j, k, nx, ny, nz)
                        
                        lambda_1 = krw / (pvt_table.get_mu_w(P_map[i, j, k]) * B_w)
                        lambda_2 = kro / (pvt_table.get_mu_o(P_map[i, j, k]) * B_o)
                        
                        lhs_matrix[idx, idx] += dt * B_w * lambda_1 * J + dt * B_o * lambda_2 * J
                        rhs_vector[idx] += dt * B_w * lambda_1 * J * bottomhole_pressure_map[i, j, k] + \
                                            dt * B_o * lambda_2 * J * (bottomhole_pressure_map[i, j, k] - P_c_matrix[i, j, k])
    return lhs_matrix, rhs_vector



def assemble_lhs_matrix(rel_perm, pvt_table, 
                        S_w, P, P_c_fn,
                        transmissibility,
                        nx, ny, nz, V_pore,
                        dx, dy, dz, dt, depth_top):
    """
    Assemble LHS matrix A for pressure equation: A·P^(n+1) = b
    Includes accumulation and flow terms with upstream mobility weighting.
    """
    N = nx * ny * nz
    A = np.zeros((N, N))
    
    # Calculate mobilities and capillary pressure at explicit time level
    P_c = P_c_fn(S_w)
    mob = calculate_mobility(rel_perm, pvt_table, P, P_c, S_w, depth_top, dz)
    
    # Extract mobility arrays
    lambda_w = mob['lambda_w']
    lambda_o = mob['lambda_o']
    
    # Neighbor offsets: [di, dj, dk]
    neighbors = {
        'right': (1, 0, 0),
        'left': (-1, 0, 0),
        'front': (0, 1, 0),
        'back': (0, -1, 0),
        'up': (0, 0, 1),
        'down': (0, 0, -1)
    }
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                idx = xyz_to_flat(i, j, k, nx, ny, nz)
                
                # Accumulation term: V_p * (c_f + S_w*c_w + S_o*c_o)
                c_w = pvt_table.get_compressibility_w(P[i, j, k])
                c_o = pvt_table.get_compressibility_o(P[i, j, k])
                c_f = pvt_table.get_compressibility_f(P[i, j, k])
                S_o = 1 - S_w[i, j, k]
                
                accum = V_pore[i, j, k] * (c_f + S_w[i, j, k] * c_w + S_o * c_o)
                A[idx, idx] += accum
                
                # Flow terms: dt * (B_w*lambda_w + B_o*lambda_o) * T
                B_w = pvt_table.get_B_w(P[i, j, k])
                B_o = pvt_table.get_B_o(P[i, j, k])
                
                for direction, (di, dj, dk) in neighbors.items():
                    i_nb, j_nb, k_nb = i + di, j + dj, k + dk
                    
                    # Check if neighbor exists
                    if not (0 <= i_nb < nx and 0 <= j_nb < ny and 0 <= k_nb < nz):
                        continue
                    
                    # Total mobility: lambda_total = B_w*lambda_w + B_o*lambda_o
                    lambda_total = B_w * lambda_w[direction][i, j, k] + B_o * lambda_o[direction][i, j, k]
                    flux_coeff = dt * lambda_total * transmissibility[direction][i, j, k]
                    
                    # Diagonal: accumulates all outflow coefficients
                    A[idx, idx] += flux_coeff
                    
                    # Off-diagonal: negative for neighbor contribution
                    idx_nb = xyz_to_flat(i_nb, j_nb, k_nb, nx, ny, nz)
                    A[idx, idx_nb] -= flux_coeff
    
    return A


def assemble_rhs_vector(P_old, S_w, rel_perm, pvt_table, 
                        Q_w, Q_o, V_pore, P_c_fn,
                        transmissibility,
                        dx, dy, dz, depth_top,
                        nx, ny, nz, dt):
    """
    Assemble RHS vector b for pressure equation: A·P^(n+1) = b
    Includes: accumulation at t^n, gravity terms, capillary pressure gradients, sources/sinks.
    """
    N = nx * ny * nz
    b = np.zeros(N)
    
    # Calculate mobilities and capillary pressure
    P_c = P_c_fn(S_w)
    mob = calculate_mobility(rel_perm, pvt_table, P_old, P_c, S_w, depth_top, dz)
    lambda_w = mob['lambda_w']
    lambda_o = mob['lambda_o']
    
    # Precompute cell-center depths
    depth_center = depth_top + np.cumsum(dz) - dz / 2
    
    g = 9.81
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                idx = xyz_to_flat(i, j, k, nx, ny, nz)
                
                # Get fluid properties at (i,j,k)
                c_w = pvt_table.get_compressibility_w(P_old[i, j, k])
                c_o = pvt_table.get_compressibility_o(P_old[i, j, k])
                c_f = pvt_table.get_compressibility_f(P_old[i, j, k])
                B_w = pvt_table.get_B_w(P_old[i, j, k])
                B_o = pvt_table.get_B_o(P_old[i, j, k])
                rho_w = pvt_table.get_density_w(P_old[i, j, k])
                rho_o = pvt_table.get_density_o(P_old[i, j, k])
                
                S_o = 1 - S_w[i, j, k]
                
                # 1. Accumulation term: V_p * (c_f + S_w*c_w + S_o*c_o) * P^n
                accum = V_pore[i, j, k] * (c_f + S_w[i, j, k] * c_w + S_o * c_o) * P_old[i, j, k]
                b[idx] = accum
                
                # 2. Gravity terms (z-direction only)
                gamma_w = rho_w * g
                gamma_o = rho_o * g
                
                if k < nz - 1:
                    dz = depth_center[k] - depth_center[k + 1]  # Current - up
                    lambda_total = B_w * lambda_w['up'][i, j, k] * gamma_w + B_o * lambda_o['up'][i, j, k] * gamma_o
                    b[idx] += dt * lambda_total * transmissibility['up'][i, j, k] * dz
                
                if k > 0:
                    dz = depth_center[k] - depth_center[k - 1]  # Current - down
                    lambda_total = B_w * lambda_w['down'][i, j, k] * gamma_w + B_o * lambda_o['down'][i, j, k] * gamma_o
                    b[idx] += dt * lambda_total * transmissibility['down'][i, j, k] * dz
                
                # 3. Capillary pressure gradient terms (oil phase only)
                # Term: -dt * B_o * lambda_o * T * dP_c
                
                # X-direction
                if i < nx - 1:
                    dP_c = P_c[i, j, k] - P_c[i + 1, j, k]
                    b[idx] -= dt * B_o * lambda_o['right'][i, j, k] * transmissibility['right'][i, j, k] * dP_c
                
                if i > 0:
                    dP_c = P_c[i, j, k] - P_c[i - 1, j, k]
                    b[idx] -= dt * B_o * lambda_o['left'][i, j, k] * transmissibility['left'][i, j, k] * dP_c
                
                # Y-direction
                if j < ny - 1:
                    dP_c = P_c[i, j, k] - P_c[i, j + 1, k]
                    b[idx] -= dt * B_o * lambda_o['front'][i, j, k] * transmissibility['front'][i, j, k] * dP_c
                
                if j > 0:
                    dP_c = P_c[i, j, k] - P_c[i, j - 1, k]
                    b[idx] -= dt * B_o * lambda_o['back'][i, j, k] * transmissibility['back'][i, j, k] * dP_c
                
                # Z-direction
                if k < nz - 1:
                    dP_c = P_c[i, j, k] - P_c[i, j, k + 1]
                    b[idx] -= dt * B_o * lambda_o['up'][i, j, k] * transmissibility['up'][i, j, k] * dP_c
                
                if k > 0:
                    dP_c = P_c[i, j, k] - P_c[i, j, k - 1]
                    b[idx] -= dt * B_o * lambda_o['down'][i, j, k] * transmissibility['down'][i, j, k] * dP_c
                
                # 4. Source/sink terms
                b[idx] += dt * (B_w * Q_w[i, j, k] + B_o * Q_o[i, j, k])
    
    return b

def get_new_saturation(S_w_old, P_new, P_old,
                       Q_w,
                       transmissibility,
                       P_c_fn,
                       rel_perm, pvt_table,
                       V_pore, depth_top,
                       dx, dy, dz, dt,
                       nx, ny, nz):
    """
    Update water saturation using explicit saturation equation:
    S_w^(n+1) = S_w^n * (1 - (c_f + c_w)*dP) - (dt/V_p) * Σ(B_w*lambda_w*T*∇phi_w) + (dt/V_p)*B_w*Q_w
    """
    S_w_new = np.zeros((nx, ny, nz))
    
    # Calculate mobilities at new pressure level
    P_c = P_c_fn(S_w_old)
    mob = calculate_mobility(rel_perm, pvt_table, P_new, P_c, S_w_old, depth_top, dz)
    lambda_w = mob['lambda_w']
    
    # Precompute cell-center depths
    depth_center = depth_top + np.cumsum(dz) - dz / 2
    
    g = 9.81
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Get fluid properties at new pressure
                c_w = pvt_table.get_compressibility_w(P_new[i, j, k])
                c_o = pvt_table.get_compressibility_o(P_new[i, j, k])
                c_f = pvt_table.get_compressibility_f(P_new[i, j, k])
                B_w = pvt_table.get_B_w(P_new[i, j, k])
                rho_w = pvt_table.get_density_w(P_new[i, j, k])
                
                gamma_w = rho_w * g
                
                # 1. Compressibility correction: S_w^n * (1 - (c_f + c_w) * dP)
                dP = P_new[i, j, k] - P_old[i, j, k]
                S_w_new[i, j, k] = S_w_old[i, j, k] * (1 - (c_f + c_w) * dP)
                
                # 2. Water flux term: -(dt/V_p) * B_w * lambda_w * T * ∇phi_w
                flux_coeff = -dt / V_pore[i, j, k]
                
                # X-direction fluxes
                if i < nx - 1:
                    dphi_w = P_new[i, j, k] - P_new[i + 1, j, k]
                    S_w_new[i, j, k] += flux_coeff * B_w * lambda_w['right'][i, j, k] * transmissibility['right'][i, j, k] * dphi_w
                
                if i > 0:
                    dphi_w = P_new[i, j, k] - P_new[i - 1, j, k]
                    S_w_new[i, j, k] += flux_coeff * B_w * lambda_w['left'][i, j, k] * transmissibility['left'][i, j, k] * dphi_w
                
                # Y-direction fluxes
                if j < ny - 1:
                    dphi_w = P_new[i, j, k] - P_new[i, j + 1, k]
                    S_w_new[i, j, k] += flux_coeff * B_w * lambda_w['front'][i, j, k] * transmissibility['front'][i, j, k] * dphi_w
                
                if j > 0:
                    dphi_w = P_new[i, j, k] - P_new[i, j - 1, k]
                    S_w_new[i, j, k] += flux_coeff * B_w * lambda_w['back'][i, j, k] * transmissibility['back'][i, j, k] * dphi_w
                
                # Z-direction fluxes (with gravity): phi_w = P - rho_w * g * z
                if k < nz - 1:
                    rho_w_up = pvt_table.get_density_w(P_new[i, j, k + 1])
                    gamma_w_up = rho_w_up * g
                    
                    phi_w_curr = P_new[i, j, k] - gamma_w * depth_center[k]
                    phi_w_up = P_new[i, j, k + 1] - gamma_w_up * depth_center[k + 1]
                    dphi_w = phi_w_curr - phi_w_up
                    
                    S_w_new[i, j, k] += flux_coeff * B_w * lambda_w['up'][i, j, k] * transmissibility['up'][i, j, k] * dphi_w
                
                if k > 0:
                    rho_w_down = pvt_table.get_density_w(P_new[i, j, k - 1])
                    gamma_w_down = rho_w_down * g
                    
                    phi_w_curr = P_new[i, j, k] - gamma_w * depth_center[k]
                    phi_w_down = P_new[i, j, k - 1] - gamma_w_down * depth_center[k - 1]
                    dphi_w = phi_w_curr - phi_w_down
                    
                    S_w_new[i, j, k] += flux_coeff * B_w * lambda_w['down'][i, j, k] * transmissibility['down'][i, j, k] * dphi_w
                
                # 3. Source term: (dt/V_p) * B_w * Q_w
                S_w_new[i, j, k] += (dt / V_pore[i, j, k]) * B_w * Q_w[i, j, k]
    
    return S_w_new


def get_production_rate(P_new, S_w_new, bhp_map, P_c_fn, rel_perm, pvt_table, J):
    """
    Calculate production rates for water and oil phases at well cells.
    q_phase = J * lambda_phase * (P_bhp - P_phase)
    Returns array [q_w, q_o, q_total] for each cell.
    """
    nx, ny, nz = bhp_map.shape
    prod_grid = np.zeros((3, nx, ny, nz))
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                P_bhp = bhp_map[i, j, k]
                
                if P_bhp == 0:  # Not a well cell
                    continue
                
                # Get fluid properties
                P_cell = P_new[i, j, k]
                S_w = S_w_new[i, j, k]
                
                B_w = pvt_table.get_B_w(P_cell)
                B_o = pvt_table.get_B_o(P_cell)
                mu_w = pvt_table.get_mu_w(P_cell)
                mu_o = pvt_table.get_mu_o(P_cell)
                
                kr_w = rel_perm.kr1(S_w)
                kr_o = rel_perm.kr2(S_w)
                
                # Calculate phase mobilities: lambda = kr / (mu * B)
                lambda_w = kr_w / (mu_w * B_w)
                lambda_o = kr_o / (mu_o * B_o)
                
                # Calculate phase potentials at well
                P_c = P_c_fn(S_w)
                Phi_w = P_bhp - P_cell
                Phi_o = P_bhp - (P_cell + P_c)
                
                # Production rates: q = J * lambda * (P_bhp - P_phase)
                q_w = J * lambda_w * Phi_w
                q_o = J * lambda_o * Phi_o
                
                prod_grid[0, i, j, k] = q_w
                prod_grid[1, i, j, k] = q_o
                prod_grid[2, i, j, k] = q_w + q_o
    
    return prod_grid