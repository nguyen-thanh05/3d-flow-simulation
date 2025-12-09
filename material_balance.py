import numpy as np

def incremental_material_balance(S1_old, S1_new, P_old, P_new, V_pore, pvt_table, prod_grid, inj_grid, dt):
    c_f = pvt_table.get_compressibility_f(P_new)
    c_w = pvt_table.get_compressibility_w(P_new)
    c_o = pvt_table.get_compressibility_o(P_new)
    B_w = pvt_table.get_B_w(P_new)
    B_o = pvt_table.get_B_o(P_new)
    lhs = V_pore * (c_f + S1_new * c_w + (1 - S1_new) * c_o) * (P_new - P_old)
    lhs = lhs.sum()
    
    rhs = (B_w * inj_grid) * dt
    rhs += (B_w * prod_grid[0] + B_o * prod_grid[1]) * dt
    rhs = rhs.sum()
    
    ratio = lhs/rhs if rhs != 0 else np.inf
    return lhs, rhs, ratio

def incremental_saturation_material_balance(S1_old, S1_new, P_new, V_pore, pvt_table, prod_grid, inj_grid, dt):
    B_w = pvt_table.get_B_w(P_new)
    lhs = V_pore * (S1_new - S1_old)
    lhs = lhs.sum()
    rhs = (B_w * inj_grid + B_w * prod_grid[0]) * dt
    rhs = rhs.sum()
    ratio = lhs/rhs if rhs != 0 else np.inf
    return lhs, rhs, ratio