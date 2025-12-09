from solver_functions import (calculate_transmissibility,
                              calculate_pore_volume,
                              add_well_model,
                              assemble_lhs_matrix,
                              assemble_rhs_vector,
                              get_new_saturation,
                              get_production_rate)
from yaml_parser import (get_raw_yaml_content,
                         create_grid,
                         parse_porosity,
                         parse_permeability,
                         parse_rock_fluid_model,
                         parse_pvt_table,
                         get_init_pressure,
                         get_init_saturation,
                         parse_well_controls)
from material_balance import incremental_material_balance, incremental_saturation_material_balance
import numpy as np

def capillary_pressure_fn(S1, P0=50000, S1r=0.2):
    return P0 * np.exp(-4 * (S1 - S1r))

def main():
    file_path = "validate/sample_config.yml" # If validation mode
    # file_path = "input_config/config.yml"  # For normal run
    
    content = get_raw_yaml_content(file_path)
    grid_config = create_grid(content)
    porosity = parse_porosity(content, grid_config)
    rel_perm = parse_rock_fluid_model(content)
    pvt_table = parse_pvt_table(content)
    permeability = parse_permeability(content, grid_config)
    
    nx, ny, nz = grid_config["nx"], grid_config["ny"], grid_config["nz"]
    dx, dy, dz = grid_config["dx"], grid_config["dy"], grid_config["dz"]
    depth_top = grid_config["depth_top"]
    
    transmissibility = calculate_transmissibility(permeability, grid_config)
    
    # Initial conditions
    P_old = get_init_pressure(content, grid_config)
    S1_old = get_init_saturation(content, grid_config)
    
    V_pore = calculate_pore_volume(nx, ny, nz, dx, dy, dz, porosity)
    
    inj_grid, bhp_map = parse_well_controls(content)
    
    # Time stepping parameters
    dt = 43200  # 0.5 days in seconds
    num_steps = 15  # Number of time steps to simulate
    total_mb_ratio = []
    water_mb_ratio = []
    # Time loop
    for step in range(num_steps):
        
        # Calculate capillary pressure
        P_c_matrix = capillary_pressure_fn(S1_old)
        
        # Assemble system
        lhs = assemble_lhs_matrix(rel_perm,
                                  pvt_table,
                                  S1_old,
                                  P_old,
                                  capillary_pressure_fn,
                                  transmissibility,
                                  nx, ny, nz, V_pore,
                                  dx, dy, dz, dt, depth_top)
        
        rhs = assemble_rhs_vector(P_old, S1_old, rel_perm, pvt_table, 
                                  np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz)), 
                                  V_pore, capillary_pressure_fn,
                                  transmissibility,
                                  dx, dy, dz, depth_top, nx, ny, nz, dt)
        
        # Add well model
        lhs, rhs = add_well_model(
            inj_grid, bhp_map, 7e-10, lhs, rhs, pvt_table, P_old, P_c_matrix, 
            rel_perm, S1_old, nx, ny, nz, dt)
        
        # Solve for new pressure
        A_inv = np.linalg.inv(lhs)
        P_new = A_inv @ rhs
        P_new = P_new.reshape((nx, ny, nz))
        
        # Update saturation
        
        
        # Calculate production rate
        prod_grid = get_production_rate(P_new, S1_old, bhp_map, 
                                        capillary_pressure_fn, rel_perm, 
                                        pvt_table, 7e-10)
        
        S1_new = get_new_saturation(S1_old, P_new, P_old, inj_grid + prod_grid[0], 
                                     transmissibility, capillary_pressure_fn, 
                                     rel_perm, pvt_table, V_pore, depth_top, 
                                     dx, dy, dz, dt, nx, ny, nz)
        
        # Print results for first time step
        if step == 0:
            print("\nPressure Field (Pa):")
            print(P_new.flatten())
            print("\nSaturation Field:")
            print(S1_new.flatten())
            print(f"\nProduction Rate at well (3,0,0):")
            print(prod_grid[:, 3, 0, 0] * 24 * 3600)
        
        # Update old values for next iteration
        
        
        lhs, rhs, ratio = incremental_material_balance(S1_old, S1_new, P_old, P_new, V_pore, pvt_table, prod_grid, inj_grid, dt)
        print(f"Step {step+1}: Material Balance Ratio = {ratio}, LHS = {lhs}, RHS = {rhs}")
        total_mb_ratio.append(ratio)
        lhs, rhs, ratio = incremental_saturation_material_balance(S1_old, S1_new, P_new, V_pore, pvt_table, prod_grid, inj_grid, dt)
        print(f"Step {step+1}: Material Balance Ratio = {ratio}, LHS = {lhs}, RHS = {rhs}")
        water_mb_ratio.append(ratio)
        
        P_old = P_new.copy()
        S1_old = S1_new.copy()
    
    print(f"\n{'='*60}")
    print("Simulation Complete")
    print(f"{'='*60}")
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_steps + 1), total_mb_ratio, marker='o', label='Total Material Balance Ratio')
    plt.plot(range(1, num_steps + 1), water_mb_ratio, marker='x', label='Water Material Balance Ratio')
    plt.ylim(0.99, 1.02)
    plt.xlabel('Time Step')
    plt.ylabel('Material Balance Ratio')
    plt.title('Incremental MB Ratio over Time Steps')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()