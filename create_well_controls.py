import numpy as np
import matplotlib.pyplot as plt

# --- GRID DIMENSIONS ---
nx = 21  # Grid cells in x-direction (odd number for center producer)
ny = 21  # Grid cells in y-direction (odd number for center producer)
nz = 5   # Number of layers

# --- WELL LOCATIONS ---
# Center producer
prod_i = nx // 2  # Center x-index (10 for 21x21 grid)
prod_j = ny // 2  # Center y-index (10 for 21x21 grid)
prod_k = 2        # Second layer

# Four injectors in 5-spot pattern (offset from center)
# Pattern radius (distance from center to injector)
pattern_radius = 6  # Grid cells from center (adjustable)

injectors = [
    (prod_i - pattern_radius, prod_j - pattern_radius, 2),  # Bottom-left
    (prod_i + pattern_radius, prod_j - pattern_radius, 2),  # Bottom-right
    (prod_i - pattern_radius, prod_j + pattern_radius, 2),  # Top-left
    (prod_i + pattern_radius, prod_j + pattern_radius, 2)   # Top-right
]

# --- INJECTION RATE GRID ---
# Total injection rate distributed among 4 injectors
total_injection_rate = 200  # m3/day (field rate)
rate_per_injector = total_injection_rate / 4  # m3/day per injector
rate_per_injector_SI = rate_per_injector / 24 / 3600  # Convert to m3/s

inj_grid = np.zeros((nx, ny, nz))
for (i, j, k) in injectors:
    inj_grid[i, j, k] = rate_per_injector_SI  # m3/s

np.save("input_config/rate_grid.npy", inj_grid)
print(f"Injection rate grid saved: rate_grid.npy")
print(f"Total injection rate: {total_injection_rate} m3/day")
print(f"Rate per injector: {rate_per_injector:.2f} m3/day = {rate_per_injector_SI:.6e} m3/s")

# --- PRODUCER BHP GRID ---
# Producer operates at constant bottom-hole pressure
producer_bhp = 3.6e6  # Pa, 3.6 MPa

bhp_grid = np.zeros((nx, ny, nz))
bhp_grid[prod_i, prod_j, prod_k] = producer_bhp  # Pa

np.save("input_config/bhp_grid.npy", bhp_grid)
print(f"\nBHP grid saved: bhp_grid.npy")
print(f"Producer BHP: {producer_bhp/1e6:.1f} MPa at location ({prod_i}, {prod_j}, {prod_k})")