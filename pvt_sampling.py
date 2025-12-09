import pandas as pd
import numpy as np

# --- 1. INPUT PARAMETERS ---
# Reservoir Conditions
T_res = 353.15  # K (80°C - reservoir temperature, typical for moderate depth)
P_ref = 1.01e5  # Pa (atmospheric pressure reference)
P_bubble = 1.5e7  # Pa (bubble point pressure for oil - below this gas comes out)

# Pressure Range
P_min = 6.9e6   # Pa (1000 psi)
P_max = 3.45e7  # Pa (5000 psi)
steps = 15  # Increased for smoother curve

# Standard Conditions (15°C, 1 atm)
rho_w_std = 1000.0  # kg/m3
rho_o_std = 850.0   # kg/m3 (API ~35° light oil)

# Reference Properties at P_bubble
mu_w_ref = 0.50e-3    # Pa·s (water at 80°C is less viscous)
mu_o_ref = 1.8e-3     # Pa·s (light oil at reservoir temp)
B_w_ref = 1.041       # Dimensionless (reservoir vol / stock tank vol)
B_o_ref = 1.28        # Dimensionless (at bubble point)

# Compressibilities (1/Pa)
c_w = 4.35e-10        # Pa^-1 (3.0e-6 psi^-1)
c_o = 2.18e-9         # Pa^-1 (15.0e-6 psi^-1, above bubble point)
c_f = 5.8e-10         # Pa^-1 (4.0e-6 psi^-1)

# Viscosity-Pressure Coefficients
alpha_w = 2.0e-11     # Pa^-1 (water viscosity pressure dependence)
alpha_o = 1.5e-10     # Pa^-1 (oil viscosity pressure dependence)

# --- 2. GENERATE TABLE ---
pressures = np.linspace(P_min, P_max, steps)
data = []

for P in pressures:
    # --- FORMATION VOLUME FACTORS (FVF) ---
    # Oil: Different behavior above/below bubble point
    if P >= P_bubble:
        # Above bubble point: undersaturated oil, isothermal compressibility
        B_o = B_o_ref * np.exp(-c_o * (P - P_bubble))
    else:
        # Below bubble point: gas liberation, B_o increases as P decreases
        # Using Standing correlation approximation
        B_o = B_o_ref * (1 + 0.15 * (P_bubble - P) / P_bubble)
    
    # Water: Simple isothermal compression
    B_w = B_w_ref * np.exp(-c_w * (P - P_bubble))
    
    # --- DENSITIES ---
    # rho_reservoir = rho_stock_tank / B
    rho_w = rho_w_std / B_w
    rho_o = rho_o_std / B_o
    
    # --- VISCOSITIES ---
    # Barus equation: mu = mu_ref * exp(alpha * delta_P)
    # More physically accurate than linear for liquids
    mu_w = mu_w_ref * np.exp(alpha_w * (P - P_bubble))
    mu_o = mu_o_ref * np.exp(alpha_o * (P - P_bubble))
    
    # --- COMPRESSIBILITIES ---
    # Oil compressibility changes at bubble point
    if P >= P_bubble:
        c_o_current = c_o  # Undersaturated (liquid only)
    else:
        # Saturated oil: much higher compressibility due to gas liberation
        c_o_current = c_o * 5.0  # Typical increase of 3-10x
    
    # Append to list
    data.append([
        f"{P:.2e}",  # Pa in scientific notation
        round(B_o, 5), 
        round(B_w, 5), 
        f"{c_w:.2e}", 
        f"{c_o_current:.2e}", 
        f"{c_f:.2e}", 
        f"{mu_w:.4e}",
        f"{mu_o:.4e}",
        round(rho_w, 2), 
        round(rho_o, 2)
    ])

# Create DataFrame
df = pd.DataFrame(data, columns=[
    'P (Pa)', 'B_o (-)', 'B_w (-)', 
    'c_w (Pa^-1)', 'c_o (Pa^-1)', 'c_f (Pa^-1)', 
    'mu_w (Pa·s)', 'mu_o (Pa·s)', 
    'rho_w (kg/m3)', 'rho_o (kg/m3)'
])

print(df.to_csv(index=False))
print("\n--- Key Points ---")
print(f"Bubble Point Pressure: {P_bubble:.2e} Pa ({P_bubble/1e6:.1f} MPa)")
print(f"Reservoir Temperature: {T_res:.2f} K ({T_res-273.15:.0f}°C)")
print(f"Pressure Range: {P_min/1e6:.1f} - {P_max/1e6:.1f} MPa")