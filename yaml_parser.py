import re
import yaml
import numpy as np
import os
import pandas as pd
from RelativePermeability import RelPermCoreyFitter
from PVT_Table import PVTTable


def get_raw_yaml_content(file_path):
    """
    Parses a YAML file and returns its contents as a Python dictionary.

    Args:
        file_path (str): The path to the YAML file.
    Returns:
        dict: The contents of the YAML file as a dictionary.
    """
    loader = yaml.SafeLoader
    # Add regex rules so yml can load 1e-12 as float
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    
    with open(file_path, 'r') as file:
        data = yaml.load(file, Loader=loader)
    return data


def create_grid(yml_dict):
    """
    Create grid based on key 'grid' in the dictionary. Also check for errors.    
    """
    if "static_variables" not in yml_dict:
        raise KeyError("Missing 'static_variables' key in YAML dictionary.")
    if "grid" not in yml_dict["static_variables"]:
        raise KeyError("Missing 'grid' key in 'static_variables' section.")
    
    grid_config = yml_dict["static_variables"]["grid"]
    dimensions = grid_config.get("dimensions")
    if not dimensions or not isinstance(dimensions, list) or len(dimensions) != 3:
        raise ValueError("'dimensions' must be a list of length 3, [nx, ny, nz].")
    nx, ny, nz = dimensions
    spacing_x = grid_config.get("spacing_x")
    spacing_y = grid_config.get("spacing_y")
    spacing_z = grid_config.get("spacing_z")
    if spacing_x is None or spacing_y is None or spacing_z is None:
        raise ValueError("Grid spacing values 'spacing_x', 'spacing_y', and 'spacing_z' must be provided.")
    
    
    spacings = {'dx': (spacing_x, nx), 'dy': (spacing_y, ny), 'dz': (spacing_z, nz)}
    results = {}
    for axis, (spacing, n) in spacings.items():
        if type(spacing) in [int, float]:
            results[axis] = np.full(n, spacing)
        elif type(spacing) == list:
            if len(spacing) != n:
                raise ValueError(f"Length of 'spacing_{axis}' list must be {n}.")
            results[axis] = np.array(spacing)
        elif type(spacing) == str:
            arr = np.load(spacing)
            if len(arr.shape) != 1:
                raise ValueError(f"'spacing_{axis}' file data must be one-dimensional.")
            if len(arr) != n:
                raise ValueError(f"Length of 'spacing_{axis}' file data must be {n}.")
            results[axis] = arr

    dx, dy, dz = results['dx'], results['dy'], results['dz']
    
    return {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "depth_top": grid_config.get("depth_top", 0.0)
    }
    

def parse_porosity(yml_dict, grid_config):
    """
    Parse porosity information from the YAML dictionary.
    """
    if "static_variables" not in yml_dict:
        raise KeyError("Missing 'static_variables' key in YAML dictionary.")
    if "physical_properties" not in yml_dict["static_variables"]:
        raise NotImplementedError("Missing key 'physical_properties' in 'static_variables' section.")
    if "porosity" not in yml_dict["static_variables"]["physical_properties"]:
        raise KeyError("Missing 'porosity' key in 'physical_properties' section.")
    porosity = yml_dict["static_variables"]["physical_properties"]["porosity"].get("phi")
    if porosity is None:
        raise ValueError("Missing 'phi' key in 'porosity' section.")
    nx, ny, nz = grid_config["nx"], grid_config["ny"], grid_config["nz"]
    if type(porosity) in [int, float]:
        return np.full((nx, ny, nz), porosity)
    elif type(porosity) == str:
        porosity_grid = np.load(porosity)
        if porosity_grid.shape != (nx, ny, nz):
            raise ValueError(f"Porosity grid shape must be ({nx}, {ny}, {nz}).")
        return porosity_grid
    else:
        raise ValueError("'phi' must be a scalar or a file path string.")


def parse_permeability(yml_dict, grid_config):
    """
    Parse permeability information from the YAML dictionary.
    """
    if "static_variables" not in yml_dict:
        raise KeyError("Missing 'static_variables' key in YAML dictionary.")
    if "physical_properties" not in yml_dict["static_variables"]:
        raise NotImplementedError("Missing key 'physical_properties' in 'static_variables' section.")
    if "permeability" not in yml_dict["static_variables"]["physical_properties"]:
        raise KeyError("Missing 'permeability' key in 'physical_properties' section.")
    
    kx = yml_dict["static_variables"]["physical_properties"]["permeability"].get("k_x")
    ky = yml_dict["static_variables"]["physical_properties"]["permeability"].get("k_y")
    kz = yml_dict["static_variables"]["physical_properties"]["permeability"].get("k_z")
    
    nx, ny, nz = grid_config["nx"], grid_config["ny"], grid_config["nz"]
    perm_dictionary = {}
    for k, name in zip([kx, ky, kz], ['k_x', 'k_y', 'k_z']):
        if k is None:
            raise ValueError(f"Missing '{name}' key in 'permeability' section.")
        if type(k) in [int, float]:
            k_array = np.full((nx, ny, nz), k)
        elif type(k) == str:
            k_array = np.load(k)
            if k_array.shape != (nx, ny, nz):
                raise ValueError(f"Permeability grid shape for '{name}' must be ({nx}, {ny}, {nz}).")
        else:
            raise ValueError(f"'{name}' must be a scalar or a file path string.")
        perm_dictionary[name] = k_array
    return perm_dictionary    
    
def parse_rock_fluid_model(yml_dict):
    """
    Parse rock and fluid model information from the YAML dictionary.
    Currently supports only 'black_oil' model.
    """
    if "static_variables" not in yml_dict:
        raise KeyError("Missing 'static_variables' key in YAML dictionary.")
    if "physical_properties" not in yml_dict["static_variables"]:
        raise NotImplementedError("Missing key 'physical_properties' in 'static_variables' section.")
    if "rock_fluid_model" not in yml_dict["static_variables"]["physical_properties"]:
        raise KeyError("Missing 'rock_fluid_model' key in 'physical_properties' section.")

    
    if "file" not in yml_dict["static_variables"]["physical_properties"]["rock_fluid_model"]:
        raise KeyError("Missing 'file' key in 'rock_fluid_model' section.")
    
    csv_path = yml_dict["static_variables"]["physical_properties"]["rock_fluid_model"]["file"]
    
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path, skipinitialspace=True)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")
    
    # Check for required columns
    required_columns = ['S1', 'kr1', 'kr2']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise KeyError(f"Missing required columns in CSV: {missing_columns}. "
                      f"Available columns: {list(df.columns)}")
    
    # Extract data as numpy arrays
    S1 = df['S1'].values
    kr1 = df['kr1'].values
    kr2 = df['kr2'].values
    
    # Validate data
    if len(S1) == 0:
        raise ValueError("CSV file contains no data rows.")
    
    if len(S1) != len(kr1) or len(S1) != len(kr2):
        raise ValueError(f"Column lengths don't match: S1={len(S1)}, "
                        f"kr1={len(kr1)}, kr2={len(kr2)}")
    
    
    # Validate physical constraints
    if np.any(S1 < 0) or np.any(S1 > 1):
        raise ValueError(f"S1 values must be in [0, 1]. Found range: [{S1.min()}, {S1.max()}]")
    
    if np.any(kr1 < 0) or np.any(kr1 > 1):
        raise ValueError(f"kr1 values must be in [0, 1]. Found range: [{kr1.min()}, {kr1.max()}]")
    
    if np.any(kr2 < 0) or np.any(kr2 > 1):
        raise ValueError(f"kr2 values must be in [0, 1]. Found range: [{kr2.min()}, {kr2.max()}]")
    
    # Check if S1 is monotonically increasing (typical for lab data)
    if not np.all(np.diff(S1) >= 0):
        print("Warning: S1 values are not monotonically increasing. "
              "This may indicate data entry errors.")
    
    # Find S1r, S2r, Kr1_max, Kr2_max
    S1r = S1[kr1 == 0]  
    S2r = 1 - S1[kr2 == 0]
    kr1_max = np.max(kr1)
    kr2_max = np.max(kr2)   
    
    rel_perm, n1, n2 = RelPermCoreyFitter.fit_corey_exponents(S1, kr1, kr2, S1r, S2r, kr1_max, kr2_max)
    print("Fitted Corey exponents: n1 =", n1, ", n2 =", n2)
    return rel_perm


def parse_pvt_table(yml_dict):
    """
    Parse PVT table information from the YAML dictionary.
    """
    if "static_variables" not in yml_dict:
        raise KeyError("Missing 'static_variables' key in YAML dictionary.")
    if "physical_properties" not in yml_dict["static_variables"]:
        raise NotImplementedError("Missing key 'physical_properties' in 'static_variables' section.")
    if "pvt_table" not in yml_dict["static_variables"]["physical_properties"]:
        raise KeyError("Missing 'pvt_table' key in 'physical_properties' section.")
    
    if "file" not in yml_dict["static_variables"]["physical_properties"]["pvt_table"]:
        raise KeyError("Missing 'file' key in 'pvt_table' section.")
    
    csv_path = yml_dict["static_variables"]["physical_properties"]["pvt_table"]["file"]
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"PVT CSV file not found: {csv_path}")
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path, skipinitialspace=True)
    except Exception as e:
        raise ValueError(f"Failed to read PVT CSV file: {e}")
    
    P = df['P'].values
    B_o = df['B_o'].values
    B_w = df['B_w'].values
    c_w = df['c_w'].values
    c_o = df['c_o'].values
    c_f = df['c_f'].values
    mu_w = df['mu_w'].values
    mu_o = df['mu_o'].values
    rho_w = df['rho_w'].values
    rho_o = df['rho_o'].values
    
    pvt_table = PVTTable(P, B_o, B_w, c_w, c_o, c_f, mu_w, mu_o, rho_w, rho_o)
    return pvt_table


def get_init_pressure(yml_dict, grid_config):
    """
    Get initial pressure from YAML dictionary.
    """
    if "initial_conditions" not in yml_dict:
        raise KeyError("Missing 'initial_conditions' key in YAML dictionary.")
    if "pressure" not in yml_dict["initial_conditions"]:
        raise KeyError("Missing 'pressure' key in 'initial_conditions' section.")
    P_init = yml_dict["initial_conditions"]["pressure"]
    nx, ny, nz = grid_config["nx"], grid_config["ny"], grid_config["nz"]
    return np.full((nx, ny, nz), P_init)


def get_init_saturation(yml_dict, grid_config):
    """
    Get initial saturation from YAML dictionary.
    """
    if "initial_conditions" not in yml_dict:
        raise KeyError("Missing 'initial_conditions' key in YAML dictionary.")
    if "S_w" not in yml_dict["initial_conditions"]:
        raise KeyError("Missing 'S_w' key in 'initial_conditions' section.")
    S1_init = yml_dict["initial_conditions"]["S_w"]
    nx, ny, nz = grid_config["nx"], grid_config["ny"], grid_config["nz"]
    return np.full((nx, ny, nz), S1_init)

def parse_well_controls(yml_dict):
    """
    Parse well control information from the YAML dictionary.
    """
    if "well_controls" not in yml_dict:
        raise KeyError("Missing 'well_controls' key in YAML dictionary.")
    
    if "injectors" not in yml_dict["well_controls"]:
        raise KeyError("Missing 'injectors' key in 'well_controls' section.")
    if "producers" not in yml_dict["well_controls"]:
        raise KeyError("Missing 'producers' key in 'well_controls' section.")
    
    injector_grid = np.load(yml_dict["well_controls"]["injectors"]["rate_grid"])
    producer_grid = np.load(yml_dict["well_controls"]["producers"]["bhp_grid"])
    return injector_grid, producer_grid

if __name__ == "__main__":
    file_path = "sample_config.yml"
    content = get_raw_yaml_content(file_path)
    grid_config = create_grid(content)
    porosity = parse_porosity(content, grid_config)
    rel_perm = parse_rock_fluid_model(content)
    pvt_table = parse_pvt_table(content)
    
    # print(parse_permeability(content, grid_config))
