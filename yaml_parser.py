import yaml
import numpy as np

def get_raw_yaml_content(file_path):
    """
    Parses a YAML file and returns its contents as a Python dictionary.

    Args:
        file_path (str): The path to the YAML file.
    Returns:
        dict: The contents of the YAML file as a dictionary.
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
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
        "dz": dz
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
    
    
if __name__ == "__main__":
    file_path = "config.yml"
    content = get_raw_yaml_content(file_path)
    grid_config = create_grid(content)
    porosity = parse_porosity(content, grid_config)
    print(porosity)
    