import numpy as np
from scipy.interpolate import interp1d

class PVTTable:
    """
    PVT (Pressure-Volume-Temperature) table handler
    Stores and interpolates fluid properties as function of pressure
    """
    def __init__(self, P, B_o, B_w, c_w, c_o, c_f, mu_w, mu_o, rho_w, rho_o, interp_method='linear'):
        """
        Parameters:
        -----------
        P : array, pressure values (kPa)
        B_o : array, oil formation volume factor (unitless)
        B_w : array, water formation volume factor (unitless)
        c_w : array, water compressibility (1/Pa)
        c_o : array, oil compressibility (1/Pa)
        mu_w : array, water viscosity (cp)
        mu_o : array, oil viscosity (cp)
        rho_w : array, water density (kg/m3)
        rho_o : array, oil density (kg/m3)
        interp_method : str, 'linear' or 'cubic'
        """
        sort_idx = np.argsort(P)
        self.P = P[sort_idx]
        self.B_o = B_o[sort_idx]
        self.B_w = B_w[sort_idx]
        self.c_f = c_f[sort_idx]
        self.c_w = c_w[sort_idx]
        self.c_o = c_o[sort_idx]
        self.mu_w = mu_w[sort_idx]
        self.mu_o = mu_o[sort_idx]
        self.rho_w = rho_w[sort_idx]
        self.rho_o = rho_o[sort_idx]
        self.P_min = self.P[0]
        self.P_max = self.P[-1]
        self.interp_method = interp_method
        
        self._create_interpolators()
    
    def _create_interpolators(self):
        """Create scipy interpolation objects for each property"""
        self.B_o_interp = interp1d(self.P, self.B_o, kind=self.interp_method, 
                                    fill_value='extrapolate')
        self.B_w_interp = interp1d(self.P, self.B_w, kind=self.interp_method, 
                                    fill_value='extrapolate')
        self.c_w_interp = interp1d(self.P, self.c_w, kind=self.interp_method, 
                                    fill_value='extrapolate')
        self.c_o_interp = interp1d(self.P, self.c_o, kind=self.interp_method, 
                                    fill_value='extrapolate')
        self.mu_w_interp = interp1d(self.P, self.mu_w, kind=self.interp_method, 
                                     fill_value='extrapolate')
        self.mu_o_interp = interp1d(self.P, self.mu_o, kind=self.interp_method, 
                                     fill_value='extrapolate')
        self.rho_w_interp = interp1d(self.P, self.rho_w, kind=self.interp_method, 
                                      fill_value='extrapolate')
        self.rho_o_interp = interp1d(self.P, self.rho_o, kind=self.interp_method, 
                                      fill_value='extrapolate')
        self.c_f_interp = interp1d(self.P, self.c_f, kind=self.interp_method,
                                   fill_value='extrapolate')
    
    def get_properties(self, P):
        """
        Get all fluid properties at given pressure(s)
        
        Parameters:
        -----------
        P : float or array, pressure value(s)
        
        Returns:
        --------
        dict with keys: B_o, B_w, c_w, c_o, mu_w, mu_o, rho_w, rho_o
        """
        # Clip pressure to table range (with warning)
        P_clipped = np.clip(P, self.P_min, self.P_max)
        
        if np.any(P < self.P_min) or np.any(P > self.P_max):
            print(f"Warning: Pressure outside table range [{self.P_min}, {self.P_max}]. Extrapolating.")
        
        return {
            'B_o': self.B_o_interp(P_clipped),
            'B_w': self.B_w_interp(P_clipped),
            'c_w': self.c_w_interp(P_clipped),
            'c_o': self.c_o_interp(P_clipped),
            'c_f': self.c_f_interp(P_clipped),
            'mu_w': self.mu_w_interp(P_clipped),
            'mu_o': self.mu_o_interp(P_clipped),
            'rho_w': self.rho_w_interp(P_clipped),
            'rho_o': self.rho_o_interp(P_clipped)
        }
    
    def get_B_o(self, P):
        """Get oil formation volume factor at pressure P"""
        return self.B_o_interp(np.clip(P, self.P_min, self.P_max))
    
    def get_B_w(self, P):
        """Get water formation volume factor at pressure P"""
        return self.B_w_interp(np.clip(P, self.P_min, self.P_max))
    
    def get_mu_o(self, P):
        """Get oil viscosity at pressure P"""
        return self.mu_o_interp(np.clip(P, self.P_min, self.P_max))
    
    def get_mu_w(self, P):
        """Get water viscosity at pressure P"""
        return self.mu_w_interp(np.clip(P, self.P_min, self.P_max))
    
    def get_compressibility_o(self, P):
        """Get oil compressibility at pressure P"""
        return self.c_o_interp(np.clip(P, self.P_min, self.P_max))
    
    def get_compressibility_w(self, P):
        """Get water compressibility at pressure P"""
        return self.c_w_interp(np.clip(P, self.P_min, self.P_max))
    
    def get_compressibility_f(self, P):
        """Get fracture compressibility at pressure P"""
        return self.c_f_interp(np.clip(P, self.P_min, self.P_max))
    
    def get_density_w(self, P):
        """Get water density at pressure P"""
        return self.rho_w_interp(np.clip(P, self.P_min, self.P_max))
    
    def get_density_o(self, P):
        """Get oil density at pressure P"""
        return self.rho_o_interp(np.clip(P, self.P_min, self.P_max))
    