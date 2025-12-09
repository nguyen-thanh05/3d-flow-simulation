import numpy as np


class RelPermCorey:
    """
    Corey-type relative permeability correlation
    Kr1 = Kr1_max * ((S1 - S1r) / (1 - S1r - S2r))^n1
    Kr2 = Kr2_max * ((S2 - S2r) / (1 - S1r - S2r))^n2
    
    Phase 1 is typically water (w)
    Phase 2 is typically oil (o)
    """
    def __init__(self, S1r, S2r, n1, n2, Kr1_max=1.0, Kr2_max=1.0):
        """
        Parameters:
        -----------
        S1r : float, residual saturation of phase 1 (e.g., Swr - connate water)
        S2r : float, residual saturation of phase 2 (e.g., Sor - residual oil)
        n1 : float, Corey exponent for phase 1 (typically 2-4 for water)
        n2 : float, Corey exponent for phase 2 (typically 2-4 for oil)
        Kr1_max : float, endpoint relative permeability for phase 1 (default 1.0)
        Kr2_max : float, endpoint relative permeability for phase 2 (default 1.0)
        """
        self.S1r = S1r
        self.S2r = S2r
        self.n1 = n1
        self.n2 = n2
        self.Kr1_max = Kr1_max
        self.Kr2_max = Kr2_max
        
        # Validate inputs
        if S1r + S2r >= 1.0:
            raise ValueError(f"S1r + S2r must be < 1.0, got {S1r + S2r}")
    
    def normalized_saturation(self, S1):
        """
        Calculate normalized saturation
        S1_norm = (S1 - S1r) / (1 - S1r - S2r)
        """
        S1_norm = (S1 - self.S1r) / (1 - self.S1r - self.S2r)
        return np.clip(S1_norm, 0, 1)
    
    def kr1(self, S1):
        """Relative permeability of phase 1 (e.g., water)"""
        S1_norm = self.normalized_saturation(S1)
        return self.Kr1_max * S1_norm**self.n1
    
    def kr2(self, S1):
        """
        Relative permeability of phase 2 (e.g., oil)
        Note: S2 = 1 - S1 (two-phase system)
        """
        S1_norm = self.normalized_saturation(S1)
        S2_norm = 1 - S1_norm
        return self.Kr2_max * S2_norm**self.n2
    
    def kr_both(self, S1):
        """Return both relative permeabilities"""
        return self.kr1(S1), self.kr2(S1)
    
    def dkr1_dS1(self, S1):
        """Derivative of kr1 with respect to S1 (needed for Jacobian in implicit methods)"""
        S1_norm = self.normalized_saturation(S1)
        if self.n1 == 0:
            return 0
        return self.Kr1_max * self.n1 * S1_norm**(self.n1 - 1) / (1 - self.S1r - self.S2r)
    
    def dkr2_dS1(self, S1):
        """Derivative of kr2 with respect to S1"""
        S1_norm = self.normalized_saturation(S1)
        S2_norm = 1 - S1_norm
        if self.n2 == 0:
            return 0
        return -self.Kr2_max * self.n2 * S2_norm**(self.n2 - 1) / (1 - self.S1r - self.S2r)


class RelPermCoreyFitter:
    """
    Fit Corey exponents (n1, n2) from laboratory data
    Endpoints (S1r, S2r, Kr1_max, Kr2_max) are provided from lab measurements
    """
    @staticmethod
    def fit_corey_exponents(S1_data, kr1_data, kr2_data, S1r, S2r, Kr1_max, Kr2_max):
        """
        Fit ONLY the Corey exponents from lab data
        Endpoints are already known from lab measurements
        
        Parameters:
        -----------
        S1_data : array, saturation data points from lab
        kr1_data : array, measured kr1 values from lab
        kr2_data : array, measured kr2 values from lab
        S1r : float, residual saturation of phase 1 (FROM LAB)
        S2r : float, residual saturation of phase 2 (FROM LAB)
        Kr1_max : float, maximum kr1 (FROM LAB)
        Kr2_max : float, maximum kr2 (FROM LAB)
        
        Returns:
        --------
        RelPermCorey object with fitted exponents
        n1, n2 : fitted Corey exponents
        """
        # Normalized saturation for phase 1
        S1_norm = (S1_data - S1r) / (1 - S1r - S2r)
        S1_norm = np.clip(S1_norm, 0, 1)
        
        # Fit phase 1 exponent (n1)
        # kr1 = Kr1_max * S1_norm^n1
        # Only fit where S1_norm > 0 to avoid log(0)
        mask1 = (S1_norm > 1e-6) & (kr1_data > 1e-6)
        if np.sum(mask1) > 1:
            # Take log: log(kr1/Kr1_max) = n1 * log(S1_norm)
            log_kr1_norm = np.log(kr1_data[mask1] / Kr1_max)
            log_S1_norm = np.log(S1_norm[mask1])
            n1_fit = np.polyfit(log_S1_norm, log_kr1_norm, 1)[0]
            n1_fit = max(0.5, min(n1_fit, 10.0))  # Bound between 0.5 and 10
        else:
            print("Warning: Insufficient data for phase 1 fitting, using n1 = 2.0")
            n1_fit = 2.0
        
        # Fit phase 2 exponent (n2)
        # kr2 = Kr2_max * S2_norm^n2, where S2_norm = 1 - S1_norm
        S2_norm = 1 - S1_norm
        mask2 = (S2_norm > 1e-6) & (kr2_data > 1e-6)
        if np.sum(mask2) > 1:
            log_kr2_norm = np.log(kr2_data[mask2] / Kr2_max)
            log_S2_norm = np.log(S2_norm[mask2])
            n2_fit = np.polyfit(log_S2_norm, log_kr2_norm, 1)[0]
            n2_fit = max(0.5, min(n2_fit, 10.0))  # Bound between 0.5 and 10
        else:
            print("Warning: Insufficient data for phase 2 fitting, using n2 = 2.0")
            n2_fit = 2.0
        
        # Create RelPermCorey object with fitted exponents
        relperm = RelPermCorey(S1r, S2r, n1_fit, n2_fit, Kr1_max, Kr2_max)
        
        return relperm, n1_fit, n2_fit
    