###########################################
###########################################
### quantum system initialization ###
'''
In this part, a quantum system can be initialized,
the setting includes:
1). boundary conditions
2). the bond among sites(only for nearest-neighbor
interactions now)
3). different quantum systems(only ising model now)
'''

import numpy as np

# define a class of quantum system that includes
# bond among sites; 
# type of boundary conditions;
# parameters of the quantum system
class QuantSys():
    def __init__(self, Lx, Ly, boundary_condition, sys_name, *sys_params):
        self.n_sites = Lx * Ly # number of system sites
        self.boundary_condition = boundary_condition
        if boundary_condition == "open":
            self.bond_site = site_bond_initialization_open(Lx,Ly)
        elif boundary_condition == "closed":
            self.bond_site = site_bond_initialization_closed(Lx,Ly)
        else:
            raise RuntimeError(
                f"No boundary Condition: {boundary_condition}. The options of boundary condition are: closed and open")
        # define quantum system type:
        if sys_name == 'ising':
            self.J = sys_params[0]
            self.hx = sys_params[1]
            self.hz = sys_params[2]
        else:
            raise RuntimeError(
                f"No quantum system: {sys_name}. The options of quantum system are: ising")
        print("The system is" ,sys_name," model with",boundary_condition,"boundary conditions.")

# we only consider nearest-neighbor interactions        
# 1. generate the site bond with open boundary condition:
def site_bond_initialization_open(Lx, Ly):
    if Lx == 0 or Ly == 0:
        raise RuntimeError(f"Wrong system lattice setting: number of site along x direction: {Lx}, and the number of site along y direction: {Ly}")
    elif Lx == 1:
        raise RuntimeError(f"for 1D system, Lx must be the system size, instead of Ly.")
    
    # generate bond in x direction:
    site_bond = np.zeros(((Lx-1)*Ly + (Ly-1)*Lx,2), dtype=int)
    count_site = 0 # count number of site
    for yi in range(Ly):
        for xi in range(Lx - 1):
            site_i = yi * Lx + xi
            site_bond[count_site,0] = site_i
            site_bond[count_site,1] = site_i + 1
            count_site += 1

    # generate bond in y direction:
    # only for 2 dimensional systems:
    if Ly >= 2:
        for yi in range(Ly - 1):
            for xi in range(Lx):
                site_i = yi * Lx + xi
                site_bond[count_site,0] = site_i
                site_bond[count_site,1] = site_i + Lx
                count_site += 1
    return site_bond

# 2. generate the site bond with closed boundary condition:
def site_bond_initialization_closed(Lx, Ly):
    if Lx == 0 or Ly == 0:
        raise RuntimeError(f"Wrong system lattice setting: number of site along x direction: {Lx}, and the number of site along y direction: {Ly}")
    elif Lx == 1:
        raise RuntimeError(f"for 1D system, Lx must be the system size, instead of Ly.")

    if Ly == 1:
        site_bond = np.zeros((Lx,2), dtype=int) # 1D system
    else:
        site_bond = np.zeros((2*Lx*Ly,2), dtype=int) # 2D system

    # generate bond in x direction:
    count_site = 0 # count number of site
    for yi in range(Ly):
        for xi in range(Lx):
            site_i = yi * Lx + xi
            site_bond[count_site,0] = site_i
            if xi < Lx-1:
                site_bond[count_site,1] = site_i + 1
            elif xi == Lx-1:
                site_bond[count_site,1] = yi * Lx
            count_site += 1

    # generate bond in y direction:
    # only for 2 dimensional systems:
    if Ly >= 2:
        for yi in range(Ly):
            for xi in range(Lx):
                site_i = yi * Lx + xi
                site_bond[count_site,0] = site_i
                if yi < Ly-1:
                    site_bond[count_site,1] = site_i + Lx
                elif yi == Ly-1:
                    site_bond[count_site,1] = xi
                count_site += 1
    return site_bond