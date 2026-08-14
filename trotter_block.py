###########################################
###########################################
### Trotter block ###
'''
This is used for generating trotter block 
'''

import torch as tc
import quante.generate.operas as op
from quante.generate.basis import spin_basis

# define an n-th unitary operator via Trotter decomposition:
class Un:
    def __init__(self, U, U_dag, ind_U_col, ind_state):
        self.U = U
        self.U_dag = U_dag
        self.ind_U_col = ind_U_col
        self.ind_state = ind_state

# define an n-th unitary operator via Trotter decomposition:
class Un_full:
    def __init__(self, U, U_dag, ind_U_col, ind_state):
        self.U = U
        self.U_dag = U_dag
        self.ind_U_col = ind_U_col
        self.ind_state = ind_state

# define an n-th unitary operator with m Hamiltonian bonds:
def get_Un(ising, dt, n, n_bonds, dtype=tc.complex64, device="cpu"):
    if n + n_bonds > ising.n_sites:
        n_bonds = ising.n_sites - n
    # define the 2 by 2 identity matrix: 
    I2 = tc.tensor([[1,0],[0,1]],dtype=dtype,device=device)
    # define sigma x:
    sigma_x = tc.tensor([[0,1],[1,0]],dtype=dtype,device=device)
    # define sigma z:
    sigma_z = tc.tensor([[1,0],[0,-1]],dtype=dtype,device=device)
    # define the Hamiltonian Hx:
    Hx = tc.zeros(2**(n_bonds+1), 2**(n_bonds+1),dtype=dtype,device=device)
    # define the Hamiltonian Hz:
    Hz = tc.zeros(2**(n_bonds+1), 2**(n_bonds+1),dtype=dtype,device=device)
    # define the Hamiltonian Hzz:
    Hzz = tc.zeros(2**(n_bonds+1), 2**(n_bonds+1),dtype=dtype,device=device)

    connect_bond_para = tc.ones(n_bonds+1,dtype=dtype,device=device)
    if n == 1:
        connect_bond_para[-1] = 0.5
    elif (n+n_bonds) == ising.n_sites:
        connect_bond_para[0] = 0.5
    else:
        connect_bond_para[-1] = 0.5
        connect_bond_para[0] = 0.5
    
    # if Full U:
    if (n == 1) & (n+n_bonds == ising.n_sites):
        connect_bond_para[-1] = 1.0
        connect_bond_para[0] = 1.0

    
    # generate the local Hamiltonian term: Hx and H_z
    # one by one and then sum over them
    for m in range(n_bonds+1):
        if m == 0:
            Hx_m = sigma_x
            Hz_m = sigma_z
        else:
            Hx_m = I2
            Hz_m = I2

        for i in range(1, n_bonds+1):
            if i == m:
                Hx_m = tc.kron(Hx_m, sigma_x)
                Hz_m = tc.kron(Hz_m, sigma_z)
            else:
                Hx_m = tc.kron(Hx_m, I2)
                Hz_m = tc.kron(Hz_m, I2)
        Hx = Hx + Hx_m * connect_bond_para[m]
        Hz = Hz + Hz_m * connect_bond_para[m]

    # generate the local Hamiltonian term: Hzz
    for m in range(n_bonds):
        if m == 0:
            Hzz_m = tc.kron(sigma_z, sigma_z)
        else:
            Hzz_m = I2
        
        for i in range(1, n_bonds):
            if i == m:
                Hzz_m = tc.kron(Hzz_m, tc.kron(sigma_z, sigma_z))
            else:
                Hzz_m = tc.kron(Hzz_m, I2)
        Hzz = Hzz + Hzz_m
    
    Hn = ising.J * Hzz + ising.hx * Hx + ising.hz * Hz

    if n + n_bonds == ising.n_sites:
        U = tc.matrix_exp(-0.5 * 1.0j * dt * Hn)
    else:
        U = tc.matrix_exp(-0.5 * 1.0j * dt * Hn)
    U_dag = U.t().conj()

    # generate the binary index
    state2col = tc.zeros(n_bonds+1,dtype=int,device=device)
    c = 0
    for i in range(n_bonds, -1, -1):
        state2col[c] = 2**i
        c+=1
    ind_state = tc.zeros(2**(n_bonds + 1), n_bonds + 1, dtype=dtype,device=device)
    # the number of state of local operator:
    n_loc_states = n_bonds + 1
    for i in range(2**(n_loc_states)):
        # tranfer decimal to binary values:
        i_bin = '{:0b}'.format(i).zfill(n_loc_states)
        for k in range(n_loc_states):
            ind_state[i, k] = tc.tensor(int(i_bin[k]))
    ind_state = -2*ind_state + 1
    return Un(U, U_dag, state2col, ind_state)


# define an n-th unitary operator with m Hamiltonian bonds:
def get_Un_closed(ising, dt, n, dtype=tc.complex64, device="cpu"):
    # define the 2 by 2 identity matrix: 
    I2 = tc.tensor([[1,0],[0,1]],dtype=dtype,device=device)
    # define sigma x:
    sigma_x = tc.tensor([[0,1],[1,0]],dtype=dtype,device=device)
    # define sigma z:
    sigma_z = tc.tensor([[1,0],[0,-1]],dtype=dtype,device=device)
    # define the Hamiltonian Hx:
    Hx = tc.zeros(2**(n), 2**(n),dtype=dtype,device=device)
    # define the Hamiltonian Hz:
    Hz = tc.zeros(2**(n), 2**(n),dtype=dtype,device=device)
    # define the Hamiltonian Hzz:
    Hzz = tc.zeros(2**(n), 2**(n),dtype=dtype,device=device)

    connect_bond_para = tc.ones(n,dtype=dtype,device=device)
    connect_bond_para[-1] = 0.5
    connect_bond_para[0] = 0.5

    # generate the local Hamiltonian term: Hx and H_z
    # one by one and then sum over them
    for m in range(n):
        if m == 0:
            Hx_m = sigma_x
            Hz_m = sigma_z
        else:
            Hx_m = I2
            Hz_m = I2

        for i in range(1, n):
            if i == m:
                Hx_m = tc.kron(Hx_m, sigma_x)
                Hz_m = tc.kron(Hz_m, sigma_z)
            else:
                Hx_m = tc.kron(Hx_m, I2)
                Hz_m = tc.kron(Hz_m, I2)
        Hx = Hx + Hx_m * connect_bond_para[m]
        Hz = Hz + Hz_m * connect_bond_para[m]

    # generate the local Hamiltonian term: Hzz
    for m in range(n-1):
        if m == 0:
            Hzz_m = tc.kron(sigma_z, sigma_z)
        else:
            Hzz_m = I2
        
        for i in range(1, n-1):
            if i == m:
                Hzz_m = tc.kron(Hzz_m, tc.kron(sigma_z, sigma_z))
            else:
                Hzz_m = tc.kron(Hzz_m, I2)
        Hzz = Hzz + Hzz_m
    
    Hn = ising.J * Hzz + ising.hx * Hx + ising.hz * Hz

    U = tc.matrix_exp(-0.5 * 1.0j * dt * Hn)
    U_dag = U.t().conj()

    # generate the binary index
    state2col = tc.zeros(n,dtype=int,device=device)
    c = 0
    for i in range(n-1, -1, -1):
        state2col[c] = 2**i
        c+=1
    ind_state = tc.zeros(2**(n), n, dtype=dtype,device=device)
    # the number of state of local operator:
    n_loc_states = n
    for i in range(2**(n_loc_states)):
        # tranfer decimal to binary values:
        i_bin = '{:0b}'.format(i).zfill(n_loc_states)
        for k in range(n_loc_states):
            ind_state[i, k] = tc.tensor(int(i_bin[k]))
    ind_state = -2*ind_state + 1
    return Un(U, U_dag, state2col, ind_state)

def get_Un_closed_full(ising, dt, n, dtype=tc.complex64, device="cpu"):
    # define the 2 by 2 identity matrix: 
    I2 = tc.tensor([[1,0],[0,1]],dtype=dtype,device=device)
    # define sigma x:
    sigma_x = tc.tensor([[0,1],[1,0]],dtype=dtype,device=device)
    # define sigma z:
    sigma_z = tc.tensor([[1,0],[0,-1]],dtype=dtype,device=device)
    # define the Hamiltonian Hx:
    Hx = tc.zeros(2**(n), 2**(n),dtype=dtype,device=device)
    # define the Hamiltonian Hz:
    Hz = tc.zeros(2**(n), 2**(n),dtype=dtype,device=device)
    # define the Hamiltonian Hzz:
    Hzz = tc.zeros(2**(n), 2**(n),dtype=dtype,device=device)

    # generate the local Hamiltonian term: Hx and H_z
    # one by one and then sum over them
    for m in range(n):
        if m == 0:
            Hx_m = sigma_x
            Hz_m = sigma_z
        else:
            Hx_m = I2
            Hz_m = I2

        for i in range(1, n):
            if i == m:
                Hx_m = tc.kron(Hx_m, sigma_x)
                Hz_m = tc.kron(Hz_m, sigma_z)
            else:
                Hx_m = tc.kron(Hx_m, I2)
                Hz_m = tc.kron(Hz_m, I2)
        Hx = Hx + Hx_m 
        Hz = Hz + Hz_m

    # generate the local Hamiltonian term: Hzz
    for m in range(n-1):
        if m == 0:
            Hzz_m = tc.kron(sigma_z, sigma_z)
        else:
            Hzz_m = I2
        
        for i in range(1, n-1):
            if i == m:
                Hzz_m = tc.kron(Hzz_m, tc.kron(sigma_z, sigma_z))
            else:
                Hzz_m = tc.kron(Hzz_m, I2)
        Hzz = Hzz + Hzz_m
    # the last one:
    Hzz_m = sigma_z
    for _ in range(n-2):
        Hzz_m = tc.kron(Hzz_m, I2)
    Hzz_m = tc.kron(Hzz_m, sigma_z)
    Hzz = Hzz + Hzz_m
    Hn = ising.J * Hzz + ising.hx * Hx + ising.hz * Hz

    U = tc.matrix_exp(-1.0j * dt * Hn)
    #U_dag = U.t().conj()

    # generate the binary index
    state2col = tc.zeros(n,dtype=int,device=device)
    c = 0
    for i in range(n-1, -1, -1):
        state2col[c] = 2**i
        c+=1
    ind_state = tc.zeros(2**(n), n, dtype=dtype,device=device)
    # the number of state of local operator:
    n_loc_states = n
    for i in range(2**(n_loc_states)):
        # tranfer decimal to binary values:
        i_bin = '{:0b}'.format(i).zfill(n_loc_states)
        for k in range(n_loc_states):
            ind_state[i, k] = tc.tensor(int(i_bin[k]))
    ind_state = -2*ind_state + 1
    return Un_full(U, state2col, ind_state)

def get_Un_full(ising, dt, n, BC:str, dtype=tc.complex64, device="cpu"):
    # define the 2 by 2 identity matrix: 
    I2 = tc.tensor([[1,0],[0,1]],dtype=dtype,device=device)
    # define sigma x:
    sigma_x = tc.tensor([[0,1],[1,0]],dtype=dtype,device=device)
    # define sigma z:
    sigma_z = tc.tensor([[1,0],[0,-1]],dtype=dtype,device=device)
    # define the Hamiltonian Hx:
    Hx = tc.zeros(2**(n), 2**(n),dtype=dtype,device=device)
    # define the Hamiltonian Hz:
    Hz = tc.zeros(2**(n), 2**(n),dtype=dtype,device=device)
    # define the Hamiltonian Hzz:
    Hzz = tc.zeros(2**(n), 2**(n),dtype=dtype,device=device)

    # generate the local Hamiltonian term: Hx and H_z
    # one by one and then sum over them
    for m in range(n):
        if m == 0:
            Hx_m = sigma_x
            Hz_m = sigma_z
        else:
            Hx_m = I2
            Hz_m = I2

        for i in range(1, n):
            if i == m:
                Hx_m = tc.kron(Hx_m, sigma_x)
                Hz_m = tc.kron(Hz_m, sigma_z)
            else:
                Hx_m = tc.kron(Hx_m, I2)
                Hz_m = tc.kron(Hz_m, I2)
        Hx = Hx + Hx_m 
        Hz = Hz + Hz_m

    # generate the local Hamiltonian term: Hzz
    for m in range(n-1):
        if m == 0:
            Hzz_m = tc.kron(sigma_z, sigma_z)
        else:
            Hzz_m = I2
        
        for i in range(1, n-1):
            if i == m:
                Hzz_m = tc.kron(Hzz_m, tc.kron(sigma_z, sigma_z))
            else:
                Hzz_m = tc.kron(Hzz_m, I2)
        Hzz = Hzz + Hzz_m
    # the last one:
    if BC == 'closed':
        Hzz_m = sigma_z
        for _ in range(n-2):
            Hzz_m = tc.kron(Hzz_m, I2)
        Hzz_m = tc.kron(Hzz_m, sigma_z)
        Hzz = Hzz + Hzz_m
    Hn = ising.J * Hzz + ising.hx * Hx + ising.hz * Hz

    U = tc.matrix_exp(-1.0j * dt * Hn)
    #U_dag = U.t().conj()

    # generate the binary index
    state2col = tc.zeros(n,dtype=int,device=device)
    c = 0
    for i in range(n-1, -1, -1):
        state2col[c] = 2**i
        c+=1
    ind_state = tc.zeros(2**(n), n, dtype=dtype,device=device)
    # the number of state of local operator:
    n_loc_states = n
    for i in range(2**(n_loc_states)):
        # tranfer decimal to binary values:
        i_bin = '{:0b}'.format(i).zfill(n_loc_states)
        for k in range(n_loc_states):
            ind_state[i, k] = tc.tensor(int(i_bin[k]))
    ind_state = -2*ind_state + 1
    return Un_full(U, state2col, ind_state)


def generate_bind_indx(n_sites, n_bonds):
    bond_indx = [1]
    while bond_indx[-1] < n_sites:
        bond_indx.append(bond_indx[-1] + n_bonds)
    if bond_indx[-1] >= n_sites:
        del bond_indx[-1]
    return bond_indx+bond_indx[::-1]


def get_Un_full_2(ising, dt, n, BC:str, dtype=tc.complex64, device="cpu"):
    basis = spin_basis(n)
    # Hx term:
    H = op.sum(ising.hx*op.x(i) for i in range(n))
    # Hz term:
    H += op.sum(ising.hz*op.z(i) for i in range(n))
    if BC == 'open':
        H += op.sum(ising.J*op.zz(i, (i+1)) for i in range(n-1))
    elif BC == 'closed':
        H += op.sum(ising.J*op.zz(i, (i+1)%n) for i in range(n))
    H = H.to_matrix(basis, pauli=True)
    H = tc.from_numpy(H).to(device)
    Es, us = tc.linalg.eigh(H)
    us = us.to(dtype)
    U = us * tc.exp(-1j*dt*Es) @ us.conj().t()
    U_dag = U.t().conj()

    # generate the binary index
    state2col = tc.zeros(n,dtype=int,device=device)
    c = 0
    for i in range(n-1, -1, -1):
        state2col[c] = 2**i
        c+=1
    ind_state = tc.zeros(2**(n), n, dtype=dtype,device=device)
    # the number of state of local operator:
    n_loc_states = n
    for i in range(2**(n_loc_states)):
        # tranfer decimal to binary values:
        i_bin = '{:0b}'.format(i).zfill(n_loc_states)
        for k in range(n_loc_states):
            ind_state[i, k] = tc.tensor(int(i_bin[k]))
    ind_state = -2*ind_state + 1
    return Un_full(U, U_dag, state2col, ind_state)