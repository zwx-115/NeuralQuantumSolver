###########################################
###########################################
### local energy ###
import torch as tc
import time 

'''
This is the realization of local energy:
energy can be obtained with Monte Carlo method.

Reference:
[1]. Carleo G, Troyer M. Solving the quantum many-body 
problem with artificial neural networks[J]. 
Science, 2017, 355(6325): 602-606.
'''
# only for ising model now,
# big room for improvement
def local_energy(ising, neural_net, sample):
    # initialize the local energy 
    E_loc = tc.zeros(sample.n_states, dtype=neural_net.dtype,device=neural_net.device)

    # 1.calculate diagonalized elements of Hamiltonian:
    # 1.1 hz*sigma^z
    E_loc += ising.hz * tc.sum(sample.states, 1)
    # 1.2 J*sigma^z_i sigma^z_i+1
    E_loc += ising.J * tc.sum(sample.states[:,ising.bond_site[:,0]] * sample.states[:,ising.bond_site[:,1]] , 1)

    # calculate off-diagonalized elements of Hamiltonian:
    output_init = neural_net(sample.states).detach()
    for k in range(ising.n_sites):
        new_states = tc.clone(sample.states)
        new_states[:,k] = -new_states[:,k]
        E_loc += ising.hx * tc.exp(neural_net(new_states).detach() - output_init)[:,0].detach()
    return E_loc

def local_energy_unique(ising, neural_net, states_unique):
    # initialize the local energy 
    n_states_uni = states_unique.size(0)
    E_loc = tc.zeros(n_states_uni, dtype=neural_net.dtype,device=neural_net.device)

    # 1.calculate diagonalized elements of Hamiltonian:
    # 1.1 hz*sigma^z
    E_loc += ising.hz * tc.sum(states_unique, 1)
    # 1.2 J*sigma^z_i sigma^z_i+1
    E_loc += ising.J * tc.sum(states_unique[:,ising.bond_site[:,0]] * states_unique[:,ising.bond_site[:,1]] , 1)

    # calculate off-diagonalized elements of Hamiltonian:
    output_init = neural_net(states_unique).detach()
    for k in range(ising.n_sites):
        new_states = tc.clone(states_unique)
        new_states[:,k] = -new_states[:,k]
        is_in_group = tc.zeros(n_states_uni,dtype=tc.int,device=neural_net.device)
        for k in range(n_states_uni):
            new_states_roll = tc.roll(new_states, k, dims=0)
            is_state_eq = tc.eq(states_unique, new_states_roll).int()
            is_state_eq = is_state_eq.prod(1)
            is_in_group = (is_in_group | is_state_eq).int()
        is_in_group = is_in_group.to(neural_net.dtype).view(-1,1)
        # state_group = tc.cat((states_unique,new_states),0)
        # # check which states are in the unique states group after flip
        # state_check = tc.unique(state_group,dim=0,return_counts=True)
        # # if the counter is 2, that means the state is in unique states
        # in_group_indx = tc.nonzero(state_check[1] == 2)
        # state_pick = state_check[0][in_group_indx[:,0],:] 
        # new_states = tc.clone(state_pick)
        # new_states[:,k] = -new_states[:,k]
        E_loc += ising.hx * tc.exp(neural_net(new_states).detach() - output_init)[:,0] * is_in_group[:,0]
    return E_loc

'''
We have 2 real-valued neural networks, whihc are used to represent amplitude and the phase, respectively.
'''
def local_energy_real_valued(ising, neural_net_am, neural_net_ph, sample):
    # initialize the local energy 
    E_loc = tc.zeros(sample.n_states, dtype=tc.complex128,device=neural_net_am.device)

    # 1.calculate diagonalized elements of Hamiltonian:
    # 1.1 hz*sigma^z
    E_loc += ising.hz * tc.sum(sample.states, 1)
    # 1.2 J*sigma^z_i sigma^z_i+1
    E_loc += ising.J * tc.sum(sample.states[:,ising.bond_site[:,0]] * sample.states[:,ising.bond_site[:,1]] , 1)

    # calculate off-diagonalized elements of Hamiltonian:
    output_init_am = neural_net_am(sample.states).abs().detach()
    output_init_ph = neural_net_ph(sample.states).detach()
    output_init = output_init_am + output_init_ph *1j
    for k in range(ising.n_sites):
        new_states = tc.clone(sample.states)
        new_states[:,k] = -new_states[:,k]
        E_loc += ising.hx * tc.exp(neural_net_am(new_states).abs().detach() + neural_net_ph(new_states).detach() *1j - output_init)[:,0]
    return E_loc

###########################################
###########################################
## time evolution part ##

'''
In this part, we use the concept of overlap as the cost function and generate the local energy (unitary operator) by Trotter decomposition and Taylor expansion up to 2nd order.
'''
def local_energy_psi_phi_trotter(nn_0, nn_t, Uk, sample, bond_ind):
    E_psi_phi = tc.zeros(sample.n_states, dtype=nn_0.dtype,device=nn_0.device)
    local_state = sample.states[:,bond_ind]
    # convert the local state to the position of element in Uk
    U_col = ((-(local_state -1)* Uk.ind_U_col / 2).sum(dim=1)).to(int)

    H_size = 2**len(bond_ind)
    res = nn_t(sample.states).detach()
    for k in range(H_size):
        new_states = tc.clone(sample.states)
        new_states[:,bond_ind] = Uk.ind_state[k,:]
        E_psi_phi += tc.exp(nn_0(new_states).detach() - res)[:,0] * Uk.U[U_col, k]
    return E_psi_phi

def local_energy_phi_psi_trotter(nn_0, nn_t, Uk, sample, bond_ind):
    E_phi_psi = tc.zeros(sample.n_states, dtype=nn_0.dtype,device=nn_0.device)
    local_state = sample.states[:,bond_ind]
    # convert the local state to the position of element in Uk
    U_col = ((-(local_state -1)* Uk.ind_U_col / 2).sum(dim=1)).to(int)
    H_size = 2**len(bond_ind)
    res = nn_0(sample.states).detach()
    for k in range(H_size):
        new_states = tc.clone(sample.states)
        new_states[:,bond_ind] = Uk.ind_state[k,:]
        E_phi_psi += tc.exp(nn_t(new_states).detach() - res)[:,0] * Uk.U_dag[U_col, k]
    return E_phi_psi

# MC method for small system(we can still find full samples)
def local_energy_psi_phi_trotter_2(nn_0, nn_t, Uk, sample):

    # convert the local state to the position of element in Uk
    res_t = nn_t(sample.states)[:,0].detach()
    res_0 = nn_0(Uk.ind_state)[:,0].detach()
    U_col = ((-(sample.states -1)* Uk.ind_U_col / 2).sum(dim=1)).to(int)
    E_psi_phi = tc.exp( -1*res_t.reshape(-1,1) + (res_0.reshape(-1) + tc.log(Uk.U[U_col,:])) ).sum(dim=1)
    #E_psi_phi = (tc.log(Uk.U[U_col,:]) @ res_0.reshape(-1,1)) /tc.exp(res_t[U_col].view(-1,1))
    # print('------')
    # print(res_0.reshape(-1).size())
    # print(E_psi_phi.size())
    return E_psi_phi

def local_energy_phi_psi_trotter_2(nn_0, nn_t, Uk, sample):
    res_t = nn_t(Uk.ind_state)[:,0].detach()
    res_0 = nn_0(sample.states)[:,0].detach()
    U_col = ((-(sample.states -1)* Uk.ind_U_col / 2).sum(dim=1)).to(int)
    # convert the local state to the position of element in Uk
    
    E_phi_psi = tc.exp( -1*res_0.reshape(-1,1) + (res_t.reshape(-1) + tc.log(Uk.U_dag[U_col,:])) ).sum(dim=1)
    return E_phi_psi


def local_energy_psi_phi_trotter_full(nn_0, nn_t, Uk, sample):
    
    # convert the local state to the position of element in Uk
    res_t = nn_t(sample.states)[:,0].detach()
    res_0 = nn_0(sample.states)[:,0].detach()
    E_psi_phi = tc.exp( -1*res_t.reshape(-1,1) + res_0.reshape(1,-1) + tc.log(Uk.U) ).sum(dim=1)
    return E_psi_phi

def local_energy_phi_psi_trotter_full(nn_0, nn_t, Uk, sample):
    # convert the local state to the position of element in Uk
    res_t = nn_t(sample.states)[:,0].detach()
    res_0 = nn_0(sample.states)[:,0].detach()
    E_phi_psi = tc.sum(tc.exp( -1*res_0.reshape(-1,1) + res_t.reshape(1,-1) + tc.log(Uk.U.t().conj()) ),dim=1)
    return E_phi_psi
### Taylor Expansion ###
'''
This is for computing 'local energy' with Taylor expansion:
U = 1 -i*dt*H - 0.5 * dt * H^2 + O(dt^3)
E_psi_phi = 1/N_s * \sum_x <\phi|U|\psi> / <\phi|\phi>
'''
def local_energy_psi_phi_taylor(ising, dt, neural_net_psi, neural_net_phi, sample):
    # E_loc_1st is used for computing <\phi|H|\psi> / <\phi|\phi>, which is smilar with local energy:
    E_loc_1st = tc.zeros(sample.n_states, dtype=neural_net_psi.dtype,device=neural_net_psi.device)
    # E_loc_2nd is used for computing <\phi|H^2|\psi> / <\phi|\phi>, which is smilar with local energy:
    E_loc_2nd = tc.zeros(sample.n_states, dtype=neural_net_psi.dtype,device=neural_net_psi.device)

    # E_loc_diag: compute diagonal part: H_xx
    E_loc_diag = tc.zeros(sample.n_states, dtype=neural_net_psi.dtype,device=neural_net_psi.device)
    psi_x = neural_net_psi(sample.states).detach()
    phi_x = neural_net_phi(sample.states).detach()
    # daigonal part
    E_loc_diag += ising.hz * tc.sum(sample.states, 1) 
    E_loc_diag += ising.J * tc.sum(sample.states[:,ising.bond_site[:,0]] * sample.states[:,ising.bond_site[:,1]] , 1)
    #print(E_loc_diag.size())
    # compute: sum_x H_xx * \psi(x) / \phi(x)
    E_loc_1st += E_loc_diag * tc.exp(psi_x - phi_x)[:,0]
    # compute: sum_x H_xx * sum_x H_xz \psi(z) / \phi(x)
    E_loc_2nd += E_loc_diag * local_energy_taylor(ising, phi_x, neural_net_psi, sample.states)
    # off-diagonal part:
    for k in range(ising.n_sites):
        new_states = tc.clone(sample.states)
        new_states[:,k] = -new_states[:,k]
        # compute: sum_x H_xx' * \psi(x) / \phi(x)
        E_loc_1st += ising.hx * tc.exp(neural_net_psi(new_states).detach() - phi_x)[:,0]
        # compute: sum_x H_xx' * sum_x' H_x'z \psi(z) / \phi(x)
        E_loc_2nd += ising.hx * local_energy_taylor(ising, phi_x, neural_net_psi, new_states)
    # the first term is \psi(z) / \phi(x)
    E_loc_taylor = tc.exp(psi_x - phi_x)[:,0] - 1j * dt * E_loc_1st -0.5 * (dt**2) * E_loc_2nd
    return E_loc_taylor
    #return E_loc_1st

'''
This is for computing 'local energy' with Taylor expansion:
U = 1 +i*dt*H - 0.5 * dt * H^2 + O(dt^3)
E_psi_phi = 1/N_s * \sum_x <\psi|U|\phi> / <\psi|\psi>
'''
def local_energy_phi_psi_taylor(ising, dt, neural_net_psi, neural_net_phi, sample):
    # E_loc_1st is used for computing <\phi|H|\psi> / <\phi|\phi>, which is smilar with local energy:
    E_loc_1st = tc.zeros(sample.n_states, dtype=neural_net_psi.dtype,device=neural_net_psi.device)
    # E_loc_2nd is used for computing <\phi|H^2|\psi> / <\phi|\phi>, which is smilar with local energy:
    E_loc_2nd = tc.zeros(sample.n_states, dtype=neural_net_psi.dtype,device=neural_net_psi.device)

    # E_loc_diag: compute diagonal part: H_xx
    E_loc_diag = tc.zeros(sample.n_states, dtype=neural_net_psi.dtype,device=neural_net_psi.device)
    psi_x = neural_net_psi(sample.states).detach()
    phi_x = neural_net_phi(sample.states).detach()
    # daigonal part
    E_loc_diag += ising.hz * tc.sum(sample.states, 1) 
    E_loc_diag += ising.J * tc.sum(sample.states[:,ising.bond_site[:,0]] * sample.states[:,ising.bond_site[:,1]] , 1)
    #print(E_loc_diag.size())
    # compute: sum_x H_xx * \phi(x) / \psi(x)
    E_loc_1st += E_loc_diag * tc.exp(phi_x - psi_x)[:,0]
    # compute: sum_x H_xx * sum_x H_xz \phi(z) / \psi(x)
    E_loc_2nd += E_loc_diag * local_energy_taylor(ising, psi_x, neural_net_phi, sample.states)
    # off-diagonal part:
    for k in range(ising.n_sites):
        new_states = tc.clone(sample.states)
        new_states[:,k] = -new_states[:,k]
        # compute: sum_x H_xx' * \phi(x) / \psi(x)
        E_loc_1st += ising.hx * tc.exp(neural_net_phi(new_states).detach() - psi_x)[:,0]
        # compute: sum_x H_xx' * sum_x' H_x'z \phi(z) / \psi(x)
        E_loc_2nd += ising.hx * local_energy_taylor(ising, psi_x, neural_net_phi, new_states)
    # the first term is \phi(z) / \psi(x)
    E_loc_taylor = tc.exp(phi_x - psi_x)[:,0] + 1j * dt * E_loc_1st -0.5 * (dt**2) * E_loc_2nd
    return E_loc_taylor
    #return E_loc_1st

'''
This is ued for computing:
sum_y H_yz * \psi(z) / phi(x)
or 
sum_y H_yz * \phi(z) / psi(x)
where, the argument nn_output is phi(x) or psi(x)
'''
def local_energy_taylor(ising, nn_output, neural_net,  initial_states):
    # initialize the local energy 
    n_states = initial_states.size(0)
    E_loc = tc.zeros(n_states, dtype=neural_net.dtype,device=neural_net.device)
    # 1.calculate diagonalized elements of Hamiltonian:
    # 1.1 hz*sigma^z
    E_loc_diag = tc.zeros(n_states, dtype=neural_net.dtype,device=neural_net.device)
    E_loc_diag += ising.hz * tc.sum(initial_states, 1)
    # 1.2 J*sigma^z_i sigma^z_i+1
    E_loc_diag += ising.J * tc.sum(initial_states[:,ising.bond_site[:,0]] * initial_states[:,ising.bond_site[:,1]] , 1)
    # compute sum_y H_yy * \psi(y) / phi(x)
    # or 
    # sum_y H_yy * \phi(y) / psi(x)
    E_loc += E_loc_diag * tc.exp(neural_net(initial_states).detach() - nn_output)[:,0].detach()

    # calculate off-diagonalized elements of Hamiltonian:
    # compute sum_y H_yz * \psi(z) / phi(x)
    # or 
    # sum_y H_yz * \phi(z) / psi(x)
    for k in range(ising.n_sites):
        new_states = tc.clone(initial_states)
        new_states[:,k] = -new_states[:,k]
        E_loc += ising.hx * tc.exp(neural_net(new_states).detach() - nn_output)[:,0].detach() 
    return E_loc

### Taylor Expansion vectorization ###
'''
This is for computing 'local energy' with Taylor expansion:
U = 1 -i*dt*H - 0.5 * dt * H^2 + O(dt^3)
E_psi_phi = 1/N_s * \sum_x <\phi|U|\psi> / <\phi|\phi>
'''
def local_energy_psi_phi_taylor_vectorized(ising, dt, neural_net_psi, neural_net_phi, sample, flip_matrix):
    # E_loc_1st is used for computing <\phi|H|\psi> / <\phi|\phi>, which is smilar with local energy:
    E_loc_1st = tc.zeros(sample.n_states, dtype=neural_net_psi.dtype,device=neural_net_psi.device)
    # E_loc_2nd is used for computing <\phi|H^2|\psi> / <\phi|\phi>, which is smilar with local energy:
    E_loc_2nd = tc.zeros(sample.n_states, dtype=neural_net_psi.dtype,device=neural_net_psi.device)

    # E_loc_diag: compute diagonal part: H_xx
    E_loc_diag = tc.zeros(sample.n_states, dtype=neural_net_psi.dtype,device=neural_net_psi.device)
    psi_x = neural_net_psi(sample.states).detach()
    phi_x = neural_net_phi(sample.states).detach()
    # daigonal part
    E_loc_diag += ising.hz * tc.sum(sample.states, 1) 
    E_loc_diag += ising.J * tc.sum(sample.states[:,ising.bond_site[:,0]] * sample.states[:,ising.bond_site[:,1]] , 1)
    #print(E_loc_diag.size())
    # compute: sum_x H_xx * \psi(x) / \phi(x)
    E_loc_1st += E_loc_diag * tc.exp(psi_x - phi_x)[:,0]
    # compute: sum_x H_xx * sum_x H_xz \psi(z) / \phi(x)
    E_loc_2nd += E_loc_diag * local_energy_taylor(ising, phi_x, neural_net_psi, sample.states)
    # off-diagonal part with vectorization:
    flip_states = tc.clone(sample.states) * flip_matrix
    E_loc_1st += ising.hx * tc.exp(neural_net_psi(flip_states).detach() - phi_x).sum(0)[:,0].detach()
    for k in range(ising.n_sites):
        new_states = tc.clone(sample.states)
        new_states[:,k] = -new_states[:,k]
        # compute: sum_x H_xx' * \psi(x) / \phi(x)
        #E_loc_1st += ising.hx * tc.exp(neural_net_psi(new_states).detach() - phi_x)[:,0]
        # compute: sum_x H_xx' * sum_x' H_x'z \psi(z) / \phi(x)
        E_loc_2nd += ising.hx * local_energy_taylor_vectorized(ising, phi_x, neural_net_psi, new_states, flip_matrix)
        #E_loc_2nd += ising.hx * local_energy_taylor(ising, phi_x, neural_net_psi, new_states)
    # the first term is \psi(z) / \phi(x)
    E_loc_taylor = tc.exp(psi_x - phi_x)[:,0] - 1j * dt * E_loc_1st -0.5 * (dt**2) * E_loc_2nd
    return E_loc_taylor
    #return E_loc_1st

'''
This is for computing 'local energy' with Taylor expansion:
U = 1 +i*dt*H - 0.5 * dt * H^2 + O(dt^3)
E_psi_phi = 1/N_s * \sum_x <\psi|U|\phi> / <\psi|\psi>
'''
def local_energy_phi_psi_taylor_vectorized(ising, dt, neural_net_psi, neural_net_phi, sample, flip_matrix):
    # E_loc_1st is used for computing <\phi|H|\psi> / <\phi|\phi>, which is smilar with local energy:
    E_loc_1st = tc.zeros(sample.n_states, dtype=neural_net_psi.dtype,device=neural_net_psi.device)
    # E_loc_2nd is used for computing <\phi|H^2|\psi> / <\phi|\phi>, which is smilar with local energy:
    E_loc_2nd = tc.zeros(sample.n_states, dtype=neural_net_psi.dtype,device=neural_net_psi.device)

    # E_loc_diag: compute diagonal part: H_xx
    E_loc_diag = tc.zeros(sample.n_states, dtype=neural_net_psi.dtype,device=neural_net_psi.device)
    psi_x = neural_net_psi(sample.states).detach()
    phi_x = neural_net_phi(sample.states).detach()
    # daigonal part
    E_loc_diag += ising.hz * tc.sum(sample.states, 1) 
    E_loc_diag += ising.J * tc.sum(sample.states[:,ising.bond_site[:,0]] * sample.states[:,ising.bond_site[:,1]] , 1)
    #print(E_loc_diag.size())
    # compute: sum_x H_xx * \phi(x) / \psi(x)
    E_loc_1st += E_loc_diag * tc.exp(phi_x - psi_x)[:,0]
    # compute: sum_x H_xx * sum_x H_xz \phi(z) / \psi(x)
    E_loc_2nd += E_loc_diag * local_energy_taylor(ising, psi_x, neural_net_phi, sample.states)

    # off-diagonal part with vectorization:
    flip_states = tc.clone(sample.states) * flip_matrix
    E_loc_1st += ising.hx * tc.exp(neural_net_phi(flip_states).detach() - psi_x).sum(0)[:,0].detach()

    # off-diagonal part:
    for k in range(ising.n_sites):
        new_states = tc.clone(sample.states)
        new_states[:,k] = -new_states[:,k]
        # compute: sum_x H_xx' * \phi(x) / \psi(x)
        #E_loc_1st += ising.hx * tc.exp(neural_net_phi(new_states).detach() - psi_x)[:,0]
        # compute: sum_x H_xx' * sum_x' H_x'z \phi(z) / \psi(x)
        E_loc_2nd += ising.hx * local_energy_taylor_vectorized(ising, psi_x, neural_net_phi, new_states, flip_matrix)
    # the first term is \phi(z) / \psi(x)
    E_loc_taylor = tc.exp(phi_x - psi_x)[:,0] + 1j * dt * E_loc_1st -0.5 * (dt**2) * E_loc_2nd
    return E_loc_taylor
    #return E_loc_1st

'''
This is ued for computing:
sum_y H_yz * \psi(z) / phi(x)
or 
sum_y H_yz * \phi(z) / psi(x)
where, the argument nn_output is phi(x) or psi(x)
'''
def local_energy_taylor_vectorized(ising, nn_output, neural_net, initial_states, flip_matrix):
    # initialize the local energy 
    n_states = initial_states.size(0)
    E_loc = tc.zeros(n_states, dtype=neural_net.dtype,device=neural_net.device)
    # 1.calculate diagonalized elements of Hamiltonian:
    # 1.1 hz*sigma^z
    E_loc_diag = tc.zeros(n_states, dtype=neural_net.dtype,device=neural_net.device)
    E_loc_diag += ising.hz * tc.sum(initial_states, 1)
    # 1.2 J*sigma^z_i sigma^z_i+1
    E_loc_diag += ising.J * tc.sum(initial_states[:,ising.bond_site[:,0]] * initial_states[:,ising.bond_site[:,1]] , 1)
    # compute sum_y H_yy * \psi(y) / phi(x)
    # or 
    # sum_y H_yy * \phi(y) / psi(x)
    E_loc += E_loc_diag * tc.exp(neural_net(initial_states).detach() - nn_output)[:,0].detach()

    # calculate off-diagonalized elements of Hamiltonian:
    # compute sum_y H_yz * \psi(z) / phi(x)
    # or 
    # sum_y H_yz * \phi(z) / psi(x)
    flip_states = initial_states * flip_matrix
    E_loc += ising.hx * tc.exp(neural_net(flip_states).detach() - nn_output).sum(0)[:,0].detach()
    return E_loc



### Taylor Expansion vectorization with time-dependent input###
'''
This is for computing 'local energy' with Taylor expansion:
U = 1 -i*dt*H - 0.5 * dt * H^2 + O(dt^3)
E_psi_phi = 1/N_s * \sum_x <\phi|U|\psi> / <\phi|\phi>
'''
def Eloc_psiphi_taylor_vect_t(ising, dt, neural_net_psi, neural_net_phi, sample, flip_matrix):
    # E_loc_1st is used for computing <\phi|H|\psi> / <\phi|\phi>, which is smilar with local energy:
    E_loc_1st = tc.zeros(sample.n_states, dtype=neural_net_psi.dtype,device=neural_net_psi.device)
    # E_loc_2nd is used for computing <\phi|H^2|\psi> / <\phi|\phi>, which is smilar with local energy:
    E_loc_2nd = tc.zeros(sample.n_states, dtype=neural_net_psi.dtype,device=neural_net_psi.device)

    # E_loc_diag: compute diagonal part: H_xx
    E_loc_diag = tc.zeros(sample.n_states, dtype=neural_net_psi.dtype,device=neural_net_psi.device)
    psi_x = neural_net_psi(sample.states).detach()
    phi_x = neural_net_phi(sample.states).detach()
    # daigonal part
    E_loc_diag += ising.hz * tc.sum(sample.states, 1) 
    E_loc_diag += ising.J * tc.sum(sample.states[:,ising.bond_site[:,0]] * sample.states[:,ising.bond_site[:,1]] , 1)
    #print(E_loc_diag.size())
    # compute: sum_x H_xx * \psi(x) / \phi(x)
    E_loc_1st += E_loc_diag * tc.exp(psi_x - phi_x)[:,0]
    # compute: sum_x H_xx * sum_x H_xz \psi(z) / \phi(x)
    E_loc_2nd += E_loc_diag * local_energy_taylor(ising, phi_x, neural_net_psi, sample.states)
    # off-diagonal part with vectorization:
    flip_states = tc.clone(sample.states) * flip_matrix
    E_loc_1st += ising.hx * tc.exp(neural_net_psi(flip_states).detach() - phi_x).sum(0)[:,0].detach()
    for k in range(ising.n_sites):
        new_states = tc.clone(sample.states)
        new_states[:,k] = -new_states[:,k]
        # compute: sum_x H_xx' * \psi(x) / \phi(x)
        #E_loc_1st += ising.hx * tc.exp(neural_net_psi(new_states).detach() - phi_x)[:,0]
        # compute: sum_x H_xx' * sum_x' H_x'z \psi(z) / \phi(x)
        E_loc_2nd += ising.hx * local_energy_taylor_vectorized(ising, phi_x, neural_net_psi, new_states, flip_matrix)
        #E_loc_2nd += ising.hx * local_energy_taylor(ising, phi_x, neural_net_psi, new_states)
    # the first term is \psi(z) / \phi(x)
    E_loc_taylor = tc.exp(psi_x - phi_x)[:,0] - 1j * dt * E_loc_1st -0.5 * (dt**2) * E_loc_2nd
    return E_loc_taylor
    #return E_loc_1st

'''
This is for computing 'local energy' with Taylor expansion:
U = 1 +i*dt*H - 0.5 * dt * H^2 + O(dt^3)
E_psi_phi = 1/N_s * \sum_x <\psi|U|\phi> / <\psi|\psi>
'''
def Eloc_phipsi_taylor_vect_t(ising, dt, neural_net_psi, neural_net_phi, sample, flip_matrix):
    # E_loc_1st is used for computing <\phi|H|\psi> / <\phi|\phi>, which is smilar with local energy:
    E_loc_1st = tc.zeros(sample.n_states, dtype=neural_net_psi.dtype,device=neural_net_psi.device)
    # E_loc_2nd is used for computing <\phi|H^2|\psi> / <\phi|\phi>, which is smilar with local energy:
    E_loc_2nd = tc.zeros(sample.n_states, dtype=neural_net_psi.dtype,device=neural_net_psi.device)

    # E_loc_diag: compute diagonal part: H_xx
    E_loc_diag = tc.zeros(sample.n_states, dtype=neural_net_psi.dtype,device=neural_net_psi.device)
    psi_x = neural_net_psi(sample.states).detach()
    phi_x = neural_net_phi(sample.states).detach()
    # daigonal part
    E_loc_diag += ising.hz * tc.sum(sample.states, 1) 
    E_loc_diag += ising.J * tc.sum(sample.states[:,ising.bond_site[:,0]] * sample.states[:,ising.bond_site[:,1]] , 1)
    #print(E_loc_diag.size())
    # compute: sum_x H_xx * \phi(x) / \psi(x)
    E_loc_1st += E_loc_diag * tc.exp(phi_x - psi_x)[:,0]
    # compute: sum_x H_xx * sum_x H_xz \phi(z) / \psi(x)
    E_loc_2nd += E_loc_diag * local_energy_taylor(ising, psi_x, neural_net_phi, sample.states)

    # off-diagonal part with vectorization:
    flip_states = tc.clone(sample.states) * flip_matrix
    E_loc_1st += ising.hx * tc.exp(neural_net_phi(flip_states).detach() - psi_x).sum(0)[:,0].detach()

    # off-diagonal part:
    for k in range(ising.n_sites):
        new_states = tc.clone(sample.states)
        new_states[:,k] = -new_states[:,k]
        # compute: sum_x H_xx' * \phi(x) / \psi(x)
        #E_loc_1st += ising.hx * tc.exp(neural_net_phi(new_states).detach() - psi_x)[:,0]
        # compute: sum_x H_xx' * sum_x' H_x'z \phi(z) / \psi(x)
        E_loc_2nd += ising.hx * local_energy_taylor_vectorized(ising, psi_x, neural_net_phi, new_states, flip_matrix)
    # the first term is \phi(z) / \psi(x)
    E_loc_taylor = tc.exp(phi_x - psi_x)[:,0] + 1j * dt * E_loc_1st -0.5 * (dt**2) * E_loc_2nd
    return E_loc_taylor
    #return E_loc_1st

'''
This is ued for computing:
sum_y H_yz * \psi(z) / phi(x)
or 
sum_y H_yz * \phi(z) / psi(x)
where, the argument nn_output is phi(x) or psi(x)
'''
def local_energy_taylor_vectorized(ising, nn_output, neural_net, initial_states, flip_matrix):
    # initialize the local energy 
    n_states = initial_states.size(0)
    E_loc = tc.zeros(n_states, dtype=neural_net.dtype,device=neural_net.device)
    # 1.calculate diagonalized elements of Hamiltonian:
    # 1.1 hz*sigma^z
    E_loc_diag = tc.zeros(n_states, dtype=neural_net.dtype,device=neural_net.device)
    E_loc_diag += ising.hz * tc.sum(initial_states, 1)
    # 1.2 J*sigma^z_i sigma^z_i+1
    E_loc_diag += ising.J * tc.sum(initial_states[:,ising.bond_site[:,0]] * initial_states[:,ising.bond_site[:,1]] , 1)
    # compute sum_y H_yy * \psi(y) / phi(x)
    # or 
    # sum_y H_yy * \phi(y) / psi(x)
    E_loc += E_loc_diag * tc.exp(neural_net(initial_states).detach() - nn_output)[:,0].detach()

    # calculate off-diagonalized elements of Hamiltonian:
    # compute sum_y H_yz * \psi(z) / phi(x)
    # or 
    # sum_y H_yz * \phi(z) / psi(x)
    flip_states = initial_states * flip_matrix
    E_loc += ising.hx * tc.exp(neural_net(flip_states).detach() - nn_output).sum(0)[:,0].detach()
    return E_loc

###########################################
###########################################
### LPE ###

### Taylor Expansion ###
'''
This is for computing 'local energy' with Taylor expansion:
U = 1 -i*dt*H *a
E_psi_phi = 1/N_s * \sum_x <\phi|U|\psi> / <\phi|\phi>
'''
def local_energy_psi_phi_LPE(ising, dt, a, neural_net_psi, neural_net_phi, sample):
    # E_loc is used for computing <\phi|H|\psi> / <\phi|\phi>, which is smilar with local energy:
    E_loc = tc.ones(sample.n_states, dtype=neural_net_psi.dtype,device=neural_net_psi.device).detach()
    
    # E_loc_diag: compute diagonal part: H_xx
    E_loc_diag = tc.zeros(sample.n_states, dtype=neural_net_psi.dtype,device=neural_net_psi.device)
    psi_x = neural_net_psi(sample.states).detach()
    phi_x = neural_net_phi(sample.states).detach()
    # daigonal part
    E_loc_diag += ising.hz * tc.sum(sample.states, 1) 
    E_loc_diag += ising.J * tc.sum(sample.states[:,ising.bond_site[:,0]] * sample.states[:,ising.bond_site[:,1]] , 1)
    #print(E_loc_diag.size())
    # compute: sum_x H_xx * \psi(x) / \phi(x)
    E_loc += E_loc_diag * tc.exp(psi_x - phi_x)[:,0]
    
    # off-diagonal part:
    for k in range(ising.n_sites):
        new_states = tc.clone(sample.states)
        new_states[:,k] = -new_states[:,k]
        # compute: sum_x H_xx' * \psi(x) / \phi(x)
        E_loc += ising.hx * tc.exp(neural_net_psi(new_states).detach() - phi_x)[:,0]
    # the first term is \psi(z) / \phi(x)
    E_loc_taylor = tc.exp(psi_x - phi_x)[:,0] - 1j * dt * a * E_loc
    return E_loc_taylor
    #return E_loc_1st

'''
This is for computing 'local energy' with Taylor expansion:
U = 1 +i*dt*H - 0.5 * dt * H^2 + O(dt^3)
E_psi_phi = 1/N_s * \sum_x <\psi|U|\phi> / <\psi|\psi>
'''
def local_energy_phi_psi_LPE(ising, dt, a, neural_net_psi, neural_net_phi, sample):
    # E_loc is used for computing <\phi|H|\psi> / <\phi|\phi>, which is smilar with local energy:
    E_loc = tc.zeros(sample.n_states, dtype=neural_net_psi.dtype,device=neural_net_psi.device)
    
    # E_loc_diag: compute diagonal part: H_xx
    E_loc_diag = tc.zeros(sample.n_states, dtype=neural_net_psi.dtype,device=neural_net_psi.device)
    psi_x = neural_net_psi(sample.states).detach()
    phi_x = neural_net_phi(sample.states).detach()
    # daigonal part
    E_loc_diag += ising.hz * tc.sum(sample.states, 1) 
    E_loc_diag += ising.J * tc.sum(sample.states[:,ising.bond_site[:,0]] * sample.states[:,ising.bond_site[:,1]] , 1)
    #print(E_loc_diag.size())
    # compute: sum_x H_xx * \phi(x) / \psi(x)
    E_loc += E_loc_diag * tc.exp(phi_x - psi_x)[:,0]
    # off-diagonal part:
    for k in range(ising.n_sites):
        new_states = tc.clone(sample.states)
        new_states[:,k] = -new_states[:,k]
        # compute: sum_x H_xx' * \phi(x) / \psi(x)
        E_loc += ising.hx * tc.exp(neural_net_phi(new_states).detach() - psi_x)[:,0]
    # the first term is \phi(z) / \psi(x)
    E_loc_taylor = tc.exp(phi_x - psi_x)[:,0] + 1j * dt * a * E_loc 
    return E_loc_taylor
    #return E_loc_1st




###########################################
###########################################
## time evolution part ##
## new idea ##
'''
In this part, we use the concept of overlap as the cost function and generate the local energy (unitary operator) by Trotter decomposition and Taylor expansion up to 2nd order.
'''
def local_energy_t_psi_phi_trotter(t0, t1, nn_0, nn_t, Uk, sample, bond_ind):
    E_psi_phi = tc.zeros(sample.n_states, dtype=nn_0.dtype,device=nn_0.device)
    local_state = sample.states[:,bond_ind]
    # convert the local state to the position of element in Uk
    U_col = ((-(local_state -1)* Uk.ind_U_col / 2).sum(dim=1)).to(int)

    H_size = 2**len(bond_ind)
    res = nn_t(sample.states, t1)
    for k in range(H_size):
        new_states = tc.clone(sample.states)
        new_states[:,bond_ind] = Uk.ind_state[k,:]
        E_psi_phi += tc.exp(nn_0(new_states, t0) - res)[:,0] * Uk.U[U_col, k]
    return E_psi_phi

def local_energy_t_phi_psi_trotter(t0, t1, nn_0, nn_t, Uk, sample, bond_ind):
    E_phi_psi = tc.zeros(sample.n_states, dtype=nn_0.dtype,device=nn_0.device)
    local_state = sample.states[:,bond_ind]
    # convert the local state to the position of element in Uk
    U_col = ((-(local_state -1)* Uk.ind_U_col / 2).sum(dim=1)).to(int)
    H_size = 2**len(bond_ind)
    res = nn_0(sample.states, t0)
    for k in range(H_size):
        new_states = tc.clone(sample.states)
        new_states[:,bond_ind] = Uk.ind_state[k,:]
        E_phi_psi += tc.exp(nn_t(new_states, t1) - res)[:,0] * Uk.U_dag[U_col, k]
    return E_phi_psi