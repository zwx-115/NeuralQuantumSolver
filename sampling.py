###########################################
###########################################
### sampling part ###
'''
Here, we mainly use Monte Carlo method to 
take samples, and the method includes;
1. Markov Hasting algorithm
2. full summation
...
'''
import torch as tc

class Sampling():
    def __init__(self, n_sites, n_states, n_thermals, n_sweeps, qunt_sys_type, dtype=tc.complex64, device='cpu'):
        self.n_sites = n_sites
        self.n_states = n_states
        self.n_thermals = n_thermals
        self.n_sweeps = n_sweeps
        self.dtype = dtype
        self.device = device
        if qunt_sys_type == 'spin':
            initial_states = tc.randint(0,2,(n_states,n_sites),device=device) *2 - 1
        else:
            RuntimeError(f"No such system: {qunt_sys_type}. The option are spin.")
        self.states = initial_states.to(dtype)
        #print("the state is on the ",self.states.device)

# Markov chain with multiple samples
def single_sweep(neural_net, sample):
    #flip_states = list(range(0,sample.n_states))
    for k in range(sample.n_sites):
        # generate a candiate sample:
        state_trial = tc.clone(sample.states)
        # flip k-th spins:
        '''
        Here, we don't select spins randomly to flip, 
        we flip spins sequentially instead(filp spins from 1st to the last spin).
        '''
        flip_pos = tc.zeros(sample.n_states,dtype=tc.int,device=sample.device) + k
        state_trial[:,k] = -state_trial[:,k]
        # get probability P = |psi(state_trial) / psi(state)|^2:
        p = tc.abs(tc.exp(neural_net(state_trial).detach() - neural_net(sample.states).detach()))**2
        #print(p[:,0])
        # create a random number:
        r = tc.rand((sample.n_states,), device=sample.device )
        # if r < min(1, P), acceept the candidate state: sr=True, 
        # otherwise reject sr=False
        sr = r < tc.min(tc.tensor(1), p[:,0])
        # find the position of these accepted states:
        site_accept_indx_0 = tc.nonzero(sr==True)  
        #print(site_accept_indx_0)
        # if we choose to flip spins sequentially, 
        # the value of site_accept_indx_1 should be k
        site_accept_indx_1 = flip_pos[site_accept_indx_0]
        # flip these accepted states:
        sample.states[site_accept_indx_0, site_accept_indx_1] = -sample.states[site_accept_indx_0, site_accept_indx_1]

def single_sweep_real_valued(neural_net_am, neural_net_ph, sample):
    #flip_states = list(range(0,sample.n_states))
    for k in range(sample.n_sites):
        # generate a candiate sample:
        state_trial = tc.clone(sample.states)
        # flip k-th spins:
        '''
        Here, we don't select spins randomly to flip, 
        we flip spins sequentially instead(filp spins from 1st to the last spin).
        '''
        flip_pos = tc.zeros(sample.n_states,dtype=tc.int,device=sample.device) + k
        state_trial[:,k] = -state_trial[:,k]
        # get probability P = |psi(state_trial) / psi(state)|^2:
        p = tc.abs(tc.exp(neural_net_am(state_trial).abs().detach() + 1j*neural_net_ph(state_trial).detach()- neural_net_am(sample.states).abs().detach() - 1j*neural_net_ph(sample.states).detach()))**2
        #print(p[:,0])
        # create a random number:
        r = tc.rand((sample.n_states,), device=sample.device )
        # if r < min(1, P), acceept the candidate state: sr=True, 
        # otherwise reject sr=False
        sr = r < tc.min(tc.tensor(1), p[:,0])
        # find the position of these accepted states:
        site_accept_indx_0 = tc.nonzero(sr==True)  
        #print(site_accept_indx_0)
        # if we choose to flip spins sequentially, 
        # the value of site_accept_indx_1 should be k
        site_accept_indx_1 = flip_pos[site_accept_indx_0]
        # flip these accepted states:
        sample.states[site_accept_indx_0, site_accept_indx_1] = -sample.states[site_accept_indx_0, site_accept_indx_1]

'''
2. full summation:
Here, we get all configurations of the quantum system
'''
class SamplingExact():
    def __init__(self, n_sites,  qunt_sys_type, dtype=tc.complex64, device='cpu'):
        assert n_sites <= 14, 'the system size must be samller than 14 sites!'
        print('exact sampling')
        self.n_sites = n_sites
        self.n_states = 2**n_sites
        self.dtype = dtype
        self.device = device
        if qunt_sys_type == 'spin':
            n_states = 2**n_sites
            exact_state = tc.zeros(self.n_states, n_sites, dtype=dtype,device=device)
            for i in range(n_states):
                i_bin = '{:0b}'.format(i).zfill(n_sites)
                for k in range(n_sites):
                    exact_state[i,k] = tc.tensor(int(i_bin[k])).to(device)
            exact_state = -2 * exact_state + 1
        else:
            RuntimeError(f"No such system: {qunt_sys_type}. The option are spin.")
        self.states = exact_state

'''
2. full summation:
Here, we get all configurations of the quantum system
'''
class SamplingLocalExact():
    def __init__(self, n_sites, n_states, qunt_sys_type, dtype=tc.complex64, device='cpu'):
        print('local exact sampling')
        self.n_sites = n_sites
        self.n_states = n_states
        assert n_states <= 10000, 'the maximum number of unique samples must be less than 10000'
        self.dtype = dtype
        self.device = device
        if qunt_sys_type == 'spin':
            states = tc.randint(0,2,(50000,n_sites)) *2 - 1
            unique_states = tc.unique(states, dim=0)
            local_uni_states = unique_states[:self.n_states, :].to(dtype).to(device)
        else:
            RuntimeError(f"No such system: {qunt_sys_type}. The option are spin.")
        self.states = local_uni_states

