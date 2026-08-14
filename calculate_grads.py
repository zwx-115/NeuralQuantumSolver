###########################################
###########################################

### Optimization Method for getting gradients ###
import torch as tc
import local_energy as eloc
import time
#### 1. GD method:

class OptiGD:
    def __init__(self, dtype=tc.complex64, device='cpu'):
        self.dtype = dtype
        self.device = device
    def __call__(self, neural_net, sample, E_loc):
        #E_loc_avg = tc.sum(E_loc)/sample.n_states
        E_loc_avg = E_loc.mean()
        '''
        the result from backward is actually conjugate of gradient, 
        so the value O_k^* (conjugate of O_k) is the result of backward directly,
        thus no need to transfer the result to the conjugate expression.
        '''
        # old method:
        # psi = neural_net(sample.states)[:,0]
        # psi.backward(E_loc/sample.n_states, retain_graph=True)
        # #print('1st computing the gradient: \n', fnn.linear_stack[0].weight.grad)
        # # compute -<E_loc>*<conj(O_k)>
        # psi.backward(-1*tc.ones(sample.n_states, dtype=self.dtype, device=self.device) * E_loc_avg/sample.n_states)

        # new method:
        dE = (E_loc - E_loc_avg)##.conj()
        force_E = neural_net(sample.states)[:,0]
        force_E.backward(dE / sample.n_states)
        return E_loc_avg
'''
For any activation fucntions
'''
class OptiGDAny:
    def __init__(self, dtype=tc.complex64, device='cpu'):
        self.dtype = dtype
        self.device = device
    def __call__(self, neural_net, sample, E_loc):
        #E_loc_avg = tc.sum(E_loc)/sample.n_states
        E_loc_avg = E_loc.mean()
        '''
        the result from backward is actually conjugate of gradient, 
        so the value O_k^* (conjugate of O_k) is the result of backward directly,
        thus no need to transfer the result to the conjugate expression.
        '''
        # old method:
        # psi = neural_net(sample.states)[:,0]
        # psi.backward(E_loc/sample.n_states, retain_graph=True)
        # #print('1st computing the gradient: \n', fnn.linear_stack[0].weight.grad)
        # # compute -<E_loc>*<conj(O_k)>
        # psi.backward(-1*tc.ones(sample.n_states, dtype=self.dtype, device=self.device) * E_loc_avg/sample.n_states)

        # new method:
        dE = (E_loc - E_loc_avg)##.conj()
        force_E = neural_net(sample.states)[:,0]
        force_E.backward(dE / sample.n_states)
        
        force_E2 = neural_net(sample.states)[:,0].conj()
        force_E2.backward(dE.conj() / sample.n_states)

        return E_loc_avg

'''
Here, we use exact sampling to get the gradients
'''
class OptiGDExact:
    def __init__(self, dtype=tc.complex64, device='cpu'):
        self.dtype = dtype
        self.device = device
    def __call__(self, neural_net, sample, E_loc):
        print('we are using exact sampling with GD!')
        #E_loc_avg = tc.sum(E_loc)/sample.n_states
        exact_ln_psi = neural_net(sample.states)[:,0]
        #exact_ln_ratio = tc.exp(exact_ln_psi - exact_ln_psi[0]).detach()
        # the amplitude of a given configuration
        #P_x = abs(exact_ln_ratio)**2 / tc.sum(abs(exact_ln_ratio)**2).detach()
        P_x = (abs(tc.exp(exact_ln_psi))**2 / tc.sum(abs(tc.exp(exact_ln_psi))**2)).detach()
        E_loc_avg = tc.sum(E_loc * P_x)
        '''
        the result from backward is actually conjugate of gradient, 
        so the value O_k^* (conjugate of O_k) is the result of backward directly,
        thus no need to transfer the result to the conjugate expression.
        '''
        # old method:
        # psi = neural_net(sample.states)[:,0]
        # psi.backward(E_loc/sample.n_states, retain_graph=True)
        # #print('1st computing the gradient: \n', fnn.linear_stack[0].weight.grad)
        # # compute -<E_loc>*<conj(O_k)>
        # psi.backward(-1*tc.ones(sample.n_states, dtype=self.dtype, device=self.device) * E_loc_avg/sample.n_states)

        # new method:
        dE = (E_loc - E_loc_avg) * P_x ##.conj()
        # force_E = neural_net(sample.states)[:,0]
        # force_E.backward(dE / sample.n_states)
        exact_ln_psi.backward(dE)  
        return E_loc_avg  

'''
Here, we can apply any non-homophic(non-analytical) activation fucntions to do. The gradient should have 4 terms.
'''

class OptiGDExactAny:
    def __init__(self, dtype=tc.complex64, device='cpu'):
        self.dtype = dtype
        self.device = device
    def __call__(self, neural_net, sample, E_loc):
        print(' GD for any!')
        #E_loc_avg = tc.sum(E_loc)/sample.n_states
        exact_ln_psi = neural_net(sample.states)[:,0]
        #exact_ln_ratio = tc.exp(exact_ln_psi - exact_ln_psi[0]).detach()
        # the amplitude of a given configuration
        #P_x = abs(exact_ln_ratio)**2 / tc.sum(abs(exact_ln_ratio)**2).detach()
        P_x = (abs(tc.exp(exact_ln_psi))**2 / tc.sum(abs(tc.exp(exact_ln_psi))**2)).detach()
        E_loc_avg = tc.sum(E_loc * P_x)
        '''
        the result from backward is actually conjugate of gradient, 
        so the value O_k^* (conjugate of O_k) is the result of backward directly,
        thus no need to transfer the result to the conjugate expression.
        '''
        # old method:
        # psi = neural_net(sample.states)[:,0]
        # psi.backward(E_loc/sample.n_states, retain_graph=True)
        # #print('1st computing the gradient: \n', fnn.linear_stack[0].weight.grad)
        # # compute -<E_loc>*<conj(O_k)>
        # psi.backward(-1*tc.ones(sample.n_states, dtype=self.dtype, device=self.device) * E_loc_avg/sample.n_states)

        # new method:
        dE = (E_loc - E_loc_avg) * P_x ##.conj()
        # force_E = neural_net(sample.states)[:,0]
        # force_E.backward(dE / sample.n_states)
        exact_ln_psi.backward(dE)
        
        # 3-rd and 4-th term:
        exact_ln_psi2 = neural_net(sample.states)[:,0].conj()
        dE = tc.conj(E_loc - E_loc_avg) * P_x
        exact_ln_psi2.backward(dE)


        return E_loc_avg 


'''
Here, we pick up the unique samples
'''
class OptiGDUnique:
    def __init__(self, dtype=tc.complex64, device='cpu', n_procs=None, pool=None):
        self.dtype = dtype
        self.device = device
        self.n_procs = n_procs
        self.pool = pool
    def __call__(self, ising, neural_net, sample):
        print('we are using unique sampling with GD!')
        #E_loc_avg = tc.sum(E_loc)/sample.n_states
        states_unique = tc.unique(sample.states.to(int),dim=0)
        n_states_uni = states_unique.size(0)
        print('size of unique samples =', n_states_uni)
        print('size of samples =', sample.states.size(0))
        states_unique = states_unique.to(self.dtype)
        exact_ln_psi = neural_net(states_unique)[:,0]
        exact_ln_ratio = tc.exp(exact_ln_psi - exact_ln_psi[0]).detach()
        # the amplitude of a given configuration
        P_x = abs(exact_ln_ratio)**2 / tc.sum(abs(exact_ln_ratio)**2)
        E_loc = eloc.local_energy_unique(ising, neural_net, states_unique)
        E_loc_avg = tc.sum(E_loc * P_x)
        '''
        the result from backward is actually conjugate of gradient, 
        so the value O_k^* (conjugate of O_k) is the result of backward directly,
        thus no need to transfer the result to the conjugate expression.
        '''
        # old method:
        # psi = neural_net(sample.states)[:,0]
        # psi.backward(E_loc/sample.n_states, retain_graph=True)
        # #print('1st computing the gradient: \n', fnn.linear_stack[0].weight.grad)
        # # compute -<E_loc>*<conj(O_k)>
        # psi.backward(-1*tc.ones(sample.n_states, dtype=self.dtype, device=self.device) * E_loc_avg/sample.n_states)

        # new method:
        dE = (E_loc - E_loc_avg) * P_x ##.conj()
        # force_E = neural_net(sample.states)[:,0]
        # force_E.backward(dE / sample.n_states)
        exact_ln_psi.backward(dE)  
        return E_loc_avg  

class OptiGDRealValued:
    def __init__(self, dtype=tc.complex64, device='cpu'):
        self.dtype = dtype
        self.device = device
    def __call__(self, neural_net_am, neural_net_ph, sample, E_loc):
        #E_loc_avg = tc.sum(E_loc)/sample.n_states
        E_loc_avg = E_loc.mean()
        '''
        the result from backward is actually conjugate of gradient, 
        so the value O_k^* (conjugate of O_k) is the result of backward directly,
        thus no need to transfer the result to the conjugate expression.
        '''
        # old method:
        # psi = neural_net(sample.states)[:,0]
        # psi.backward(E_loc/sample.n_states, retain_graph=True)
        # #print('1st computing the gradient: \n', fnn.linear_stack[0].weight.grad)
        # # compute -<E_loc>*<conj(O_k)>
        # psi.backward(-1*tc.ones(sample.n_states, dtype=self.dtype, device=self.device) * E_loc_avg/sample.n_states)

        # new method:
        dE = 2*(E_loc - E_loc_avg)##.conj()
        force_E_am = neural_net_am(sample.states)[:,0].abs() + 0j
        force_E_ph = 1j* neural_net_ph(sample.states)[:,0]
        ## take real number only:
        force_E_am.backward(dE / sample.n_states)
        force_E_ph.backward(dE / sample.n_states)
        return E_loc_avg

#### 2. SR method:
class OptiSR:
    def __init__(self, neural_net, sr_lambda, n_batches=1, dtype=tc.complex64, device='cpu'):
        #layer_size = np.array(layer_size)
        #n_params = sum(layer_size[0:len(layer_size)-1] * layer_size[1:len(layer_size)]) + sum(layer_size[1:len(layer_size)])
        n_params = neural_net.params_info[-1][1][1]
        #self.p_theta = tc.zeros(n_params, n_states, dtype=dtype, device=device).share_memory_()
        self.regularization = sr_lambda * tc.eye(n_params, dtype=dtype, device=device)
        self.n_batches = n_batches
    def __call__(self, neural_net, sample, E_loc):
        print('SR method')
        E_loc_avg = E_loc.mean()
        '''
        For large ssystem, becasue of the limitation of memeory, 
        we cannot use vap to get the derivatives for all samples,
        thus we need get derivatives of a batch of samples sequentially.
        '''
        n_states_batch = int(sample.n_states / self.n_batches)
        p_theta = []
        for k in range(self.n_batches):
            p_theta.append(get_gradients_vmap(neural_net, sample.states[n_states_batch * k : n_states_batch * (k+1), : ]))
        p_theta = tc.cat(p_theta , 1)    
        S1 = tc.matmul(p_theta, p_theta.t().conj())/sample.n_states
        p_theta_avg = p_theta.sum(dim=1)/sample.n_states
        S2 = tc.matmul(p_theta_avg.reshape(-1,1), p_theta_avg.conj().reshape(1,-1))
        #print('size of S1 = ',S1.size())
        S_matrix = S1 - S2 + self.regularization
        #print(S_matrix.size())
        F = (E_loc * p_theta).sum(dim=1)/sample.n_states - E_loc.sum() * p_theta_avg /sample.n_states
        p_params = tc.linalg.solve(S_matrix,F)
        c = 0
        for grad_n_k in neural_net.parameters():
            grad_n_k.grad = p_params[neural_net.params_info[c][1][0]:neural_net.params_info[c][1][1]].reshape(neural_net.params_info[c][0]) 
            c += 1
        return E_loc_avg

class OptiSRExact:
    def __init__(self, neural_net, sr_lambda, n_batches=1, dtype=tc.complex64, device='cpu'):
        #layer_size = np.array(layer_size)
        #n_params = sum(layer_size[0:len(layer_size)-1] * layer_size[1:len(layer_size)]) + sum(layer_size[1:len(layer_size)])
        n_params = neural_net.params_info[-1][1][1]
        #self.p_theta = tc.zeros(n_params, n_states, dtype=dtype, device=device).share_memory_()
        self.regularization = sr_lambda * tc.eye(n_params, dtype=dtype, device=device)
        self.n_batches = n_batches
    def __call__(self, neural_net, sample, E_loc):
        print('we are using exact sampling with SR!')

        '''
        For large ssystem, becasue of the limitation of memeory, 
        we cannot use vap to get the derivatives for all samples,
        thus we need get derivatives of a batch of samples sequentially.
        '''
        n_states_batch = int(sample.n_states / self.n_batches)
        p_theta = []
        for k in range(self.n_batches):
            p_theta.append(get_gradients_vmap(neural_net, sample.states[n_states_batch * k : n_states_batch * (k+1), : ]))
        p_theta = tc.cat(p_theta , 1)   
        print('the size of p_theta = ',p_theta.size()) 
        #p_theta = get_gradients_vmap(neural_net, sample.states)
        exact_ln_psi = neural_net(sample.states)[:,0].detach()
        exact_ln_ratio = tc.exp(exact_ln_psi - exact_ln_psi[0])
        # the amplitude of a given configuration
        P_x = abs(exact_ln_ratio)**2 / tc.sum(abs(exact_ln_ratio)**2)
        S1 = tc.matmul(p_theta * P_x, p_theta.t().conj())
        p_theta_avg = (p_theta * P_x).sum(dim=1)
        S2 = tc.matmul(p_theta_avg.reshape(-1,1), p_theta_avg.conj().reshape(1,-1))
        #print('size of S1 = ',S1.size())
        S_matrix = S1 - S2 + self.regularization
        #print(S_matrix.size())
        E_loc_avg = tc.sum(E_loc * P_x)
        F = (E_loc * p_theta * P_x).sum(dim=1) - E_loc_avg * p_theta_avg
        p_params = tc.linalg.solve(S_matrix,F)
        c = 0
        for grad_n_k in neural_net.parameters():
            grad_n_k.grad = p_params[neural_net.params_info[c][1][0]:neural_net.params_info[c][1][1]].reshape(neural_net.params_info[c][0]) 
            c += 1
        return E_loc_avg
    
class OptiSRExact2:
    def __init__(self, neural_net, sr_lambda, n_batches=1, dtype=tc.complex64, device='cpu'):
        #layer_size = np.array(layer_size)
        #n_params = sum(layer_size[0:len(layer_size)-1] * layer_size[1:len(layer_size)]) + sum(layer_size[1:len(layer_size)])
        n_params = sum(p.numel() for p in neural_net.parameters() if p.requires_grad)
        #self.p_theta = tc.zeros(n_params, n_states, dtype=dtype, device=device).share_memory_()
        self.regularization = sr_lambda * tc.eye(n_params, dtype=dtype, device=device)
        self.n_batches = n_batches
    def __call__(self, neural_net, sample, E_loc):
        print('we are using exact sampling with SR(new code)!')

        '''
        For large ssystem, becasue of the limitation of memeory, 
        we cannot use vap to get the derivatives for all samples,
        thus we need get derivatives of a batch of samples sequentially.
        '''
        n_states_batch = int(sample.n_states / self.n_batches)
        p_theta = []
        p_theta_conj = []
        for k in range(self.n_batches):
            p_theta.append(get_gradients_vmap(neural_net, sample.states[n_states_batch * k : n_states_batch * (k+1), : ]))
            p_theta_conj.append(get_gradients_vmap_conj(neural_net, sample.states[n_states_batch * k : n_states_batch * (k+1), : ]))
        p_theta = tc.cat(p_theta , 1)   
        p_theta_conj = tc.cat(p_theta_conj , 1)   
        #print('the size of p_theta = ',p_theta.size())
        #print('the size of p_theta_conj = ',p_theta_conj.size())
        #p_theta = get_gradients_vmap(neural_net, sample.states)
        exact_ln_psi = neural_net(sample.states)[:,0].detach()
        exact_ln_ratio = tc.exp(exact_ln_psi - exact_ln_psi[0])
        # the amplitude of a given configuration
        P_x = abs(exact_ln_ratio)**2 / tc.sum(abs(exact_ln_ratio)**2)

        p_theta_avg = (p_theta * P_x).sum(dim=1)
        p_theta_conj_avg = (p_theta_conj * P_x).sum(dim=1)

        S1 = tc.matmul(p_theta * P_x, p_theta.t().conj())
        S2 = tc.matmul(p_theta_avg.reshape(-1,1), p_theta_avg.conj().reshape(1,-1))

        S1_conj = tc.matmul(p_theta_conj * P_x, p_theta_conj.t().conj())
        S2_conj = tc.matmul(p_theta_conj_avg.reshape(-1,1), p_theta_conj_avg.conj().reshape(1,-1))
        
        

        #print('size of S1 = ',S1.size())
        S_matrix = S1 - S2 + self.regularization
        S_matrix = 0.5*(S1 - S2 + S1_conj - S2_conj) + self.regularization

        #print(S_matrix.size())
        E_loc_avg = tc.sum(E_loc * P_x)
        F1 = (E_loc * p_theta * P_x).sum(dim=1) - E_loc_avg * p_theta_avg
        F2 = (E_loc.conj() * p_theta_conj * P_x).sum(dim=1) - E_loc_avg.conj() * p_theta_conj_avg
        F = 0.5*(F1 + F2)
        p_params = tc.linalg.solve(S_matrix,F)

        pointer = 0
        for param in neural_net.parameters():
            if param.requires_grad:
                numel = param.numel()
                grad_segment = p_params[pointer:pointer+numel].view_as(param)
                param.grad = grad_segment.clone()
                pointer += numel
        return E_loc_avg

'''
Here, we use vmap to get gradients of all samples of each processors
'''
def get_gradients_vmap(neural_net, states):
    #print('working!')
    n_states = states.size()[0]
    I_N = tc.eye(n_states, dtype=neural_net.dtype, device=neural_net.device)
    z = neural_net(states)
    # generate a function that can be vectoried:
    f = lambda v : tc.autograd.grad(z[:,0], neural_net.parameters(),v)
    grad_vmap = tc.vmap(f)(I_N)
    z = []
    '''
    The gradient for each set of weight W or bias b is [n_states, n, m], where the size of W or b is [n,m], so we need to flatten it as a vector with size of [n_states, n * m] and do it for all weights and bias and together them as a matrix of gradient with size of [n_state, n_p] where n_p is the number of parameters, when return the conjugate part with the size of [n_p, n_states].
    '''

    grad_procs_i = tc.cat([tc.flatten(grad_k, 1) for grad_k in grad_vmap], 1)
    return tc.t(grad_procs_i)

def get_gradients_vmap_conj(neural_net, states):
    #print('working!')
    n_states = states.size()[0]
    I_N = tc.eye(n_states, dtype=neural_net.dtype, device=neural_net.device)
    z = neural_net(states).conj()
    # generate a function that can be vectoried:
    f = lambda v : tc.autograd.grad(z[:,0], neural_net.parameters(),v)
    grad_vmap = tc.vmap(f)(I_N)
    z = []
    '''
    The gradient for each set of weight W or bias b is [n_states, n, m], where the size of W or b is [n,m], so we need to flatten it as a vector with size of [n_states, n * m] and do it for all weights and bias and together them as a matrix of gradient with size of [n_state, n_p] where n_p is the number of parameters, when return the conjugate part with the size of [n_p, n_states].
    '''

    grad_procs_i = tc.cat([tc.flatten(grad_k, 1) for grad_k in grad_vmap], 1)
    return tc.t(grad_procs_i)

###########################################
###########################################
## evolution part ##

class OptiGDTime:
    def __init__(self, dtype=tc.complex64, device='cpu', n_procs=None, pool=None):
        self.dtype = dtype
        self.device = device
        self.n_procs = n_procs
        self.pool = pool
    def __call__(self, neural_net, sample, E_loc_psi_phi, E_loc_phi_psi):
        E_loc_psi_phi_avg = tc.sum(E_loc_psi_phi)/sample.n_states
        E_loc_phi_psi_avg = tc.sum(E_loc_phi_psi)/sample.n_states
        dE1 = E_loc_phi_psi_avg * E_loc_psi_phi
        dE2 = E_loc_phi_psi_avg * E_loc_psi_phi_avg
        dE = (dE2 - dE1).detach()
        #print('E_loc_phi_psi_avg = ',E_loc_phi_psi_avg)
        #print('dE2 = ', dE2)
        #print('size ===> ', dE1.size())
        psi = neural_net(sample.states)[:,0]
        '''
        the result from backward is actually conjugate of gradient, 
        so the value O_k^* (conjugate of O_k) is the result of backward directly,
        thus no need to transfer the result to the conjugate expression.
        '''
        # psi.backward(-1 * dE1/sample.n_states, retain_graph=True)
        # #print('1st computing the gradient: \n', fnn.linear_stack[0].weight.grad)
        # # compute -<E_loc>*<conj(O_k)>
        # psi.backward(tc.ones(sample.n_states, dtype=self.dtype, device=self.device) * dE2/sample.n_states)
        psi.backward(dE)
        return (1 - dE2)


class OptiGDExactTime:
    def __init__(self, dtype=tc.complex64, device='cpu', n_procs=None, pool=None):
        self.dtype = dtype
        self.device = device
        self.n_procs = n_procs
        self.pool = pool
    def __call__(self, neural_net0, neural_net, sample, E_loc_psi_phi, E_loc_phi_psi):
        exact_ln_phi = neural_net(sample.states)[:,0]
        exact_ln_psi = neural_net0(sample.states)[:,0]
        P_phi = (abs(tc.exp(exact_ln_phi))**2 / tc.sum(abs(tc.exp(exact_ln_phi))**2)).detach()
        P_psi = (abs(tc.exp(exact_ln_psi))**2 / tc.sum(abs(tc.exp(exact_ln_psi))**2)).detach()
        E_loc_psi_phi_avg = tc.sum(E_loc_psi_phi * P_phi)
        E_loc_phi_psi_avg = tc.sum(E_loc_phi_psi * P_psi)
        dE1 = E_loc_phi_psi_avg * E_loc_psi_phi
        dE2 = E_loc_phi_psi_avg * E_loc_psi_phi_avg
        dE = (dE2 - dE1) * P_phi
        #dE_cv = dE - tc.ones(sample.n_states, dtype=neural_net.dtype, device=neural_net.device)
        '''
        the result from backward is actually conjugate of gradient, 
        so the value O_k^* (conjugate of O_k) is the result of backward directly,
        thus no need to transfer the result to the conjugate expression.
        '''
        #print('E_loc_phi_psi_avg = ',E_loc_phi_psi_avg)
        #print('dE2 = ', dE2)
        #print('size ===> ', dE1.size())
        #force_E = neural_net(sample.states)[:,0]
        #force_E.backward(dE / sample.n_states)

        exact_ln_phi.backward(dE)

        # with control variate method:
        #force_E.backward(dE_cv / sample.n_states)

        # psi = neural_net(sample.states)[:,0]
        
        # psi.backward(-1 * dE1/sample.n_states, retain_graph=True)
        # #print('1st computing the gradient: \n', fnn.linear_stack[0].weight.grad)
        # # compute -<E_loc>*<conj(O_k)>
        # psi.backward(tc.ones(sample.n_states, dtype=self.dtype, device=self.device) * dE2/sample.n_states)
        #print(f'E_loc_phi_psi_avg = {E_loc_phi_psi_avg:.10f}')
        #print(f'E_loc_psi_phi_avg = {E_loc_psi_phi_avg:.10f}')
        return (1 - dE2)
    
class OptiGDExactTimeTry:
    def __init__(self, dtype=tc.complex64, device='cpu', n_procs=None, pool=None):
        self.dtype = dtype
        self.device = device
        self.n_procs = n_procs
        self.pool = pool
    def __call__(self, neural_net0, neural_nett, neural_net2t, sample, E_loc_01, E_loc_10, E_loc_12, E_loc_21):
        exact_ln_0 = neural_net0(sample.states)[:,0]
        exact_ln_t = neural_nett(sample.states)[:,0]
        exact_ln_2t= neural_net2t(sample.states)[:,0]
        P_t_0 = (abs(tc.exp(exact_ln_0))**2 / tc.sum(abs(tc.exp(exact_ln_0))**2)).detach()
        P_t_1 = (abs(tc.exp(exact_ln_t))**2 / tc.sum(abs(tc.exp(exact_ln_t))**2)).detach()
        P_t_2 = (abs(tc.exp(exact_ln_2t))**2 / tc.sum(abs(tc.exp(exact_ln_2t))**2)).detach()
        E_loc_01_avg = tc.sum(E_loc_01 * P_t_1)
        E_loc_10_avg = tc.sum(E_loc_10 * P_t_0)
        E_loc_12_avg = tc.sum(E_loc_12 * P_t_2)
        E_loc_21_avg = tc.sum(E_loc_21 * P_t_1)
        I_t_1 = E_loc_01_avg * E_loc_10_avg
        I_t_2 = E_loc_12_avg * E_loc_21_avg

        # get derivatives of nn_t1:
        dE1 = E_loc_12_avg * I_t_1 * (E_loc_21 - E_loc_21_avg) + E_loc_10_avg * I_t_2 * (E_loc_01  - E_loc_01_avg)
        dE1 = -dE1 * P_t_1
        exact_ln_t.backward(dE1)

        # get derivatives of nn_t2:
        dE2 = E_loc_21_avg * I_t_1 *(E_loc_12 - E_loc_12_avg)
        dE2 = -dE2 * P_t_2
        exact_ln_2t.backward(dE2)



        #dE_cv = dE - tc.ones(sample.n_states, dtype=neural_net.dtype, device=neural_net.device)
        '''
        the result from backward is actually conjugate of gradient, 
        so the value O_k^* (conjugate of O_k) is the result of backward directly,
        thus no need to transfer the result to the conjugate expression.
        '''
        #print('E_loc_phi_psi_avg = ',E_loc_phi_psi_avg)
        #print('dE2 = ', dE2)
        #print('size ===> ', dE1.size())
        #force_E = neural_net(sample.states)[:,0]
        #force_E.backward(dE / sample.n_states)


        # with control variate method:
        #force_E.backward(dE_cv / sample.n_states)

        # psi = neural_net(sample.states)[:,0]
        
        # psi.backward(-1 * dE1/sample.n_states, retain_graph=True)
        # #print('1st computing the gradient: \n', fnn.linear_stack[0].weight.grad)
        # # compute -<E_loc>*<conj(O_k)>
        # psi.backward(tc.ones(sample.n_states, dtype=self.dtype, device=self.device) * dE2/sample.n_states)
        print(f'E_loc_1 = {E_loc_01_avg * E_loc_10_avg:.10f}')
        print(f'E_loc_2 = {E_loc_12_avg * E_loc_21_avg:.10f}')
        return (1 - E_loc_01_avg * E_loc_10_avg * E_loc_12_avg * E_loc_21_avg)
    
class OptiGDExactTimeTry2:
    def __init__(self, dtype=tc.complex64, device='cpu', n_procs=None, pool=None):
        self.dtype = dtype
        self.device = device
        self.n_procs = n_procs
        self.pool = pool
    def __call__(self, neural_net_list, sample, E_loc_r_list, E_loc_l_list):
        n_neural_nets = len(neural_net_list)
        print('n_neural_nets = ',n_neural_nets)
        P_t = []

        E_loc_r_avg = tc.zeros(n_neural_nets - 1, dtype=neural_net_list[1].dtype,device=neural_net_list[1].device)
        E_loc_l_avg = tc.zeros(n_neural_nets - 1, dtype=neural_net_list[1].dtype,device=neural_net_list[1].device)
        for s in range(n_neural_nets - 1):  
            #print('s = ',s)
            if s == 0:
                exact_s = tc.exp(neural_net_list[s+1](sample.states)[:,0].detach())
                P_t.append(abs(exact_s)**2 / tc.sum(abs(exact_s)**2))
            exact_s = tc.exp(neural_net_list[s+1](sample.states)[:,0].detach())
            P_t.append(abs(exact_s)**2 / tc.sum(abs(exact_s)**2))
            E_loc_r_avg[s]=tc.sum(E_loc_r_list[s] * P_t[s+1])
            E_loc_l_avg[s]=tc.sum(E_loc_l_list[s] * P_t[s])
        
        for s in range(1, n_neural_nets - 1):
            #print('s = ', s)
            exact_ln_s= neural_net_list[s](sample.states)[:,0]
            dE = 1.0 + 0.0j
            for k in range(1, n_neural_nets):
                if (k != s) | (k != s+1):
                    dE = dE * E_loc_r_avg[k-1] * E_loc_l_avg[k-1]
            I_s1 = E_loc_r_avg[s-1] * E_loc_l_avg[s-1]
            I_s2 = E_loc_r_avg[s] * E_loc_l_avg[s]
            # dE1 = E_loc_12_avg * I_t_1 * (E_loc_21 - E_loc_21_avg) + E_loc_10_avg * I_t_2 * (E_loc_01  - E_loc_01_avg)
            # dE1 = -dE1 * P_t_1
            dE1 = I_s2 * E_loc_l_avg[s-1] * (E_loc_r_list[s-1] - E_loc_r_avg[s-1])
            dE1 += I_s1 * E_loc_r_avg[s] * (E_loc_l_list[s] - E_loc_l_avg[s])
            dE = -1*(dE * dE1) * P_t[s]
            exact_ln_s.backward(dE)
        
        # compute the last one:
        exact_ln_s= neural_net_list[-1](sample.states)[:,0]
        dE = 1.0 + 0.0j
        for k in range(1, n_neural_nets):
            if (k != n_neural_nets - 1):
                dE = dE * E_loc_r_avg[k-1] * E_loc_l_avg[k-1]
            else:
                #print('k=',k)
                dE = dE*(E_loc_l_avg[k-1]) * (E_loc_r_list[k-1] - E_loc_r_avg[k-1])
        dE = -dE * P_t[-1]
        exact_ln_s.backward(dE)

        return (1 - tc.prod(E_loc_l_avg * E_loc_r_avg))

'''
smooth neural network
'''
class OptiGDExactTimeTrySmooth:
    def __init__(self, dtype=tc.complex64, device='cpu', n_procs=None, pool=None):
        self.dtype = dtype
        self.device = device
        self.n_procs = n_procs
        self.pool = pool
    def __call__(self, evo_t, dt, n_q, neural_net_list, neural_net_base, sample, E_loc_r_list, E_loc_l_list):
        n_neural_nets = len(neural_net_list)
        #print('n_neural_nets = ',n_neural_nets)
        P_t = []

        E_loc_r_avg = tc.zeros(n_neural_nets - 1, dtype=neural_net_list[1].dtype,device=neural_net_list[1].device)
        E_loc_l_avg = tc.zeros(n_neural_nets - 1, dtype=neural_net_list[1].dtype,device=neural_net_list[1].device)
        for s in range(n_neural_nets - 1):  
            #print('s = ',s)
            if s == 0:
                exact_s = tc.exp(neural_net_list[s](sample.states)[:,0].detach())
                P_t.append(abs(exact_s)**2 / tc.sum(abs(exact_s)**2))
            exact_s = tc.exp(neural_net_list[s+1](sample.states)[:,0].detach())
            P_t.append(abs(exact_s)**2 / tc.sum(abs(exact_s)**2))
            E_loc_r_avg[s]=tc.sum(E_loc_r_list[s] * P_t[s+1])
            E_loc_l_avg[s]=tc.sum(E_loc_l_list[s] * P_t[s])
        
        ## get derivatives:
        for q in range(n_q): # each parameters \theta_q:
            for param_k in neural_net_base.param_space[q].parameters():
                param_k.grad = tc.zeros(param_k.size(), dtype=neural_net_list[1].dtype,device=neural_net_list[1].device)


        for q in range(n_q): # each parameters \theta_q:
            #print('q = ',q)
            for s in range(1, n_neural_nets - 1): # each s, given q
                #print('sss')
                #print('s = ', s)
                exact_ln_s= neural_net_list[s](sample.states)[:,0]
                dE = 1.0 + 0.0j
                for k in range(1, n_neural_nets):
                    if (k != s) | (k != s+1):
                        dE = dE * E_loc_r_avg[k-1] * E_loc_l_avg[k-1]
                I_s1 = E_loc_r_avg[s-1] * E_loc_l_avg[s-1]
                I_s2 = E_loc_r_avg[s] * E_loc_l_avg[s]
                # dE1 = E_loc_12_avg * I_t_1 * (E_loc_21 - E_loc_21_avg) + E_loc_10_avg * I_t_2 * (E_loc_01  - E_loc_01_avg)
                # dE1 = -dE1 * P_t_1
                dE1 = I_s2 * E_loc_l_avg[s-1] * (E_loc_r_list[s-1] - E_loc_r_avg[s-1])
                dE1 += I_s1 * E_loc_r_avg[s] * (E_loc_l_list[s] - E_loc_l_avg[s])
                dE = -1*(dE * dE1) * P_t[s] * ((evo_t + s * dt)**q)
                #exact_ln_s.backward(dE)
                grad_s = tc.autograd.grad(exact_ln_s, neural_net_list[s].parameters(), grad_outputs=dE)
                c=0
                for param_k in neural_net_base.param_space[q].parameters():
                    param_k.grad +=  tc.clone(grad_s[c])#.detach()
                    c+=1
            # compute the last one:
            exact_ln_s= neural_net_list[-1](sample.states)[:,0]
            dE = 1.0 + 0.0j
            for k in range(1, n_neural_nets):
                if (k != n_neural_nets - 1):
                    dE = dE * E_loc_r_avg[k-1] * E_loc_l_avg[k-1]
                else:
                    #print('k=',k)
                    dE = dE*(E_loc_l_avg[k-1]) * (E_loc_r_list[k-1] - E_loc_r_avg[k-1])
            dE = -dE * P_t[-1] * ((evo_t + (n_neural_nets - 1) * dt)**q)
            #exact_ln_s.backward(dE)
            grad_s = tc.autograd.grad(exact_ln_s, neural_net_list[-1].parameters(), grad_outputs=dE)

            # exact_ln_s= neural_net_list[1](sample.states)[:,0]
            # dE = 1.0 + 0.0j
            # for k in range(1, n_neural_nets):
            #     if (k != n_neural_nets - 1):
            #         dE = dE * E_loc_r_avg[k-1] * E_loc_l_avg[k-1]
            #     else:
            #         #print('k=',k)
            #         dE = dE*(E_loc_l_avg[k-1]) * (E_loc_r_list[k-1] - E_loc_r_avg[k-1])
            # dE = -dE * P_t[1] * (((n_neural_nets - 1) * dt)**q)
            # #exact_ln_s.backward(dE)
            # grad_s = tc.autograd.grad(exact_ln_s, neural_net_list[1].parameters(), grad_outputs=dE)
            c=0
            for param_k in neural_net_base.param_space[q].parameters():
                param_k.grad +=  tc.clone(grad_s[c])#.detach()
                c+=1

            # c=0    
            # for param_k in neural_net_list[1].parameters():
            #     param_k.grad =  grad_s[c]
            #     c+=1

        #return (1 - tc.prod(E_loc_l_avg * E_loc_r_avg))
        return 1 - E_loc_l_avg * E_loc_r_avg, (1 - tc.prod(E_loc_l_avg * E_loc_r_avg))
        #return tc.log(tc.prod(E_loc_l_avg * E_loc_r_avg))

'''
Here, the loss function is: loss = log(overlap)
'''
class OptiGDExactTimeTrySmooth2:
    def __init__(self, dtype=tc.complex64, device='cpu', n_procs=None, pool=None):
        self.dtype = dtype
        self.device = device
        self.n_procs = n_procs
        self.pool = pool
    def __call__(self, dt, n_q, neural_net_list, neural_net_base, sample, E_loc_r_list, E_loc_l_list):
        n_neural_nets = len(neural_net_list)
        #print('n_neural_nets = ',n_neural_nets)
        P_t = []

        E_loc_r_avg = tc.zeros(n_neural_nets - 1, dtype=neural_net_list[1].dtype,device=neural_net_list[1].device)
        E_loc_l_avg = tc.zeros(n_neural_nets - 1, dtype=neural_net_list[1].dtype,device=neural_net_list[1].device)
        for s in range(n_neural_nets - 1):  
            #print('s = ',s)
            if s == 0:
                exact_s = tc.exp(neural_net_list[s](sample.states)[:,0].detach())
                P_t.append(abs(exact_s)**2 / tc.sum(abs(exact_s)**2))
            exact_s = tc.exp(neural_net_list[s+1](sample.states)[:,0].detach())
            P_t.append(abs(exact_s)**2 / tc.sum(abs(exact_s)**2))
            E_loc_r_avg[s]=tc.sum(E_loc_r_list[s] * P_t[s+1])
            E_loc_l_avg[s]=tc.sum(E_loc_l_list[s] * P_t[s])
        
        ## get derivatives:
        for q in range(n_q): # each parameters \theta_q:
            for param_k in neural_net_base.param_space[q].parameters():
                param_k.grad = tc.zeros(param_k.size(), dtype=neural_net_list[1].dtype,device=neural_net_list[1].device)


        for q in range(n_q): # each parameters \theta_q:
            #print('q = ',q)
            for s in range(1, n_neural_nets - 1): # each s, given q
                #print('sss')
                #print('s = ', s)
                exact_ln_s= neural_net_list[s](sample.states)[:,0]
                dE1 = E_loc_l_avg[s-1] * (E_loc_r_list[s-1] - E_loc_r_avg[s-1]) / (E_loc_l_avg[s-1] * E_loc_r_avg[s-1])
                dE2 = E_loc_r_avg[s] * (E_loc_l_list[s] - E_loc_l_avg[s]) / (E_loc_l_avg[s] * E_loc_r_avg[s])
                dE = -1*(dE1 + dE2) * P_t[s] * ((s * dt)**q)
                #exact_ln_s.backward(dE)
                grad_s = tc.autograd.grad(exact_ln_s, neural_net_list[s].parameters(), grad_outputs=dE)
                c=0
                for param_k in neural_net_base.param_space[q].parameters():
                    param_k.grad +=  grad_s[c]#.detach()
                    c+=1
            # compute the last one:
            exact_ln_s= neural_net_list[-1](sample.states)[:,0]
            dE1 = E_loc_l_avg[-1] * (E_loc_r_list[-1] - E_loc_r_avg[-1]) / (E_loc_l_avg[-1] * E_loc_r_avg[-1])
            dE = -dE1 * P_t[-1] * (((n_neural_nets - 1) * dt)**q)
            #exact_ln_s.backward(dE)
            grad_s = tc.autograd.grad(exact_ln_s, neural_net_list[-1].parameters(), grad_outputs=dE)
            c=0
            for param_k in neural_net_base.param_space[q].parameters():
                param_k.grad +=  tc.clone(grad_s[c])#.detach()
                c+=1

            # c=0    
            # for param_k in neural_net_list[1].parameters():
            #     param_k.grad =  grad_s[c]
            #     c+=1

        return (1 - tc.prod(E_loc_l_avg * E_loc_r_avg))
        #return tc.log(tc.prod(E_loc_l_avg * E_loc_r_avg))

class OptiGDTimeTrySmooth:
    def __init__(self, dtype=tc.complex64, device='cpu', n_procs=None, pool=None):
        self.dtype = dtype
        self.device = device
        self.n_procs = n_procs
        self.pool = pool
    def __call__(self, dt, n_q, neural_net_list, neural_net_base, sample_list, E_loc_r_list, E_loc_l_list):
        n_neural_nets = len(neural_net_list)
        #print('n_neural_nets = ',n_neural_nets)
        P_t = []

        E_loc_r_avg = tc.zeros(n_neural_nets - 1, dtype=neural_net_list[1].dtype,device=neural_net_list[1].device)
        E_loc_l_avg = tc.zeros(n_neural_nets - 1, dtype=neural_net_list[1].dtype,device=neural_net_list[1].device)
        for s in range(n_neural_nets - 1):  
            E_loc_r_avg[s]=E_loc_r_list[s].mean()
            E_loc_l_avg[s]=E_loc_l_list[s].mean()
        
        ## get derivatives:
        for q in range(n_q): # each parameters \theta_q:
            for param_k in neural_net_base.param_space[q].parameters():
                param_k.grad = tc.zeros(param_k.size(), dtype=neural_net_list[1].dtype,device=neural_net_list[1].device)


        for q in range(n_q): # each parameters \theta_q:
            #print('q = ',q)
            for s in range(1, n_neural_nets - 1): # each s, given q
                #print('sss')
                #print('s = ', s)
                ln_s= neural_net_list[s](sample_list[s].states)[:,0]
                dE = 1.0 + 0.0j
                for k in range(1, n_neural_nets):
                    if (k != s) | (k != s+1):
                        dE = dE * E_loc_r_avg[k-1] * E_loc_l_avg[k-1]
                I_s1 = E_loc_r_avg[s-1] * E_loc_l_avg[s-1]
                I_s2 = E_loc_r_avg[s] * E_loc_l_avg[s]
                # dE1 = E_loc_12_avg * I_t_1 * (E_loc_21 - E_loc_21_avg) + E_loc_10_avg * I_t_2 * (E_loc_01  - E_loc_01_avg)
                # dE1 = -dE1 * P_t_1
                dE1 = I_s2 * E_loc_l_avg[s-1] * (E_loc_r_list[s-1] - E_loc_r_avg[s-1])
                dE1 += I_s1 * E_loc_r_avg[s] * (E_loc_l_list[s] - E_loc_l_avg[s])
                dE = -1*(dE * dE1) * ((s * dt)**q) / sample_list[s].n_states
                #exact_ln_s.backward(dE)
                grad_s = tc.autograd.grad(ln_s, neural_net_list[s].parameters(), grad_outputs=dE)
                c=0
                for param_k in neural_net_base.param_space[q].parameters():
                    param_k.grad +=  grad_s[c]#.detach()
                    c+=1
            # compute the last one:
            ln_s= neural_net_list[-1](sample_list[-1].states)[:,0]
            dE = 1.0 + 0.0j
            for k in range(1, n_neural_nets):
                if (k != n_neural_nets - 1):
                    dE = dE * E_loc_r_avg[k-1] * E_loc_l_avg[k-1]
                else:
                    #print('k=',k)
                    dE = dE*(E_loc_l_avg[k-1]) * (E_loc_r_list[k-1] - E_loc_r_avg[k-1])
            dE = -dE  * (((n_neural_nets - 1) * dt)**q) / sample_list[-1].n_states
            #exact_ln_s.backward(dE)
            grad_s = tc.autograd.grad(ln_s, neural_net_list[-1].parameters(), grad_outputs=dE)

            # exact_ln_s= neural_net_list[1](sample.states)[:,0]
            # dE = 1.0 + 0.0j
            # for k in range(1, n_neural_nets):
            #     if (k != n_neural_nets - 1):
            #         dE = dE * E_loc_r_avg[k-1] * E_loc_l_avg[k-1]
            #     else:
            #         #print('k=',k)
            #         dE = dE*(E_loc_l_avg[k-1]) * (E_loc_r_list[k-1] - E_loc_r_avg[k-1])
            # dE = -dE * P_t[1] * (((n_neural_nets - 1) * dt)**q)
            # #exact_ln_s.backward(dE)
            # grad_s = tc.autograd.grad(exact_ln_s, neural_net_list[1].parameters(), grad_outputs=dE)
            c=0
            for param_k in neural_net_base.param_space[q].parameters():
                param_k.grad +=  tc.clone(grad_s[c])#.detach()
                c+=1

            # c=0    
            # for param_k in neural_net_list[1].parameters():
            #     param_k.grad =  grad_s[c]
            #     c+=1
        return (1 - tc.prod(E_loc_l_avg * E_loc_r_avg))

class OptiGDTime2:
    def __init__(self, dtype=tc.complex64, device='cpu', n_procs=None, pool=None):
        self.dtype = dtype
        self.device = device
        self.n_procs = n_procs
        self.pool = pool
    def __call__(self, neural_net, sample0, sample, E_loc_psi_phi):
        print('test new loss')
        phi_C = tc.sum(tc.abs(neural_net(sample0.states)[:,0].detach())**2)
        E_loc = (tc.abs(E_loc_psi_phi)**2 / phi_C).to(self.dtype)
        E_loc_avg = tc.sum(E_loc) / sample.n_states
        dE = -2 * tc.real(E_loc - E_loc_avg).to(self.dtype)
        dE = -1* (E_loc - E_loc_avg)
        print(f'E_loc_avg = {E_loc_avg:.15f}')
        # E_loc_psi_phi_avg = tc.sum(E_loc_psi_phi)/sample.n_states
        # E_loc_phi_psi_avg = tc.sum(E_loc_phi_psi)/sample.n_states
        # dE1 = E_loc_phi_psi_avg * E_loc_psi_phi
        # dE2 = E_loc_phi_psi_avg * E_loc_psi_phi_avg
        # dE = dE2 - dE1
        #dE_cv = dE - tc.ones(sample.n_states, dtype=neural_net.dtype, device=neural_net.device)
        '''
        the result from backward is actually conjugate of gradient, 
        so the value O_k^* (conjugate of O_k) is the result of backward directly,
        thus no need to transfer the result to the conjugate expression.
        '''
        #print('E_loc_phi_psi_avg = ',E_loc_phi_psi_avg)
        #print('dE2 = ', dE2)
        #print('size ===> ', dE1.size())
        force_E = neural_net(sample.states)[:,0]
        force_E.backward(dE / sample.n_states)

        # with control variate method:
        #force_E.backward(dE_cv / sample.n_states)

        # psi = neural_net(sample.states)[:,0]
        
        # psi.backward(-1 * dE1/sample.n_states, retain_graph=True)
        # #print('1st computing the gradient: \n', fnn.linear_stack[0].weight.grad)
        # # compute -<E_loc>*<conj(O_k)>
        # psi.backward(tc.ones(sample.n_states, dtype=self.dtype, device=self.device) * dE2/sample.n_states)

class OptiGDCVTime:
    def __init__(self, neural_net, dtype=tc.complex64, device='cpu', n_procs=None, pool=None):
        n_params = neural_net.params_info[-1][1][1]
        #self.p_theta = tc.zeros(n_params, n_states, dtype=dtype, device=device).share_memory_()
        self.n_batches = 10
        self.n_procs = n_procs
        self.pool = pool
        self.c = tc.tensor(n_params, dtype=neural_net.dtype, device=neural_net.device)
    def __call__(self, neural_net, sample, E_loc_psi_phi, E_loc_phi_psi):
        print('GD with CV ')
        E_loc_psi_phi_avg = tc.sum(E_loc_psi_phi)/sample.n_states
        E_loc_phi_psi_avg = tc.sum(E_loc_phi_psi)/sample.n_states
        dE = E_loc_phi_psi_avg * (E_loc_psi_phi - E_loc_psi_phi_avg)
        dE2 = E_loc_phi_psi_avg * E_loc_psi_phi
        n_states_batch = int(sample.n_states / self.n_batches)
        p_theta = []
        for k in range(self.n_batches):
            p_theta.append(get_gradients_vmap(neural_net, sample.states[n_states_batch * k : n_states_batch * (k+1), : ]))
        p_theta = tc.cat(p_theta , 1)    
        F = (dE * p_theta).sum(dim=1) / sample.n_states
        F_length = F.size(0)
        # compute c in control variate:
        # c = -Cov(m,t)/ Var(t)
        c_Cov = (dE * p_theta - F.view(F_length,1)) * (dE2 - 1)
        c_Cov = c_Cov.sum(dim=1) / sample.n_states
        c_Var = ((dE2 - 1).abs())**2
        c_Var = c_Var.sum() / sample.n_states
        CV_c = -c_Cov / c_Var
        print(CV_c.size())
        CV_coff = tc.mm(CV_c.view(F_length,1), (dE2 - 1).view(1,sample.n_states))
        #CV_coff = 0
        print('size of CV_coff= ',CV_coff.size())
        F_new = (dE * p_theta + CV_coff).sum(dim=1) /sample.n_states
        p_params = -F_new
        c = 0
        for grad_n_k in neural_net.parameters():
            grad_n_k.grad = p_params[neural_net.params_info[c][1][0]:neural_net.params_info[c][1][1]].reshape(neural_net.params_info[c][0]) 
            c += 1



class OptiSRTime:
    def __init__(self, neural_net, sr_lambda, n_batches=1, dtype=tc.complex64, device='cpu'):
        n_params = neural_net.params_info[-1][1][1]
        #self.p_theta = tc.zeros(n_params, n_states, dtype=dtype, device=device).share_memory_()
        self.regularization = sr_lambda * tc.eye(n_params, dtype=dtype, device=device)
        self.n_batches = n_batches
    def __call__(self, neural_net, sample, E_loc_psi_phi, E_loc_phi_psi):
        E_loc_psi_phi_avg = tc.sum(E_loc_psi_phi)/sample.n_states
        E_loc_phi_psi_avg = tc.sum(E_loc_phi_psi)/sample.n_states
        dE1 = E_loc_phi_psi_avg * E_loc_psi_phi
        dE2 = E_loc_phi_psi_avg * E_loc_psi_phi_avg
        n_states_batch = int(sample.n_states / self.n_batches)
        p_theta = []
        for k in range(self.n_batches):
            p_theta.append(get_gradients_vmap(neural_net, sample.states[n_states_batch * k : n_states_batch * (k+1), : ]))
        p_theta = tc.cat(p_theta , 1)    
        #p_theta = get_gradients_vmap(neural_net, sample.states)
        #p_theta = get_gradients_vmap(neural_net, sample.states)
        S1 = tc.matmul(p_theta, p_theta.t().conj())/sample.n_states
        p_theta_avg = p_theta.sum(dim=1)/sample.n_states
        S2 = tc.matmul(p_theta_avg.reshape(-1,1), p_theta_avg.conj().reshape(1,-1))
        S_matrix = S1 - S2 + self.regularization

        F1 = (dE1 * p_theta).sum(dim=1) / sample.n_states
        F2 = dE2 * p_theta_avg
        #F = -(F1 - F2)
        F =-(F1 - F2)
        p_params = tc.linalg.solve(S_matrix,F)
        c = 0
        for grad_n_k in neural_net.parameters():
            grad_n_k.grad = p_params[neural_net.params_info[c][1][0]:neural_net.params_info[c][1][1]].reshape(neural_net.params_info[c][0]) 
            c += 1
        return (1 - dE2)

class OptiSRExactTime:
    def __init__(self, neural_net, sr_lambda, n_batches=1, dtype=tc.complex64, device='cpu'):
        n_params = neural_net.params_info[-1][1][1]
        #self.p_theta = tc.zeros(n_params, n_states, dtype=dtype, device=device).share_memory_()
        self.regularization = sr_lambda * tc.eye(n_params, dtype=dtype, device=device)
        self.n_batches = n_batches
    def __call__(self, neural_net0, neural_nett, sample, E_loc_psi_phi, E_loc_phi_psi):
        n_states_batch = int(sample.n_states / self.n_batches)
        p_theta = []
        for k in range(self.n_batches):
            p_theta.append(get_gradients_vmap(neural_nett, sample.states[n_states_batch * k : n_states_batch * (k+1), : ]))
        p_theta = tc.cat(p_theta , 1)    

        exact_ln_psi_0 = neural_net0(sample.states)[:,0].detach()
        exact_ratio_0 = tc.exp(exact_ln_psi_0 - exact_ln_psi_0[0])
        # the amplitude of a given configuration
        P_x_0 = abs(exact_ratio_0)**2 / tc.sum(abs(exact_ratio_0)**2)

        exact_ln_psi_t = neural_nett(sample.states)[:,0].detach()
        exact_ratio_t = tc.exp(exact_ln_psi_t - exact_ln_psi_t[0])
        # the amplitude of a given configuration
        P_x_t = abs(exact_ratio_t)**2 / tc.sum(abs(exact_ratio_t)**2)

        #print('size =', (E_loc_psi_phi * P_x_t).size())
        E_loc_psi_phi_avg = tc.sum(E_loc_psi_phi * P_x_t)
        E_loc_phi_psi_avg = tc.sum(E_loc_phi_psi * P_x_0)
        dE1 = E_loc_phi_psi_avg * E_loc_psi_phi
        dE2 = E_loc_phi_psi_avg * E_loc_psi_phi_avg

        S1 = tc.matmul(p_theta * P_x_t, p_theta.t().conj())
        p_theta_avg = (p_theta * P_x_t).sum(dim=1)
        S2 = tc.matmul(p_theta_avg.reshape(-1,1), p_theta_avg.conj().reshape(1,-1))
        S_matrix = S1 - S2 + self.regularization

        F = (dE1 * p_theta * P_x_t).sum(dim=1) - dE2 * p_theta_avg

        p_params = tc.linalg.solve(S_matrix,-F)
        c = 0
        for grad_n_k in neural_nett.parameters():
            grad_n_k.grad = p_params[neural_nett.params_info[c][1][0]:neural_nett.params_info[c][1][1]].reshape(neural_nett.params_info[c][0]) 
            c += 1
        return (1 - dE2)


class OptiMinSRTime:
    def __init__(self, n_states, sr_lambda, dtype=tc.complex64, device='cpu', n_procs=None, pool=None):
        #self.p_theta = tc.zeros(n_params, n_states, dtype=dtype, device=device).share_memory_()
        self.regularization = sr_lambda * tc.eye(n_states, dtype=dtype, device=device)
        self.n_procs = n_procs
        self.pool = pool
    def __call__(self, neural_net, sample, E_loc_psi_phi, E_loc_phi_psi):
        #print("tc.cuda.memory_allocated: %fGB"%(tc.cuda.memory_allocated(0)/1024/1024/1024))

        E_loc_psi_phi_avg = tc.sum(E_loc_psi_phi)/sample.n_states
        E_loc_phi_psi_avg = tc.sum(E_loc_phi_psi)/sample.n_states
        dE1 = -E_loc_phi_psi_avg * (E_loc_psi_phi - E_loc_psi_phi_avg)
        n_batch = 10
        n_states_k = int(sample.n_states / n_batch)
        p_theta = []
        for k in range(n_batch):
            p_theta.append(get_gradients_vmap(neural_net, sample.states[n_states_k * k : n_states_k * (k+1), : ]))
        p_theta = tc.cat(p_theta , 1)    
        print(p_theta.size())
        #p_theta = get_gradients_vmap(neural_net, sample.states)
        minX = p_theta / np.sqrt(sample.n_states)
        #rint("tc.cuda.memory_allocated: %fGB"%(tc.cuda.memory_allocated(0)/1024/1024/1024))
        p_theta = []
        minS = tc.mm(minX.t().conj(), minX) + self.regularization
        minF = tc.linalg.solve(minS, dE1.view(dE1.size()[0], 1))
        p_params = tc.mm(minX, minF)
        minS = []
        minX = []
        #print("tc.cuda.memory_allocated: %fGB"%(tc.cuda.memory_allocated(0)/1024/1024/1024))
        tc.cuda.empty_cache() 

        #print("tc.cuda.memory_allocated: %fGB"%(tc.cuda.memory_allocated(0)/1024/1024/1024))


        c = 0
        for grad_n_k in neural_net.parameters():
            grad_n_k.grad = p_params[neural_net.params_info[c][1][0]:neural_net.params_info[c][1][1]].reshape(neural_net.params_info[c][0]) 
            c += 1


class OptiMinSR2Time:
    def __init__(self, neural_net, n_states, c_inv, sr_lambda, dtype=tc.complex64, device='cpu', n_procs=None, pool=None):
        #self.p_theta = tc.zeros(n_params, n_states, dtype=dtype, device=device).share_memory_()
        self.regularization = sr_lambda * tc.eye(n_states, dtype=dtype, device=device)
        self.n_procs = n_procs
        self.pool = pool
        n_params = neural_net.params_info[-1][1][1]
        self.minX = tc.zeros(n_params, n_states, dtype=neural_net.dtype,device=neural_net.device)
        self.minS_inv = tc.zeros(n_states, n_states, dtype=neural_net.dtype,device=neural_net.device)
        self.c_inv = c_inv
    def __call__(self, epoch_k, neural_net, sample, E_loc_psi_phi, E_loc_phi_psi):
        #print("tc.cuda.memory_allocated: %fGB"%(tc.cuda.memory_allocated(0)/1024/1024/1024))

        E_loc_psi_phi_avg = tc.sum(E_loc_psi_phi)/sample.n_states
        E_loc_phi_psi_avg = tc.sum(E_loc_phi_psi)/sample.n_states
        dE1 = -E_loc_phi_psi_avg * (E_loc_psi_phi - E_loc_psi_phi_avg)
        n_batch = 10
        n_states_k = int(sample.n_states / n_batch)
        p_theta = []
        if epoch_k % self.c_inv == 0:
            for k in range(n_batch):
                p_theta.append(get_gradients_vmap(neural_net, sample.states[n_states_k * k : n_states_k * (k+1), : ]))
            p_theta = tc.cat(p_theta , 1)    
            print(p_theta.size())
            #p_theta = get_gradients_vmap(neural_net, sample.states)
            self.minX = p_theta / np.sqrt(sample.n_states)
            #rint("tc.cuda.memory_allocated: %fGB"%(tc.cuda.memory_allocated(0)/1024/1024/1024))
            p_theta = []
            minS = tc.mm(self.minX.t().conj(), self.minX) + self.regularization
            self.minS_inv = tc.linalg.inv(minS)

        #minF = tc.linalg.solve(minS, dE1.view(dE1.size()[0], 1))
        p_params = tc.mm(self.minX, tc.mm(self.minS_inv, dE1.view(dE1.size()[0], 1)))
        #minS = []
        #minX = []
        #print("tc.cuda.memory_allocated: %fGB"%(tc.cuda.memory_allocated(0)/1024/1024/1024))
        tc.cuda.empty_cache() 

        #print("tc.cuda.memory_allocated: %fGB"%(tc.cuda.memory_allocated(0)/1024/1024/1024))
        print('S[0,0] = ',self.minS_inv[0,0])

        c = 0
        for grad_n_k in neural_net.parameters():
            grad_n_k.grad = p_params[neural_net.params_info[c][1][0]:neural_net.params_info[c][1][1]].reshape(neural_net.params_info[c][0]) 
            c += 1

'''
t-VMC
'''
class OptitVMC:
    def __init__(self, neural_net, n_batches=1, dtype=tc.complex64, device='cpu'):
        #layer_size = np.array(layer_size)
        #n_params = sum(layer_size[0:len(layer_size)-1] * layer_size[1:len(layer_size)]) + sum(layer_size[1:len(layer_size)])
        n_params = neural_net.params_info[-1][1][1]
        #self.p_theta = tc.zeros(n_params, n_states, dtype=dtype, device=device).share_memory_()
        self.n_batches = n_batches
    def __call__(self, neural_net, sample, E_loc):
        #print('SR method')
        E_loc_avg = E_loc.mean()
        '''
        For large ssystem, becasue of the limitation of memeory, 
        we cannot use vap to get the derivatives for all samples,
        thus we need get derivatives of a batch of samples sequentially.
        '''
        n_states_batch = int(sample.n_states / self.n_batches)
        p_theta = []
        for k in range(self.n_batches):
            p_theta.append(get_gradients_vmap(neural_net, sample.states[n_states_batch * k : n_states_batch * (k+1), : ]))
        p_theta = tc.cat(p_theta , 1)    
        S1 = tc.matmul(p_theta, p_theta.t().conj())/sample.n_states
        p_theta_avg = p_theta.sum(dim=1)/sample.n_states
        S2 = tc.matmul(p_theta_avg.reshape(-1,1), p_theta_avg.conj().reshape(1,-1))
        #print('size of S1 = ',S1.size())
        S_matrix = S1 - S2
        #print(S_matrix.size())
        F = (E_loc * p_theta).sum(dim=1)/sample.n_states - E_loc.sum() * p_theta_avg /sample.n_states
        Spinv = tc.linalg.pinv(S_matrix)
        p_params = tc.matmul(-1.0j * Spinv, F)
        return E_loc_avg, p_params

class OptitVMCExact:
    def __init__(self, neural_net, n_batches=1, dtype=tc.complex64, device='cpu'):
        #layer_size = np.array(layer_size)
        #n_params = sum(layer_size[0:len(layer_size)-1] * layer_size[1:len(layer_size)]) + sum(layer_size[1:len(layer_size)])
        n_params = neural_net.params_info[-1][1][1]
        #self.p_theta = tc.zeros(n_params, n_states, dtype=dtype, device=device).share_memory_()
        self.n_batches = n_batches
    def __call__(self, neural_net, sample, E_loc):
        #print('we are using exact sampling with SR!')
        '''
        For large system, becasue of the limitation of memeory, 
        we cannot use vap to get the derivatives for all samples,
        thus we need get derivatives of a batch of samples sequentially.
        '''
        n_states_batch = int(sample.n_states / self.n_batches)
        p_theta = []
        for k in range(self.n_batches):
            p_theta.append(get_gradients_vmap(neural_net, sample.states[n_states_batch * k : n_states_batch * (k+1), : ]))
        p_theta = tc.cat(p_theta , 1)   
        #print('the size of p_theta = ',p_theta.size()) 
        #p_theta = get_gradients_vmap(neural_net, sample.states)
        exact_ln_psi = neural_net(sample.states)[:,0].detach()
        exact_ln_ratio = tc.exp(exact_ln_psi - exact_ln_psi[0])
        # the amplitude of a given configuration
        P_x = abs(exact_ln_ratio)**2 / tc.sum(abs(exact_ln_ratio)**2)
        S1 = tc.matmul(p_theta * P_x, p_theta.t().conj())
        p_theta_avg = (p_theta * P_x).sum(dim=1)
        S2 = tc.matmul(p_theta_avg.reshape(-1,1), p_theta_avg.conj().reshape(1,-1))
        #print('size of S1 = ',S1.size())
        S_matrix = S1 - S2
        #print(S_matrix.size())
        E_loc_avg = tc.sum(E_loc * P_x)
        F = (E_loc * p_theta * P_x).sum(dim=1) - E_loc_avg * p_theta_avg
        Spinv = tc.linalg.pinv(S_matrix)
        p_params = tc.matmul(-1.0j * Spinv, F)
        #p_params = tc.linalg.solve(S_matrix, -1j * F)
        c = 0
        # for grad_n_k in neural_net.parameters():
        #     grad_n_k.grad = p_params[neural_net.params_info[c][1][0]:neural_net.params_info[c][1][1]].reshape(neural_net.params_info[c][0]) 
        #     c += 1
        return E_loc_avg, p_params

def update_params(n0, neural_net, Km):
    params0 = list(n0.parameters())
    c = 0
    for params_n in neural_net.parameters():
        params_n.data = params0[c].data + Km[neural_net.params_info[c][1][0]:neural_net.params_info[c][1][1]].reshape(neural_net.params_info[c][0]) 
        c += 1
    
    

def save_params(neural_net):
    nn0 = []
    c = 0
    for params_n in neural_net.parameters():
        nn0.append(tc.clone(params_n).data)
    return nn0


    

###########################################
###########################################
## multiple time ##

class OptiGDExactTimeOneNN:
    def __init__(self, dtype=tc.complex64, device='cpu'):
        self.dtype = dtype
        self.device = device
    def __call__(self, K, ct, neural_net0, neural_nett, sample, E_loc_r_list, E_loc_l_list):
        # K means how many time points we need to compute 
        E_loc_r_avg = tc.zeros(K, dtype=neural_nett.dtype,device=neural_nett.device)
        E_loc_l_avg = tc.zeros(K, dtype=neural_nett.dtype,device=neural_nett.device)
        P_eloc = []
        for s in range(K+1):
            if s == 0:
                exact_s = tc.exp(neural_net0(sample.states))[:,0]
                P_eloc.append(abs(exact_s.detach())**2 / tc.sum(abs(exact_s.detach())**2))
            else:
                neural_nett.t = 1 + s*ct
                exact_s = tc.exp(neural_nett(sample.states))[:,0]
                P_eloc.append(abs(exact_s.detach())**2 / tc.sum(abs(exact_s.detach())**2))
        for s in range(K):
            E_loc_r_avg[s] = sum(E_loc_r_list[s] * P_eloc[s+1].detach())
            E_loc_l_avg[s] = sum(E_loc_l_list[s] * P_eloc[s].detach())
        
        # compute the loss function:
        loss = 1.0+0.0j
        for s in range(1, K):
            #print('s=',s)
            neural_nett.t = 1 + s*ct
            exact_ln_s = neural_nett(sample.states)[:,0]
            dE = 1.0 + 0.0j
            for k in range(1,K+1):
                if (k != s) | (k != s+1):
                    dE = dE * E_loc_r_avg[k-1] * E_loc_l_avg[k-1]
                I_s1 = E_loc_r_avg[s-1] * E_loc_l_avg[s-1]
                I_s2 = E_loc_r_avg[s] * E_loc_l_avg[s]
                dE1 = I_s2 * E_loc_l_avg[s-1] * (E_loc_r_list[s-1] - E_loc_r_avg[s-1])
                dE1 += I_s1 * E_loc_r_avg[s] * (E_loc_l_list[s] - E_loc_l_avg[s])
                dE = -1*(dE * dE1) * P_eloc[s]
            exact_ln_s.backward(dE)
        # compute the last one:
        neural_nett.t = 1 + K*ct
        exact_ln_s= neural_nett(sample.states)[:,0]
        dE = 1.0 + 0.0j
        for k in range(1, K+1):
            if (k != K):
                dE = dE * E_loc_r_avg[k-1] * E_loc_l_avg[k-1]
            else:
                #print('k=',k)
                dE = dE*(E_loc_l_avg[k-1]) * (E_loc_r_list[k-1] - E_loc_r_avg[k-1])
        dE = -dE * P_eloc[-1]
        exact_ln_s.backward(dE)
        return (1 - tc.prod(E_loc_l_avg * E_loc_r_avg))
        #return tc.log(tc.prod(E_loc_l_avg * E_loc_r_avg))

