###########################################
###########################################
### measurement part ###
'''
Here, we use full summation or Monte Carlo method to get samples and get the values of observations
'''
import torch as tc
import sampling as sp

'''
1. Full summation(exact sampling) when the system size <= 14:
'''
def measurement_exact(neural_net, n_sites, site_n, measure_type):
    n_states = 2**n_sites
    exact_state = tc.zeros(n_states, n_sites, dtype=neural_net.dtype,device=neural_net.device)
    for i in range(n_states):
        i_bin = '{:0b}'.format(i).zfill(n_sites)
        for k in range(n_sites):
            exact_state[i,k] = tc.tensor(int(i_bin[k]))
    exact_state = -2 * exact_state + 1
    exact_ln_psi = neural_net(exact_state)
    exact_lnψ_ratio = tc.exp(exact_ln_psi - exact_ln_psi[0])
    abs_psi = tc.abs(exact_lnψ_ratio) ** 2
    P_psi = abs_psi / abs_psi.sum()
    # exact_lnψ = model(exact_state) 
    # exact_lnψ_ratio = CUDA.@allowscalar exp.(exact_lnψ .- exact_lnψ[1])
    # P_ψ = abs.(exact_lnψ_ratio).^2 / sum(abs.(exact_lnψ_ratio).^2)
    if measure_type == 'X':
        new_states = tc.clone(exact_state)
        new_states[:,site_n-1] = -new_states[:,site_n+1]
        sigma_x = tc.exp( neural_net(new_states) - exact_ln_psi)
        sigma_x_avg = (sigma_x * P_psi).sum()
        return sigma_x_avg
    
def measurement_exact_all(neural_net, n_sites, measure_type):
    assert n_sites <= 14, f'system size must beless than 14 sites!'
    n_states = 2**n_sites
    exact_state = tc.zeros(n_states, n_sites, dtype=neural_net.dtype,device=neural_net.device)
    for i in range(n_states):
        i_bin = '{:0b}'.format(i).zfill(n_sites)
        for k in range(n_sites):
            exact_state[i,k] = tc.tensor(int(i_bin[k]))
    exact_state = -2 * exact_state + 1
    exact_ln_psi = neural_net(exact_state).detach()
    exact_lnψ_ratio = tc.exp(exact_ln_psi - exact_ln_psi[0])
    abs_psi = tc.abs(exact_lnψ_ratio) ** 2
    P_psi = abs_psi / abs_psi.sum()
    # exact_lnψ = model(exact_state) 
    # exact_lnψ_ratio = CUDA.@allowscalar exp.(exact_lnψ .- exact_lnψ[1])
    # P_ψ = abs.(exact_lnψ_ratio).^2 / sum(abs.(exact_lnψ_ratio).^2)
    sigma_x_avg = tc.zeros(n_sites,dtype=neural_net.dtype, device=neural_net.device)
    if measure_type == 'X':
        for site_k in range(n_sites):
            new_states = tc.clone(exact_state)
            new_states[:,site_k] = -new_states[:,site_k]
            sigma_x = tc.exp( neural_net(new_states).detach() - exact_ln_psi)
            sigma_x_avg[site_k] = (sigma_x * P_psi).sum()
        return tc.real(sigma_x_avg)


'''
2. measurement for Monte Carlo method when system size > 14 :
'''
def measurement_metropolis(neural_net, sample, site_n, measure_type):
    sigma_x_avg = 0
    for _ in range(sample.n_thermals):
        sp.single_sweep(neural_net, sample)
    for _ in range(10):
        sp.single_sweep(neural_net, sample)
        if measure_type == 'X':
            new_states = tc.clone(sample.states)
            new_states[:,site_n-1] = -new_states[:,site_n-1]
            sigma_x = tc.exp(neural_net(new_states).detach() - neural_net(sample.states).detach())
            sigma_x_avg += tc.mean(sigma_x)
    return sigma_x_avg / 10

def measurement_metropolis_x_z(neural_net, sample):
    n_sites = sample.states.size(1)
    sigma_x_avg = tc.zeros(n_sites,dtype=neural_net.dtype, device=neural_net.device)
    sigma_z_avg = tc.zeros(n_sites,dtype=neural_net.dtype, device=neural_net.device)
    for _ in range(10):
        sp.single_sweep(neural_net, sample)
        for site_n in range(n_sites):
            new_states = tc.clone(sample.states)
            new_states[:,site_n] = -new_states[:,site_n]
            sigma_x = tc.exp(neural_net(new_states).detach() - neural_net(sample.states).detach())
            sigma_x_avg[site_n] += tc.mean(sigma_x)
            sigma_z_avg[site_n] += tc.sum(sample.states[:,site_n]) / sample.n_states
    return sigma_x_avg / 10, sigma_z_avg / 10
