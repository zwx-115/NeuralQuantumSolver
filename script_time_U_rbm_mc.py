import sys
sys.path.append("/home/wenxuan/CODE/NNQS/")
import torch as tc
from torch import nn
import os
import numpy as np 
import time

# include nnqs
import nn_model 
import quantum_system as quant_sys
import sampling as sp
import calculate_grads as gds
import local_energy as eloc
import trotter_block as tb
import measurement as meas

def main_run():
    #--------------------
    # data type and device:
    dtype = tc.complex128
    device = 'cuda:1'
    seed_num = 4111
    #--------------------
    # neural network:
    Lx = 4
    Ly = 1
    n_sites = Lx * Ly
    # load ground energy neural network:
    neural_net_gs = tc.load(f'RBM_gs_4_410.pt')
    #neural_net_gs = tc.load(f'FNN_gs_2_by_2_4666.pt')
    # generate new neural network:
    nh = 10
    neural_net0 = nn_model.RBM(n_sites, nh*n_sites, dtype=dtype, device=device, seed_num=seed_num)
    neural_nett =  nn_model.RBM(n_sites, nh*n_sites, dtype=dtype, device=device, seed_num=seed_num)
    # change parameters:
    neural_net0.load_state_dict(neural_net_gs.state_dict())
    neural_nett.load_state_dict(neural_net_gs.state_dict())
    #--------------------
    # quantum system:
    quant_sys_name = 'ising' 
    boundary_condition = 'open'
    ising = quant_sys.QuantSys(Lx, Ly, boundary_condition, quant_sys_name, *[1.0, -0.5, -0.5])
    #--------------------
    # sampling:
    # sampling:
    n_thermals = 100
    n_sweeps = 1
    n_states = 10000
    #sample_0 = sp.SamplingExact(n_sites, 'spin', dtype=dtype,device=device)
    #sample_t = sp.SamplingExact(n_sites, 'spin', dtype=dtype,device=device)
    sample_0 = sp.Sampling(n_sites, n_states, n_thermals, n_sweeps, 'spin', dtype=dtype,device=device)
    sample_t = sp.Sampling(n_sites, n_states, n_thermals, n_sweeps, 'spin', dtype=dtype,device=device)
    #--------------------
    # the method of updating parameters:
    # update stragety:
    get_grads = gds.OptiGDTime(dtype=dtype, device=device)
    sr_lambda = 0.01
    n_batches = 10
    #get_grads = gds.OptiSRTime(neural_nett, sr_lambda, n_batches,dtype=dtype, device=device)

    #--------------------
    # optimizer:
    cut_off = 1e-9
    n_epochs = 1000
    lr = 1e-3
    optimizer = tc.optim.AdamW(neural_nett.parameters(),lr=lr)
    #optimizer = tc.optim.SGD(neural_nett.parameters(),lr=lr)
    #--------------------
    # time setting:
    dt = 0.1
    evo_t = 0.0
    time_final = 0.1
    n_time = int(time_final/dt)
    n_bonds = 1
    #--------------------
    # measurement:
    sigma_x_avg = meas.measurement_exact_all(neural_net0, n_sites, 'X')
    # record parameter settings:
    # create the text file:
    if Ly == 1:
        sys_dim = '1D'
    else:
        sys_dim = '1D'
    f = open(f"time_full_U_mc_{neural_net0.__class__.__name__}_{Lx}_by_{Ly}_{n_sites}_sites_{sys_dim}_system_{sample_0.n_states}_samples_{cut_off:.1e}_{seed_num}.txt","w")
    f_loss=open(f"overlap_full_U_mc_{neural_net0.__class__.__name__}_{Lx}_by_{Ly}_{n_sites}_sites_{sys_dim}_system_{sample_0.n_states}_samples_{cut_off:.1e}_{seed_num}.txt","w")
    # print time in terms of DD-MMM-YYYY 13:45:56
    f.write(time.strftime("%d-%b-%Y %H:%M:%S \n", time.localtime())) 
    f.write(f'the quantum system is {quant_sys_name} with {boundary_condition}\n')
    f.write(f'the site: Lx = {Lx}, Ly = {Ly} \n')
    f.write('--------------------\n')
    f.write(f'device is on: {device}; data type: {dtype} \n')
    f.write(f'neural network is {neural_net0.__class__.__name__} \n')
    f.write(f'total number of neural network parameters Np={neural_net0.params_info[-1][1][1]}\n')
    f.write('--------------------\n')
    f.write(f'the sampling is {sample_0.__class__.__name__}\n')
    f.write(f'total number of samples Ns = {sample_0.n_states} \n')
    f.write(f'total number of sweep: n_sweeps={n_sweeps}, the number of thermalizations: n_thermals={n_thermals}\n')
    f.write('--------------------\n')
    f.write(f'the method of getting gradients: {get_grads.__class__.__name__}\nshift diag sr_lambda={sr_lambda} and number of batchs= {n_batches}(sr_lambda and n_batches are only available for SR method) \n')
    f.write(f'the optimizer: {optimizer.__class__.__name__}  with initial learning rate: lr = {lr}\n')
    f.write(f'number of epochs n_epochs={n_epochs}\n')
    f.write(f'cut off for each trotter block is:{cut_off:.2e}\n')
    f.write('--------------------\n')
    f.write(f'time interval dt = {dt}, final time tf = {time_final}, and time points n_time = {n_time}\n')
    f.write(f'the number bond for each Hamiltonian is {n_bonds}\n')
    f.write('--------------------\n')
    f.write(f'time: {evo_t} ==> sigma_x_1_avg = {sigma_x_avg[0]: .10f} \n')
    f.write('--------------------\n')
    f.flush()
    #--------------------
    # displaying on the screen:
    print('device is on: {device}; data type: {dtype}')
    print("the system is ising model, and the site is:", n_sites)
    print('number of thermalization is ',n_thermals)
    print(neural_net0.params_info)
    #--------------------
    # t0_thermal = time.time()
    for _ in range(n_thermals):
        sp.single_sweep(neural_net0, sample_0)
        sp.single_sweep(neural_nett, sample_t)
    # tf_thermal = time.time()
    # print(f'time on thermalization is: {tf_thermal - t0_thermal: .5f}s')
    bond_indx = tb.generate_bind_indx(n_sites, n_bonds)
    print('bond_indx = ',bond_indx)
    for _ in range(n_time):
        print(f'time: {evo_t:.3f}  =====================')
        for n in [1]:
            print('--------------------')
            print('n = ', n)
            f.write(f'{n} >>>>> ')
            local_U = tb.get_Un(ising, 2*dt, n, n_bonds, dtype=dtype, device=device)
            print('size of U: ',local_U.U.size())
            bond_ind = list(range(n-1, min(n+n_bonds, ising.n_sites)))
            print(bond_ind)
            overlap_best = 1
            overlap_last = 1
            c_epoch = 0

            optimizer = tc.optim.AdamW(neural_nett.parameters(),lr=lr)
            #optimizer = tc.optim.SGD(neural_nett.parameters(),lr=lr)

            for k in range(n_epochs):
                # if k % 100 == 0:
                #     print(k, ': ==============>')
                c_epoch += 1
                t0_epoch = time.time()
                #print("0: %fGB"%(tc.cuda.memory_allocated(0)/1024/1024/1024))
                sp.single_sweep(neural_net0, sample_0)
                sp.single_sweep(neural_nett, sample_t)
                #print("1: %fGB"%(tc.cuda.memory_allocated(0)/1024/1024/1024))
                tf = time.time()
                #print(f'time on sampling is: {tf - t0: .5f}s')
                # compute local energy:
                t0_E = time.time()
                E_psi_phi = eloc.local_energy_psi_phi_trotter(neural_net0, neural_nett, local_U, sample_t, bond_ind)
                #print("1.5: %fGB"%(tc.cuda.memory_allocated(0)/1024/1024/1024))
                E_phi_psi = eloc.local_energy_phi_psi_trotter(neural_net0, neural_nett, local_U, sample_0, bond_ind)
                tf_E = time.time()
                #tc.cuda.empty_cache() 

                #print("2: %fGB"%(tc.cuda.memory_allocated(0)/1024/1024/1024))
                #print(f'time on energy is: {tf_E - t0_E: .5f}s')
                #overlap = abs(1 - tc.mean(E_psi_phi) * tc.mean(E_phi_psi))
                overlap = get_grads(neural_nett, sample_t, E_psi_phi, E_phi_psi)
                #print(f'the overlap is: {overlap.item(): .20f}')
                if k % 100 == 0:
                    print(k,': ==============>')
                    print(f'the overlap is: {tc.abs(overlap): .5e}')
                overlap_best = min(overlap_best, abs(overlap))
                overlap_last = abs(overlap)
                if abs(overlap) < cut_off:
                    print(f'{k}: the overlap is: {tc.abs(overlap): .5e}')
                    break
                t0_opti = time.time()
                #opti_params(neural_nett, sample_t, E_psi_phi, E_phi_psi)
                #print("3: %fGB"%(tc.cuda.memory_allocated(0)/1024/1024/1024))
                #E_psi_phi=[]
                #E_phi_psi = []
                tf_opti = time.time()
                #print(f'time on gradients is: {tf_opti - t0_opti: .5f}s')
                optimizer.step()
                optimizer.zero_grad()
                #print("4: %fGB"%(tc.cuda.memory_allocated(0)/1024/1024/1024))
                #tc.cuda.empty_cache() 
                tf_epoch = time.time()
                #print(f'time: {tf_epoch - t0_epoch: .5f}s')
                #print(neural_net0.linear_stack[0].weight[0,0])
                #print(neural_nett.linear_stack[0].weight[0,0])
                #time.sleep(1)
                f_loss.write(f'{abs(overlap):.10f}\n')
            #print('before: nn0: ',neural_net0.fc_stack[0].weight[0,0])
            #print('before: nnt: ',neural_nett.fc_stack[0].weight[0,0])
            neural_net0.load_state_dict(neural_nett.state_dict())
            #print('after: nn0: ',neural_net0.fc_stack[0].weight[0,0])
            #print('after: nnt: ',neural_nett.fc_stack[0].weight[0,0])
            f.write(f'{c_epoch}: {overlap_last: .10f},  {overlap_best: .10f} \n')
            
            f.flush()
            f_loss.flush()


        evo_t += dt    
        #sigma_x_avg = nt.measurement_metropolis(neural_nett, sample_0, 1, 'X')
        sigma_x_avg = meas.measurement_exact_all(neural_nett, n_sites, 'X')
        print(f't = {evo_t: .5f}: sigma_x_1_avg =  {sigma_x_avg[0]: .10f}')
        f.write(f'========> {evo_t: .5f}: {sigma_x_avg[0]: .10f} \n')
        f.flush()
        nn_name = f'{neural_net0.__class__.__name__}_time_trotter_{Lx}_by_{Ly}_{n_sites}_{evo_t:.1f}_{cut_off:.1e}_{seed_num}.pt'
        #tc.save(neural_nett, nn_name)
    f.write('------------------------\n')
    f.write(time.strftime("%d %b %Y %H:%M:%S", time.localtime())) 
    f.close()
if __name__ == '__main__':
    main_run()