import sys
sys.path.append("/home/users/sutd/1004957/smooth_neural_network/code/")
import torch as tc
import time
import math
# include nnqs
import nn_model 
import quantum_system as quant_sys
import sampling as sp
import calculate_grads as gds
import local_energy as eloc

def main_run():
    #--------------------
    # data type and device:
    dtype=tc.complex128
    device = 'cuda:0'
    seed_num = 1420
    #--------------------
    # neural network:
    Lx = 14
    Ly = 1
    n_sites = Lx * Ly

    #neural_net = nn_model.FNN([n_sites, 20*n_sites, 10*n_sites, 1], 'poly1', dtype=dtype, device=device, seed_num = seed_num)
    nh = 20
    neural_net = nn_model.RBM(n_sites, nh*n_sites, dtype=dtype, device=device, seed_num = seed_num)
    #--------------------
    # quantum system:
    # H = J\sum sigma_i^z sigma_j^z + (hx\sum sigma_i^x + hz\sum sigma_i^z)
    boundary_condition = 'closed'
    J = 0.0
    hx = -1.0
    hz = -0.0
    quant_sys_name = 'ising'
    ising = quant_sys.QuantSys(Lx, Ly, boundary_condition, quant_sys_name, *[J, hx, hz])
    #--------------------
    # sampling:
    n_sweeps = 0
    n_thermals = 0
    n_states = 2**n_sites
    qunt_sys_type = 'spin'
    sample = sp.SamplingExact(n_sites, qunt_sys_type, dtype=dtype,device=device)
    #sample = sp.Sampling(n_sites, n_states, n_thermals, n_sweeps, 'spin', dtype=dtype,device=device)
    #--------------------
    # the method of updating parameters:
    sr_lambda = 0.01
    n_batches = 1
    #get_grads = gds.OptiSR(neural_net, sr_lambda, n_batches=n_batches,dtype=dtype, device=device)
    #get_grads = gds.OptiSRExact(neural_net, sr_lambda, n_batches=n_batches,dtype=dtype, device=device)
    get_grads = gds.OptiGDExact(dtype=dtype, device=device)
    #--------------------
    # optimizer:
    n_epochs = 500
    lr = 1e-4
    optimizer = tc.optim.AdamW(neural_net.parameters(),lr=lr)
    #optimizer = tc.optim.SGD(neural_net.parameters(),lr=lr)
    #scheduler = lr_policy.MultiStepLR(optimizer, milestones = [10, 20, 40, n_epochs], gamma=0.8)

    #--------------------
    # record parameter settings:
    # create the text file:
    if Ly == 1:
        sys_dim = '1D'
    else:
        sys_dim = '2D'
    f = open(f"gs_{neural_net.__class__.__name__}_{n_sites}_sites_{sys_dim}_system_{sample.n_states}_samples_{seed_num}.txt","w")
    # print time in terms of DD-MMM-YYYY 13:45:56
    f.write(time.strftime("%d-%b-%Y %H:%M:%S \n", time.localtime())) 
    f.write(f'the quantum system is {quant_sys_name} with {boundary_condition}\n')
    f.write(f'the site: Lx = {Lx}, Ly = {Ly} \n')
    f.write('--------------------\n')
    f.write(f'device is on: {device}; data type: {dtype} \n')
    f.write(f'neural network is {neural_net.__class__.__name__}, the hidden node is {nh} \n')
    f.write(f'nh = {nh*n_sites}, and total number of neural network parameters Np={neural_net.params_info[-1][1][1]}\n')
    f.write('--------------------\n')
    f.write(f'the sampling is {sample.__class__.__name__}\n')
    f.write(f'total number of samples Ns = {sample.n_states} \n')
    f.write(f'total number of sweep: n_sweeps={n_sweeps}, the number of thermalizations: n_thermals={n_thermals}\n')
    f.write('--------------------\n')
    f.write(f'the method of getting gradients: {get_grads.__class__.__name__}\n shift diag sr_lambda={sr_lambda} and number of batchs= {n_batches}(sr_lambda and n_batches are only available for SR method) \n')
    f.write(f'the optimizer: {optimizer.__class__.__name__}  with initial learning rate: lr = {lr}\n')
    f.write(f'number of epochs n_epochs={n_epochs}\n')
    f.write('--------------------\n')

    #--------------------
    # displaying on the screen:
    print('device is on: {device}; data type: {dtype}')
    print("the system is ising model, and the site is:", n_sites)
    print('number of thermalization is ',n_thermals)
    print(neural_net.params_info)
    #--------------------
    # run:
    # for _ in range(n_thermals):
    #     sp.single_sweep(neural_net, sample)
    time_run = 0
    t_energy = 0
    for k in range(n_epochs):
        print('------------------------')
        t0_epoch = time.time()
        print(f'====> Epoch: {k}')
        #print('lr: {1: .10f}'.format(k, optimizer.param_groups[0]['lr']))
        #print('before:', neural_net.linear_stack[0].weight[0])
        # t1 = time.time()
        #sp.single_sweep(neural_net, sample)
        # t2 = time.time()
        #print(f'time on sample = {t2 - t1: .5f}s')
        #if k >= 100:
        #print('state = \n',sample.states)
        t0_eloc = time.time()
        E_loc = eloc.local_energy(ising, neural_net, sample)
        tf_eloc = time.time()
        if k>0:
            t_energy += tf_eloc - t0_eloc
        t0_get_grads = time.time()
        E_loc_avg = get_grads(neural_net, sample, E_loc)
        tf_get_grads = time.time()
        
        #print(f'time on getting local energy = {t_e2 - t_e1: .5f}s')
        print(f'E_avg = {E_loc_avg.item(): .15f}')
        if math.isnan(tc.abs(E_loc_avg)):
            raise RuntimeError(f'Wrong ground energy:  {E_loc_avg.item()}')
        
        #print(f'time on gettting derivatives = {t2 - t1: .5f}s')
        t1 = time.time()
        optimizer.step()
        #scheduler.step()
        optimizer.zero_grad()
        t2 = time.time()
        #print(f'time on optimizer: {t2 - t1: .5f}s')
        tf_epoch = time.time()
        if k>0:
            time_run += tf_epoch - t0_epoch
        #sigma_x_avg = gs.measurement_exact(neural_net, n_sites, 1, 'X')
        #print(f'sigma_x_avg =  {sigma_x_avg: .20f}')
        #print(f'time on each epoch = {tf_epoch - t0_epoch: .5f}s')

        f.write(f'{k+1}: {E_loc_avg.item(): .15f} , {tf_epoch - t0_epoch: .5f}s \n')
        f.flush()
    f.write('------------------------\n')
    f.write(f'total time = {time_run: .8f}s \n')
    f.write(time.strftime("%d %b %Y %H:%M:%S", time.localtime())) 
    f.close()
    nn_name = f'{neural_net.__class__.__name__}_gs_{n_sites}_{seed_num}.pt'
    tc.save(neural_net, nn_name)

    
    

if __name__ == '__main__':
    main_run()