# This is the soruce file for building neural network models

###########################################
###########################################
### import packages ###
import torch as tc
from torch import nn
import numpy as np 
import torch.multiprocessing as mp
import torch.distributed as dist
import time
from inspect import getfullargspec
import torch.nn.functional as F

###########################################
###########################################
### environment setting ###

###########################################
###########################################
### neural nertwork ###

'''
In this part, we can customize the neural network
and define activation fucntions that doesn't exist 
in the pytorch package

1. We record the information of neural network parameters:
the position of each layer in order to reshape dervitives of all 
parameters as a vector and collect them of all samples to 
construct S matrix in SR method. If we don't need to use SR method,
we use gradient descent and gradient force isntead, no need to use 
the information.

'''

###--------------------
###--------------------
# 1. feedforward neural network:

class FNN(nn.Module):
    def __init__(self, layer_size:list, act_func, dtype=tc.complex64, device='cpu', seed_num = 1234):
        super().__init__()
        tc.random.manual_seed(seed_num)
        print("the model is on the ",device)
        self.dtype = dtype
        self.device = device
        n_layers = len(layer_size) # total number of nn layers
        nn_models = []
        # generate a fnn struct except the last layer
        for m in range(0, n_layers - 2):
            nn_models.append(
            nn.Linear(layer_size[m], layer_size[m+1], dtype=dtype, device=device)
            )
            # generate the activation function
            # the name corresponds to the different activation functions:
            if act_func == 'poly1':
                nn_models.append(act_func_poly1())
            else:
                nn_models.append(act_func)
        # add the last layer:
        nn_models.append(nn.Linear(layer_size[-2], layer_size[-1], dtype=dtype, device=device))
        # construct fnn structure:
        self.linear_stack = nn.Sequential(*nn_models)
        self.linear_stack.apply(init_weights)

        # 2. record information of neurla network parameters:
        params_info = []
        count_k = 0
        for params_k in self.linear_stack.parameters():
            # the first element is the parameter's size:
            param_info_k = [list(params_k.size())]
            # the second element is the parameter's position:
            n_params_k = count_k + tc.numel(params_k)
            param_info_k.append([count_k, n_params_k])
            count_k = n_params_k
            params_info.append(param_info_k)
        self.params_info = params_info
    def forward(self, x):
            return self.linear_stack(x)

def init_weights(m):
    if isinstance(m,nn.Linear):
        nn.init.normal_(m.weight,std=0.05)
        nn.init.normal_(m.bias,std=0.05)
def init_weights_pbc(m):
    if isinstance(m,nn.Linear):
        nn.init.normal_(m.weight,std=0.05)
        nn.init.normal_(m.bias,std=0.05)

def num_parameters_fnn(nn_layers):
    layer_array = np.array(nn_layers)
    n_weights = sum(layer_array[0:len(layer_array)-1] * layer_array[1:len(layer_array)])
    n_bias = sum(layer_array[1:len(layer_array)])
    print("the total parameters of fnn = ",n_weights + n_bias)

def num_parameters_nn(neural_net):
    n_params = neural_net.params_info[-1][1][1]
    print("the total parameters of neural network = ", n_params)
    return n_params
    
###--------------------
###--------------------

###--------------------
###--------------------
# 2. convolutional neural network:

class CNN(nn.Module):
    def __init__(self, input_shape, conv_layers, fc_layers, act_func, dtype=tc.complex64, device='cpu', seed_num=1234):
        super().__init__()
        tc.random.manual_seed(seed_num)
        print("the model is on the ", device)
        self.dtype = dtype
        self.device = device

        # CNN Layers
        cnn_models = []
        in_channels = input_shape[0]
        for out_channels, kernel_size, stride, padding in conv_layers:
            cnn_models.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dtype=dtype, device=device)
            )
            if act_func == 'poly1':
                cnn_models.append(act_func_poly1())
            else:
                cnn_models.append(act_func)
            in_channels = out_channels
        self.cnn_stack = nn.Sequential(*cnn_models)

        # Fully Connected Layers
        n_flatten = self._get_flatten_size(input_shape)
        fc_layers = [n_flatten] + fc_layers
        fc_models = []
        for m in range(len(fc_layers) - 1):
            fc_models.append(
                nn.Linear(fc_layers[m], fc_layers[m + 1], dtype=dtype, device=device)
            )
            if m < len(fc_layers) - 2:
                if act_func == 'poly1':
                    cnn_models.append(act_func_poly1())
                else:
                    cnn_models.append(act_func)
        self.fc_stack = nn.Sequential(*fc_models)
        self.fc_stack.apply(init_weights)
        #self.fc_stack.apply(init_weights)
    
        params_info = []
        count_k = 0
        for params_k in self.cnn_stack.parameters():
            # the first element is the parameter's size:
            param_info_k = [list(params_k.size())]
            # the second element is the parameter's position:
            n_params_k = count_k + tc.numel(params_k)
            param_info_k.append([count_k, n_params_k])
            count_k = n_params_k
            params_info.append(param_info_k)
            
        for params_k in self.fc_stack.parameters():
            # the first element is the parameter's size:
            param_info_k = [list(params_k.size())]
            # the second element is the parameter's position:
            n_params_k = count_k + tc.numel(params_k)
            param_info_k.append([count_k, n_params_k])
            count_k = n_params_k
            params_info.append(param_info_k)
        self.params_info = params_info
    

    def _get_flatten_size(self, input_shape):
        x = tc.zeros((1, *input_shape), dtype=self.dtype, device=self.device)
        x = self.cnn_stack(x)
        return x.numel()

    def forward(self, x):
        x = x.view(x.size(0), 1, 1, x.size(1))  # Reshape input to match CNN expected input
        #print("before cnn, size = ",x.size())
        x = self.cnn_stack(x)
        #print("after cnn, size = ",x.size())
        x = x.view(x.size(0), -1)  # Flatten before passing to FC layers
        #print('after reshape: ',x.size())
        return self.fc_stack(x)

###--------------------
###--------------------

###--------------------
###--------------------
# 3. restricted Boltzmann machine:

class RBM(nn.Module):
    def __init__(self, nv:int, nh:int, dtype=tc.complex64, device='cpu', seed_num=1234):
        super().__init__()
        tc.random.manual_seed(seed_num)
        print("the model is on the ", device)
        self.dtype = dtype
        self.device = device
        self.v_bias = nn.Parameter(tc.randn(1, nv, dtype=dtype, device=device))
        nn.init.normal_(self.v_bias,std=0.05)
        nn_model = [nn.Linear(nv, nh, dtype=dtype, device=device)]
        self.rbm_stack = nn.Sequential(*nn_model)
        self.rbm_stack.apply(init_weights)
        params_info = [[list(self.v_bias.size()), [0, nv]]]
        count_k = nv
        for params_k in self.rbm_stack.parameters():
            # the first element is the parameter's size:
            param_info_k = [list(params_k.size())]
            # the second element is the parameter's position:
            n_params_k = count_k + tc.numel(params_k)
            param_info_k.append([count_k, n_params_k])
            count_k = n_params_k
            params_info.append(param_info_k)
        self.params_info = params_info
    def forward(self, x):
        if len(x.shape) == 2:
            res_v = (self.v_bias * x).sum(1,keepdim=True)
        elif len(x.shape) == 3:
            a2 = tc.unsqueeze(self.v_bias, 0)
            res_v=(a2 * x).sum(2,keepdim=True)
        res_h = 2 * (self.rbm_stack(x).cosh())
        return res_v + tc.log(res_h).sum(-1, keepdim=True)

            # res_v = (self.v_bias * x).sum(1,keepdim=True)
            # res_h = 2 * (self.rbm_stack(x).cosh())
            # #return (res_v.exp() * (res_h.prod(1,keepdim=True))).log()
            # return res_v + tc.log(res_h).sum(1, keepdim=True)
            #return res.prod(1,keepdim=True)
###--------------------
###--------------------

###--------------------
###--------------------
# 3. restricted Boltzmann machine 
# with periodic bonudary condition(PBC):

class RBMPBC1d(nn.Module):
    def __init__(self, nv:int, nh:int, dtype=tc.complex64, device='cpu', seed_num=1234):
        super().__init__()
        self.nv=nv
        self.nh=nh
        self.dtype = dtype
        self.device = device
        self.a = nn.Parameter(tc.randn(1, nh, dtype=dtype, device=device))
        nn.init.normal_(self.a,std=0.01)
        self.rbm_stack = nn.Linear(nv, nh, dtype=dtype, device=device)
        self.rbm_stack.apply(init_weights_pbc)
        params_info = [[list(self.a.size()), [0, nv]]]
        count_k = nv
        for params_k in self.rbm_stack.parameters():
            # the first element is the parameter's size:
            param_info_k = [list(params_k.size())]
            # the second element is the parameter's position:
            n_params_k = count_k + tc.numel(params_k)
            param_info_k.append([count_k, n_params_k])
            count_k = n_params_k
            params_info.append(param_info_k)
        self.params_info = params_info
    def forward(self, x):
        res = self.nv*tc.mul(x.sum(1,keepdim=True), self.a).sum(1,keepdim=True)
        for k in range(self.nv):
            x_new = x.roll(k,1)
            res += ((2*self.rbm_stack(x_new).cosh()).log()).sum(1,keepdim=True)
        return res   


###########################################
###########################################
### activation function ###
''' 
We define some common activation fucntions 
that doesn't exist in pytorch
'''

# the actiovation fucntion with polynomial expression
class act_func_poly1(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, z):
        # z .- z.^3/3 .+ z.^5 * 2/15
        return z - 1/3 * z**3 + 2/15 * z**5
    

###########################################
###########################################
### new idea ###

'''
Here, we insert the time t between input and the neural network
'''
###--------------------
###--------------------
# 1. feedforward neural network:

class FNNT(nn.Module):
    def __init__(self, t, layer_size:list, act_func, dtype=tc.complex64, device='cpu', seed_num = 1234):
        super().__init__()
        tc.random.manual_seed(seed_num)
        print("the model is on the ",device)
        self.dtype = dtype
        self.device = device
        self.t = t
        n_layers = len(layer_size) # total number of nn layers
        nn_models = []
        # generate a fnn struct except the last layer
        for m in range(0, n_layers - 2):
            nn_models.append(
            nn.Linear(layer_size[m], layer_size[m+1], dtype=dtype, device=device)
            )
            # generate the activation function
            # the name corresponds to the different activation functions:
            if act_func == 'poly1':
                nn_models.append(act_func_poly1())
            else:
                nn_models.append(act_func)
        # add the last layer:
        nn_models.append(nn.Linear(layer_size[-2], layer_size[-1], dtype=dtype, device=device))
        # construct fnn structure:
        self.linear_stack = nn.Sequential(*nn_models)
        self.linear_stack.apply(init_weights)

        # 2. record information of neurla network parameters:
        params_info = []
        count_k = 0
        for params_k in self.linear_stack.parameters():
            # the first element is the parameter's size:
            param_info_k = [list(params_k.size())]
            # the second element is the parameter's position:
            n_params_k = count_k + tc.numel(params_k)
            param_info_k.append([count_k, n_params_k])
            count_k = n_params_k
            params_info.append(param_info_k)
        self.params_info = params_info
    def forward(self, x):
            return self.linear_stack(x*self.t)
    
'''
Demo:
Autoregressive sampling via fnn
https://arxiv.org/pdf/1502.03509
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.124.020503

Here, we design a new structure of neural network to generate 
expression of wavefuntion by autoregressive sampling:

The last leyer has M * L neurons, where M is the all possible 
configuration for each site and L is the system size.
'''

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, dtype=tc.complex64, device='cpu'):
        # create a fully-connected linear layer
        super(MaskedLinear, self).__init__(in_features, out_features, bias, dtype=dtype, device=device)
        self.register_buffer('mask', tc.ones(out_features, in_features, dtype=self.weight.dtype, device=self.weight.device))

    def set_mask(self, mask):
        self.mask.data.copy_(mask.clone().detach().to(dtype=self.weight.dtype, device=self.weight.device))

    def forward(self, input):
        # Apply masked weight: x @ (W * M)^T
        # print(input.dtype, input.device)
        # print(self.weight.dtype, self.weight.device)
        # print(self.mask.dtype, self.mask.device)
        return F.linear(input, self.weight * self.mask, self.bias)


def create_mask(prev_deg, curr_deg, pos_layer='middle'):
    prev_deg = tc.tensor(prev_deg)
    curr_deg = tc.tensor(curr_deg)
    """
    Create M[j, k] = 1 if m_k <= m_j
    where:
        prev_deg: m_k (from layer l-1), shape [D_in]
        curr_deg: m_j (from layer l), shape [D_out]
    Return shape: [D_out, D_in]
    """
    if pos_layer=='middle':
        M = (prev_deg.unsqueeze(1) <= curr_deg.unsqueeze(0)).float()
    elif pos_layer=='last':
        M = (prev_deg.unsqueeze(1) < curr_deg.unsqueeze(0)).float()
    return M.T

class FNNAuto(nn.Module):
    def __init__(self, layer_size:list, dtype=tc.complex64, device='cpu', seed_num = 1234):
        super().__init__() 
        # test: 
        self.dtype=dtype
        self.device=device
        # layer_size = [4, 12, 9, 8]
        n_sites = layer_size[0]
        mask_index = []
        # input:
        mask_index.append([i for i in range(1, n_sites+1)])
        #print(mask_index[0])
        for l in range(1, len(layer_size)-1):
            n = int(layer_size[l] // (n_sites-1))
            mask_index_i = [i for i in range(1, n_sites) for _ in range(n)]
            #print(mask_index_i)
            mask_index.append(mask_index_i)
        
        '''
        before output, we add additional layer with 2L neurons,
        it can be viewed as [L,2], where each column represents 
        2 possible values for spin up and spin down states.
        '''
        mask_index.append([i for i in range(1, n_sites+1) for _ in range(2)])
        # print(mask_index[-1])
        # time.sleep(100)

        layers = []
        for l in range(len(layer_size)-2):
            in_features = layer_size[l]
            out_features = layer_size[l+1]

            mask = create_mask(mask_index[l], mask_index[l+1])
            layer = MaskedLinear(in_features, out_features, dtype=dtype, device=device)
            layer.set_mask(mask)
            layers.append(layer)
            layers.append(act_func_poly1())

        mask = create_mask(mask_index[-2], mask_index[-1],pos_layer='last')
        layer = MaskedLinear(layer_size[-2], layer_size[-1], dtype=dtype, device=device)
        layer.set_mask(mask)
        layers.append(layer)
        self.layers = layers
        self.net = nn.Sequential(*layers)

        # 2. record information of neurla network parameters:
        params_info = []
        count_k = 0
        for params_k in self.net.parameters():
            # the first element is the parameter's size:
            param_info_k = [list(params_k.size())]
            # the second element is the parameter's position:
            n_params_k = count_k + tc.numel(params_k)
            param_info_k.append([count_k, n_params_k])
            count_k = n_params_k
            params_info.append(param_info_k)
        self.params_info = params_info

    # def forward(self, x):
    #     return self.net(x)
    def forward(self, x):
        v = self.net(x)

        # x: [batch_size, L]
        x = x.unsqueeze(-1)
        # compute v: [batch_size, 2L] wit hall possible values fo spin up and spin down

        # change the shape of v to : [batch_size, L, 2], each row represents the 2 values of spin up and spin down for each site
        v = v.view(v.size(0),-1,2)
        #print(v)
        # map configuration as select index: x={+1,-1} -> s_index = {0,1}
        # meaning that when x_i = 1, we select v[i,0] as the value of input of spin up, otherwise when x_i = -1, we choose v[i,1] as the value of spin down
        # v_selected: [batch_size, L, 1]
        s_index = ((-x.real + 1) // 2).long() 
        v_selected = v.gather(dim=2,index=s_index)

        # compute sum_k |exp(v_{i,s_k})|^2
        # v_norm2: [batch_size, L, 1]
        v_norm2 = tc.sum(tc.abs(tc.exp(v))**2, dim=-1, keepdim=True)

        # the probability of ln psi is given by:
        # ln_psi_cd: [batch_size, L, 1]
        ln_psi_cd = v_selected - 0.5 * v_norm2.log()
        # print(ln_psi_cd)
        # ln_psi: [batch_size, 1]
        ln_psi = tc.sum(ln_psi_cd, dim=1)
        return ln_psi
    
    def autosampling(self, n_states, n_sites):
        x = tc.zeros(n_states, n_sites, dtype=self.dtype, device=self.device)
        for k in range(n_sites):
            # x : [batch_size, L]
            v = self.net(x)
            # v:[batch_size, 2L] -> [batch_size, L, 2]
            v = v.view(v.size(0),-1,2)
            # just choose the output of site k:
            # vk: [batch, 2] 
            vk = v[:, k, :]
            v_norm2 = tc.sum(tc.abs(tc.exp(vk))**2, dim=-1, keepdim=True)
            Pk = tc.abs(tc.exp(vk))**2 / v_norm2 # Pk: [batch_size, 2]
            #print('------')
            #print(Pk)
            #print(Pk.sum(dim=-1))
            r = tc.rand(n_states, device=Pk.device)
            #print('r')
            ##print(r)
            spin_k = tc.where(r < Pk[:, 0], tc.tensor(1, device=Pk.device), tc.tensor(-1, device=Pk.device))
            ##print('spin_k')
            #print(spin_k)
            x[:,k] = spin_k
        return x

