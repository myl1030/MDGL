# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 17:25:00 2022

@author: 86178
"""

import torch
import numpy as np
import torch.nn as nn
from einops import rearrange, repeat   
import torch.nn.functional as F


class LayerGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, epsilon=True):
        super().__init__()
        if epsilon: self.epsilon = nn.Parameter(torch.Tensor([[0.0]])) # assumes that the adjacency matrix includes self-loop
        else: self.epsilon = 0.0
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU())


    def forward(self, v, a):
        v_aggregate = torch.sparse.mm(a, v) 
        v_aggregate += self.epsilon * v # assumes that the adjacency matrix includes self-loop
        v_combine = self.mlp(v_aggregate)
        return v_combine


class ModuleMeanReadout(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, node_axis=1):
        return x.mean(node_axis), torch.zeros(size=[1,1,1], dtype=torch.float32) 


class ModuleSERO(nn.Module):
    def __init__(self, hidden_dim, input_dim, dropout=0.1, upscale=1.0):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(hidden_dim, round(upscale*hidden_dim)), nn.BatchNorm1d(round(upscale*hidden_dim)), nn.GELU()) 
        self.attend = nn.Linear(round(upscale*hidden_dim), input_dim) 
        self.dropout = nn.Dropout(dropout) 


    def forward(self, x, node_axis=1):
        # assumes shape [... x node x ... x feature]
        x_readout = x.mean(node_axis)
        x_shape = x_readout.shape
        x_embed = self.embed(x_readout.reshape(-1,x_shape[-1])) 
        x_graphattention = torch.sigmoid(self.attend(x_embed)).view(*x_shape[:-1],-1) 
        permute_idx = list(range(node_axis))+[len(x_graphattention.shape)-1]+list(range(node_axis,len(x_graphattention.shape)-1))
        x_graphattention = x_graphattention.permute(permute_idx) 
        return (x * self.dropout(x_graphattention.unsqueeze(-1))).mean(node_axis), x_graphattention.permute(1,0,2)



class ModuleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(input_dim, num_heads) 
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim))


    def forward(self, x):
        x_attend, attn_matrix = self.multihead_attn(x, x, x)
        x_attend = self.dropout1(x_attend) # no skip connection
        x_attend = self.layer_norm1(x_attend)
        x_attend2 = self.mlp(x_attend)
        x_attend = x_attend + self.dropout2(x_attend2)
        x_attend = self.layer_norm2(x_attend)
        return x_attend, attn_matrix

#### Dynamic Graph Learning(AAL Spatial Scale)
class MDGLAAL(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads, num_layers, sparsity, cls_token='sum', readout='sero', garo_upscale=1.0):
        super().__init__()
        assert cls_token in ['sum', 'mean', 'param']
        if cls_token=='sum': self.cls_token = lambda x: x.sum(0) 
        elif cls_token=='mean': self.cls_token = lambda x: x.mean(0)
        elif cls_token=='param': self.cls_token = lambda x: x[-1]
        else: raise
        if readout=='sero': readout_module = ModuleSERO
        elif readout=='mean': readout_module = ModuleMeanReadout
        else: raise

        self.token_parameter = nn.Parameter(torch.randn([num_layers, 1, 1, hidden_dim])) if cls_token=='param' else None 
        self.num_classes = num_classes
        self.sparsity = sparsity

        # define modules
        self.initial_linear = nn.Linear(input_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList()
        self.readout_modules = nn.ModuleList()
        self.transformer_modules = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
   

        for i in range(num_layers):
            self.gnn_layers.append(LayerGIN(hidden_dim, hidden_dim, hidden_dim))
            self.readout_modules.append(readout_module(hidden_dim=hidden_dim, input_dim=input_dim, dropout=0.1))
            self.transformer_modules.append(ModuleTransformer(hidden_dim, 2*hidden_dim, num_heads=num_heads, dropout=0.1))
            self.linear_layers.append(nn.Linear(hidden_dim, num_classes))


    def _collate_adjacency(self, a, sparse=True):
        i_list = []
        v_list = []
        for sample, _dyn_a in enumerate(a):
            for timepoint, _a in enumerate(_dyn_a):
                thresholded_a = (_a > np.percentile(_a.detach().cpu().numpy(), 100-self.sparsity))
                _i = thresholded_a.nonzero(as_tuple=False)
                _v = torch.ones(len(_i))
                _i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                i_list.append(_i)
                v_list.append(_v)
        _i = torch.cat(i_list).T.to(a.device)
        _v = torch.cat(v_list).to(a.device)

        return torch.sparse.FloatTensor(_i, _v, (a.shape[0]*a.shape[1]*a.shape[2], a.shape[0]*a.shape[1]*a.shape[3]))


    def forward(self, v1, a1, t1, sampling_endpoints1):
        # assumes shape [minibatch x time x node x feature] for v
        # assumes shape [minibatch x time x node x node] for a
        reg_ortho1 = 0.0
        latent_list = []
        minibatch_size, num_timepoints, num_nodes = a1.shape[:3]
       
        h = v1
        h = rearrange(h, 'b t n c -> (b t n) c')
        h = self.initial_linear(h)
        a1 = self._collate_adjacency(a1)
        
        for layer, (G, R, T, L) in enumerate(zip(self.gnn_layers, self.readout_modules, self.transformer_modules, self.linear_layers)):
            h = G(h, a1)
            h_bridge = rearrange(h, '(b t n) c -> t b n c', t=num_timepoints, b=minibatch_size, n=num_nodes)
            h_readout, node_attn = R(h_bridge, node_axis=2)
            if self.token_parameter is not None: h_readout = torch.cat([h_readout, self.token_parameter[layer].expand(-1,h_readout.shape[1],-1)])
            h_attend, time_attn = T(h_readout)
            ortho_latent = rearrange(h_bridge, 't b n c -> (t b) n c') 
            matrix_inner = torch.bmm(ortho_latent, ortho_latent.permute(0,2,1)) 
            reg_ortho1 += (matrix_inner/matrix_inner.max(-1)[0].unsqueeze(-1) - torch.eye(num_nodes, device=matrix_inner.device)).triu().norm(dim=(1,2)).mean() #eye生成对角线全1，其余部分全0的二维数组
                                                                                                              
            latent1 = self.cls_token(h_attend)
            latent_list.append(latent1)

        latent1 = torch.stack(latent_list, dim=1)
        latent1 = latent1.flatten(start_dim=1,end_dim=2) 

        
        return latent1,reg_ortho1

#### Dynamic Graph Learning(CC200 Spatial Scale)
class MDGLCC200(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads, num_layers, sparsity,
                 cls_token='sum', readout='sero', garo_upscale=1.0):
        super().__init__()
        assert cls_token in ['sum', 'mean', 'param']
        if cls_token == 'sum':
            self.cls_token = lambda x: x.sum(0)  
        elif cls_token == 'mean':
            self.cls_token = lambda x: x.mean(0)
        elif cls_token == 'param':
            self.cls_token = lambda x: x[-1]
        else:
            raise
        if readout == 'sero':
            readout_module = ModuleSERO
        elif readout == 'mean':
            readout_module = ModuleMeanReadout
        else:
            raise

        self.token_parameter = nn.Parameter(torch.randn([num_layers, 1, 1, hidden_dim])) if cls_token == 'param' else None  
        self.num_classes = num_classes
        self.sparsity = sparsity

        # define modules
        self.initial_linear = nn.Linear(input_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList()
        self.readout_modules = nn.ModuleList()
        self.transformer_modules = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        

        for i in range(num_layers):
            self.gnn_layers.append(LayerGIN(hidden_dim, hidden_dim, hidden_dim))
            self.readout_modules.append(readout_module(hidden_dim=hidden_dim, input_dim=input_dim, dropout=0.1))
            self.transformer_modules.append(
                ModuleTransformer(hidden_dim, 2 * hidden_dim, num_heads=num_heads, dropout=0.1))
            self.linear_layers.append(nn.Linear(hidden_dim, num_classes))

    def _collate_adjacency(self, a, sparse=True):
        i_list = []
        v_list = []
        for sample, _dyn_a in enumerate(a):
            for timepoint, _a in enumerate(_dyn_a):
                thresholded_a = (_a > np.percentile(_a.detach().cpu().numpy(), 100 - self.sparsity))
                _i = thresholded_a.nonzero(as_tuple=False)
                _v = torch.ones(len(_i))
                _i += sample * a.shape[1] * a.shape[2] + timepoint * a.shape[2]
                i_list.append(_i)
                v_list.append(_v)
        _i = torch.cat(i_list).T.to(a.device)
        _v = torch.cat(v_list).to(a.device)

        return torch.sparse.FloatTensor(_i, _v,
                                        (a.shape[0] * a.shape[1] * a.shape[2], a.shape[0] * a.shape[1] * a.shape[3]))

    def forward(self, v2, a2, t2, sampling_endpoints2):
        # assumes shape [minibatch x time x node x feature] for v
        # assumes shape [minibatch x time x node x node] for a
        reg_ortho2 = 0.0
        latent_list = []
        minibatch_size, num_timepoints, num_nodes = a2.shape[:3]
     
        h = v2
        h = rearrange(h, 'b t n c -> (b t n) c')
        h = self.initial_linear(h)
        a2 = self._collate_adjacency(a2)
     
        for layer, (G, R, T, L) in enumerate(zip(self.gnn_layers, self.readout_modules, self.transformer_modules, self.linear_layers)):
            
            h = G(h, a2)
            h_bridge = rearrange(h, '(b t n) c -> t b n c', t=num_timepoints, b=minibatch_size, n=num_nodes)
            h_readout, node_attn = R(h_bridge, node_axis=2)
            if self.token_parameter is not None: h_readout = torch.cat([h_readout, self.token_parameter[layer].expand(-1, h_readout.shape[1], -1)])
            h_attend, time_attn = T(h_readout)
            ortho_latent = rearrange(h_bridge, 't b n c -> (t b) n c')  
            matrix_inner = torch.bmm(ortho_latent, ortho_latent.permute(0, 2, 1))  
            reg_ortho2 += (matrix_inner / matrix_inner.max(-1)[0].unsqueeze(-1) - torch.eye(num_nodes,device=matrix_inner.device)).triu().norm(dim=(1, 2)).mean()  # eye生成对角线全1，其余部分全0的二维数组
                                                                                                               
            latent1 = self.cls_token(h_attend)
            latent_list.append(latent1)

        latent1 = torch.stack(latent_list, dim=1)
        latent1 = latent1.flatten(start_dim=1,end_dim=2)

       
        return latent1,reg_ortho2
    

#### Multi-scale Fusion and Classification
class Fusion(nn.Module):
    def __init__(self, input_dim1,input_dim2, hidden_dim, num_classes, num_heads, num_layers, sparsity, dropout=0.5,
                 cls_token='sum', readout='sero', garo_upscale=1.0):
        super(Fusion, self).__init__()
        
        self.out1 = MDGLAAL(input_dim1, hidden_dim, num_classes, num_heads, num_layers, sparsity, 
                 cls_token='sum', readout='sero', garo_upscale=1.0)
        self.out2 = MDGLCC200(input_dim2, hidden_dim, num_classes, num_heads, num_layers, sparsity, 
                 cls_token='sum', readout='sero', garo_upscale=1.0)
    
        self.fc2 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, v1, a1, t1, sampling_endpoints1,v2, a2, t2, sampling_endpoints2):
        
        out1,reg_ortho1 = self.out1(v1, a1, t1, sampling_endpoints1)
        out2,reg_ortho2 = self.out2(v2, a2, t2, sampling_endpoints2)
        x1 = torch.cat((out1, out2), dim=1)
        logits = self.dropout(self.fc2(x1))
        
        
        return logits,reg_ortho1,reg_ortho2
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)