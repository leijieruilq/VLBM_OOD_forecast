import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch
import os

class RevIN(nn.Module):
    def __init__(self, num_channels, num_nodes, eps = 1e-5, affine = True):
        super().__init__()
        self.num_channels = num_channels
        self.num_nodes = num_nodes
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, self.num_channels, self.num_nodes, 1))
            self.beta = nn.Parameter(torch.zeros(1, self.num_channels, self.num_nodes, 1))
        self.mean = None
        self.std = None

    def forward(self, x):
        self.mean = torch.mean(x, dim=-1, keepdim=True)
        self.std = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + self.eps)
        x_norm = (x - self.mean) / self.std
        if self.affine:
            x_norm = x_norm * self.gamma + self.beta
        return x_norm

    def reverse(self, x_pred_norm):
        if self.affine:
            x_no_affine = (x_pred_norm - self.beta) / (self.gamma + 1e-8) 
            x_denorm = x_no_affine * self.std + self.mean
        else:
            x_denorm = x_pred_norm * self.std + self.mean
        return x_denorm


class Embedding(nn.Module):
    def __init__(self, node, hidden, input_len, internal):
        super(Embedding, self).__init__()
        
        self.node_emb = nn.Parameter(torch.empty(node, hidden))
        nn.init.xavier_uniform_(self.node_emb)

        self.time_emb = nn.Parameter(torch.empty(int(1440/internal), hidden))
        nn.init.xavier_uniform_(self.time_emb)

        self.day_emb = nn.Parameter(torch.empty(7, hidden))
        nn.init.xavier_uniform_(self.day_emb)

        self.pos_emb = nn.Parameter(torch.empty(input_len, hidden))
        nn.init.xavier_uniform_(self.pos_emb)
        
        self.conv1 = nn.Conv2d(in_channels=input_len, out_channels=hidden, kernel_size=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(in_channels=hidden*4, out_channels=hidden, kernel_size=(1, 1), bias=True)

    def forward(self, x):
        batch, in_step, node, _ = x.size()
        # x: batch, step, node, hidden
        spatial_emb = self.node_emb.unsqueeze(0).expand(batch, -1, -1)       
        # spatial_emb: batch, node, hidden

        input_data = x[..., 0].unsqueeze(-1)
        # input_data: batch, step, node

        input_emb = self.conv1(input_data).squeeze(-1).transpose(1, 2)
        # input_emb_p: batch, node, hidden      
        
        day_id = x[..., 2]
        temporal_day_emb = self.day_emb[(day_id[:,-1,:]).type(torch.LongTensor)]
        # temporal_day_emb: batch, node, hidden
       
        time_id = x[..., 1]
        temporal_time_emb = self.time_emb[(time_id[:,-1,:]).type(torch.LongTensor)]
        # temporal_time_emb: batch, node, hidden

        p = torch.cat([temporal_time_emb, temporal_day_emb, input_emb, spatial_emb], dim=-1)
        p = p.transpose(1,2).unsqueeze(-1)
        p = self.conv2(p)
        # p: batch, hidden, node, 1

        # r: batch, step, node, hidden
        return p


class ST_Lpattern(nn.Module):
    def __init__(self, m,hidden_dim):
        super(ST_Lpattern, self).__init__()
        self.B = nn.Parameter(torch.empty(m, hidden_dim))
        nn.init.xavier_uniform_(self.B)
    def forward(self):
        return self.B


class Cross_GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, alpha=0.2):
        super(Cross_GAT, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.attn_fc = nn.Linear(2 * hidden_dim, 1)

        self.alpha = alpha
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        
        self.relu = nn.ReLU()
    def forward(self, x, y, adj):

        B, C, N, _ = x.size()

        x_emb = self.relu(self.conv1(x)).squeeze(-1).permute(0, 2, 1)
        y_emb = self.relu(self.conv2(y)).squeeze(-1).permute(0, 2, 1)

        # Expand for attention computation
        x_i = x_emb.unsqueeze(2).repeat(1, 1, N, 1)  # (B, N, N, hidden)
        y_j = y_emb.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, N, hidden)

        concat = torch.cat([x_i, y_j], dim=-1)  # (B, N, N, 2*hidden)
        e = self.leaky_relu(self.attn_fc(concat)).squeeze(-1)  # (B, N, N)

        # Apply mask with adjacency matrix
        adj_mask = (adj > 0).float()  # (1, N, N)
        attention = torch.where(adj_mask > 0, e, torch.tensor(-1e9, device=x.device))  # masked attention
        attention = torch.softmax(attention, dim=-1)  # (B, N, N)

        out = torch.bmm(attention, y_emb)  # (B, N, hidden)
        return out


class Q_Phi(nn.Module):
    def __init__(self, m, input_dim, hidden_dim):
        super(Q_Phi, self).__init__()
        self.l_pattern = ST_Lpattern(m,hidden_dim)
        self.m_size = m
        self.cross_gat = Cross_GAT(input_dim, hidden_dim)
        
        self.conv2 = nn.Conv2d(in_channels=hidden_dim*2, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.relu = nn.ReLU()

        self.w_mu = nn.Linear(hidden_dim, m)
        self.w_logvar = nn.Linear(hidden_dim, m)

    def forward(self, input, gt, adj):
        # input: batch, hidden, node, 1
        
        b = self.l_pattern() #(m, hidden)

        # Design a mechanism to generate latent z based on input x and ground truth y
        gat = self.cross_gat(input, gt, adj)

        # Calculate w for z = w * b
        mu_w = self.w_mu(gat)      # (B, N, m)
        logvar_w = self.w_logvar(gat)  # (B, N, m)
        std_w = torch.exp(0.5 * logvar_w)  # (B, N, m)

        # reparameterize: w = mu + std * eps
        eps = torch.randn_like(std_w)
        w = mu_w + eps * std_w  # (B, N, m)
        
        # z = w * B
        z = torch.matmul(w, b)  # (B, N, hidden)
        return z, mu_w, logvar_w
    

class P_Phi(nn.Module):
    def __init__(self, m, input_dim, hidden_dim):
        super(P_Phi, self).__init__()
        self.l_pattern = ST_Lpattern(m, hidden_dim)
        self.m_size = m
        self.cross_gat = Cross_GAT(input_dim, hidden_dim)

        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1))
        self.relu = nn.ReLU()

        self.w_mu = nn.Linear(hidden_dim, m)
        self.w_logvar = nn.Linear(hidden_dim, m)

    def forward(self, input, adj):
        B, _, N, _ = input.shape
        b = self.l_pattern()  # (m, hidden)

        # Use input twice to simulate self-attention mechanism
        gat = self.cross_gat(input, input, adj)  # (B, N, hidden)

        mu_w = self.w_mu(gat)      # (B, N, m)
        logvar_w = self.w_logvar(gat)  # (B, N, m)
        std_w = torch.exp(0.5 * logvar_w)

        eps = torch.randn_like(std_w)
        w = mu_w + std_w * eps

        z = torch.matmul(w, b)  # (B, N, hidden) Generate latent variable z 

        return z, mu_w, logvar_w # (B, N, hidden), (B, N, m), (B, N, m)

class Period(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Period, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)     
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data):
        hidden = self.relu(self.conv1(input_data))
        hidden = self.drop(hidden)
        hidden = self.conv2(hidden)      
        out = hidden + input_data
        return out
    

class Graph_Memory_Attn(nn.Module):
    def __init__(self, hidden_dim, top_k=10, alpha=0.2):
        super(Graph_Memory_Attn, self).__init__()
        self.top_k = top_k
        self.alpha = alpha  # Topological graph fusion ratio

        self.score_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z, topo=None):
        """
        z: Tensor (B, N, H)
        topo: Tensor (N, N) - Fixed topological adjacency matrix
        """
        B, N, H = z.shape

        # 1. Similarity calculation: inner product or cosine
        z_norm = F.normalize(z, dim=-1)  # (B, N, H)
        sim_matrix = torch.matmul(z_norm, z_norm.transpose(1, 2))  # (B, N, N)

        # 2. Fuse fixed topology
        if topo is not None:
            topo = topo.to(z.device).float()
            sim_matrix = (1 - self.alpha) * sim_matrix + self.alpha * topo

        # 3. Sparsity processing: keep only top-k
        topk = torch.topk(sim_matrix, self.top_k, dim=-1)
        mask = torch.zeros_like(sim_matrix).scatter(-1, topk.indices, 1.0)
        sparse_sim = sim_matrix * mask
        sparse_sim = sparse_sim.masked_fill(mask == 0, -1e9)
        # 4. softmax normalization
        A = F.softmax(sparse_sim, dim=-1)  # (B, N, N)

        return A  # Can be used as adjacency matrix input for GCN


class Residual(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, r, A):
        """
        r: Tensor (B, C, N, 1)
        A: Tensor (B, N, N)
        """
        B, C, N, _ = r.shape

        # reshape: (B, C, N, 1) -> (B, N, C)
        r_reshape = r.squeeze(-1).permute(0, 2, 1)  # (B, N, C)

        # Graph convolution operation: A @ r
        r_gcn = torch.bmm(A, r_reshape)  # (B, N, C)
        r_gcn = r_gcn.permute(0, 2, 1).unsqueeze(-1)  # (B, C, N, 1)

        hidden = self.relu(self.conv1(r_gcn))
        hidden = self.drop(hidden)
        hidden = self.conv2(hidden)

        out = hidden + r  # Residual connection
        return out

class Generator(nn.Module):
    def __init__(self, hidden, num_Period_Block, head, output_length):
        super(Generator, self).__init__()
        # Periodic blocks
        self.period = nn.ModuleList([Period(hidden, hidden) for _ in range(num_Period_Block)])
        # Use one Conv2d layer each for baseline prediction and residual branch
        self.base_pred = nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=1)
        self.res_pred  = nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=1)

        # Input normalization
        self.x_norm = nn.Conv2d(in_channels=hidden*2, out_channels=hidden, kernel_size=1)
        self.r_norm = nn.Conv2d(in_channels=hidden*2, out_channels=hidden, kernel_size=1)

        # Residual graph convolution blocks
        self.residual = nn.ModuleList([Residual(hidden, hidden) for _ in range(num_Period_Block)])
        self.gen_graph = Graph_Memory_Attn(hidden, top_k=head)

        self.prediction1 = nn.Conv2d(in_channels=hidden, out_channels=output_length, kernel_size=1)
        self.prediction2 = nn.Conv2d(in_channels=hidden, out_channels=output_length, kernel_size=1)

    def forward(self, z, x, topo):
        # z: (B, N, hidden), x: (B, hidden, N, 1)
        B, N, H = z.shape
        z_feat = z.permute(0,2,1).unsqueeze(-1)  # -> (B, H, N, 1)

        # 1) Historical residual input r
        r = self.r_norm(torch.cat([x - z_feat, x], dim=1))  # (B, H, N, 1)
        # 2) Regular input p
        p = self.x_norm(torch.cat([z_feat, x], dim=1))      # (B, H, N, 1)

        # 3) Stack periodic modules
        for net in self.period:
            p = net(p)

        # 4) Generate graph A
        A = self.gen_graph(z, topo)  # (B, N, N)

        # 5) Residual graph convolution modules
        for net in self.residual:
            r = net(r, A)

        Y_pred = p + r
        # 6) Baseline and residual predictions
        Y_base = self.prediction1(self.base_pred(p))  # (B, T, N, 1)
        R_hat  = self.prediction2(self.res_pred(r))   # (B, T, N, 1)
        # 7) Final summation
        return Y_pred, Y_base, R_hat

class Model(nn.Module):
    def __init__(self,configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.inp_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.hidden_dim = configs.hiddens
        self.num_Period_Block = configs.num_Period_Block
        self.head = configs.heads
        self.m = configs.m
        self.interval = configs.interval
        self.x_embedding = Embedding(configs.enc_in, self.hidden_dim, self.inp_len, self.interval)
        self.y_embedding = Embedding(configs.enc_in, self.hidden_dim, self.pred_len, self.interval)
        self.q_phi = Q_Phi(self.m, self.hidden_dim, self.hidden_dim)
        self.p_phi = P_Phi(self.m, self.hidden_dim, self.hidden_dim)
        self.generator = Generator(self.hidden_dim, self.num_Period_Block, self.head, self.pred_len)
        self.prediction = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.pred_len, kernel_size=1)
        self.use_norm = configs.use_norm

    def encoder(self, x, y):
        # x: (B, T, N, C), y: same, adj: (B, N, N)
        B, T, N, C = x.size()
        adj = torch.eye(N).to(x.device).unsqueeze(0).expand(B, -1, -1)
        if self.use_norm:
            # Perform instance normalization only on the 0-th channel
            x_enc_0 = x[..., 0:1]  # Extract 0-th channel and keep dimension (B, T, N, 1)
            # Calculate mean and std (along the time dimension)
            means = x_enc_0.mean(dim=1, keepdim=True).detach()
            x_enc_0 = x_enc_0 - means
            stdev = torch.sqrt(torch.var(x_enc_0, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc_0 = x_enc_0 / stdev
            # Put normalized 0-th channel back to original position
            x = torch.cat([x_enc_0, x[..., 1:]], dim=-1)
        
        # Embedding
        x_emb = self.x_embedding(x)  # (B, hidden, N, 1)
        
        # Prior and Posterior
        z_p, mu_p, logvar_p = self.p_phi(x_emb, adj) # (B, N, hidden), (B, N, m), (B, N, m)
        if self.training:
            y_emb = self.y_embedding(y) # (B, hidden, N, 1)
            z_q, mu_q, logvar_q = self.q_phi(x_emb, y_emb, adj) # (B, N, hidden), (B, N, m), (B, N, m)
        else:
            z_q = None; mu_q = logvar_q = None
        
        # Generator output
        Y_pred, Y_base, R_hat = self.generator(z_p, x_emb, adj)
        
        # Project back to prediction dimension
        out = self.prediction(Y_pred).squeeze(-1)  # (B, output_length, N)
        if self.use_norm:
            out = out * (stdev[:, 0, :, 0].unsqueeze(1).repeat(1, out.shape[1], 1))
            out = out + (means[:, 0, :, 0].unsqueeze(1).repeat(1, out.shape[1], 1))
        return {
            'z_q': z_q, 'mu_q': mu_q, 'logvar_q': logvar_q,
            'z_p': z_p, 'mu_p': mu_p, 'logvar_p': logvar_p,
            'Y_pred': out, 'Y_base': Y_base.squeeze(-1), 'R_hat': R_hat.squeeze(-1)
        }
    def forecast(self,x_enc, y_enc):
        return self.encoder(x_enc, y_enc)
    def forward(self, x_enc, y_enc):
            dec_out = self.forecast(x_enc,y_enc)
            return dec_out