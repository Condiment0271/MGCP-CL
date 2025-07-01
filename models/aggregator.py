import random

import kmeans_pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from util import config
from kmeans_pytorch import kmeans


class GCN(nn.Module):

    def __init__(self, in_dim, out_dim, act, p):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

    def forward(self, g, h):
        h = self.drop(h)
        h = torch.matmul(g, h)
        h = self.proj(h)
        h = self.act(h)
        return h

def select_center_node(scores, weights, indices):
    # 计算加权平均值
    if indices.dim() == 0:
        return indices
    else:
        weighted_scores = scores * weights
        weighted_mean = torch.sum(weighted_scores) / torch.sum(weights)

        # 找到与加权平均值最接近的节点
        closest_indices = torch.argsort(torch.abs(scores - weighted_mean))

        # 选择最接近加权平均值的节点
        #print(closest_indices)
        center_index = closest_indices[0]

        # 如果有多个节点与加权平均值相等，选择得分较大的节点
        #if torch.abs(scores[1] - weighted_mean) == torch.abs(scores[center_index] - weighted_mean):
        #    for idx in closest_indices[1:]:
        #        if torch.abs(scores[idx] - weighted_mean) == torch.abs(scores[center_index] - weighted_mean) and scores[
        #            idx] > scores[center_index]:
        #            center_index = idx
        #            break
        return indices[center_index]

def cluster_nodes(scores, h, g, k):
    # 使用 KMeans 聚类算法将节点表示聚类成 k 类
    #print(scores.size())
    num_nodes = g.shape[1]
    if k*num_nodes < 1:
        return g, h
    num_centers = int(k*num_nodes)
    new_scores_list = []
    new_h_list = []
    new_g_list = []
    #print(h.size())
    #if len(h.size()) == 2:
    #    scores = torch.unsqueeze(1,dim=0)
    #    h = torch.unsqueeze(1,dim=0)
    #    g = torch.unsqueeze(1,dim=0)
    if h.size(0) == 1:
        kernel = KMeans(n_clusters=num_centers)
        scores = scores.cpu().detach().numpy().reshape(-1, 1)
        clusters = kernel.fit_predict(scores)
        scores = torch.tensor(scores)
        clusters = torch.tensor(clusters)
        #.reshape(-1, 1).cpu().detach().numpy()
        center_index = []
        for cluster_id in range(num_centers):
            # 找到当前类别的所有节点索引
            cluster_indices = torch.nonzero(clusters == cluster_id).squeeze().long()
            if cluster_indices.numel() == 0:
                median_scores = torch.median(scores)
                median_scores_index = torch.where(scores == median_scores)[0]
                if median_scores_index.size(0) > 1:
                    median_scores_index = median_scores_index[0]
                center_index.append(median_scores_index)
            else:
                weights = F.sigmoid(scores[cluster_indices])
                center_index.append(select_center_node(scores[cluster_indices], weights, cluster_indices))
        #print(h.size(), center_index)
        new_h = h[:, center_index, :]
        new_g = g[:, center_index, :]
        new_g = new_g[:,:, center_index]

        #new_h_list.append(new_h_simple)
        #new_g_list.append(new_g_simple)

        #new_h = torch.stack(new_h_list)
        #new_g = torch.stack(new_g_list)

        return new_g, new_h

    else:
        for scores_simple, h_simple, g_simple in zip(scores, h, g):
            kernel = KMeans(n_clusters=num_centers)
            scores_simple = scores_simple.cpu().detach().numpy().reshape(-1, 1)
            clusters = kernel.fit_predict(scores_simple)
            scores_simple = torch.tensor(scores_simple)
            clusters = torch.tensor(clusters)
            center_index = []
            median_scores = torch.median(scores)
            for cluster_id in range(num_centers):
                # 找到当前类别的所有节点索引
                cluster_indices = torch.nonzero(clusters == cluster_id).squeeze().long()
                if cluster_indices.numel() == 0:
                    center_index.append(median_scores)
                else:
                    weights = F.sigmoid(scores_simple[cluster_indices])
                    center_index.append(select_center_node(scores_simple[cluster_indices], weights, cluster_indices))
            #print(h_simple.size(), center_index)
            new_h_simple = h_simple[center_index, :]
            new_g_simple = g_simple[center_index, :]
            new_g_simple = new_g_simple[ :, center_index]

            new_h_list.append(new_h_simple)
            new_g_list.append(new_g_simple)

        new_h = torch.stack(new_h_list)
        new_g = torch.stack(new_g_list)

        return new_g, new_h


class Pool(nn.Module):

    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        #scores = self.sigmoid(weights)
        scores = weights

        #return top_k_graph(scores, g, h, self.k)
        return cluster_nodes(scores, h, g, self.k)

def top_k_graph(scores, g, h, k):
    num_nodes = g.shape[1]
    values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
    #print(scores.size(),idx.size(), k, num_nodes)
    if len(idx.size()) > 1:
        new_h = h[:, idx[1], :]
        values = torch.unsqueeze(values, -1)
        new_h = torch.mul(new_h, values)
        un_g = g.bool().float()
        un_g = torch.matmul(un_g, un_g).bool().float()
        un_g = un_g[:, idx[1], :]
        un_g = un_g[:, :, idx[1]]
        g = norm_g(un_g)
    else:
        new_h = h[:,idx, :]
        values = torch.unsqueeze(values, -1)
        new_h = torch.mul(new_h, values)
        un_g = g.bool().float()
        un_g = torch.matmul(un_g, un_g).bool().float()
        un_g = un_g[:,idx, :]
        un_g = un_g[:,:, idx]
        g = norm_g(un_g)
    return g, new_h

def norm_g(g):
    for i in g:
        degrees = torch.sum(i, 1)
        i = i / degrees
    return g

class AttentionAggregator(nn.Module):
    def __init__(self, hidden_size, input_dim=128,):
        super(AttentionAggregator, self).__init__()
        self.attn = nn.Linear(hidden_size, 1, bias=False)
        self.ks = [0.3]
        self.drop_p = 0.3
        self.down_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(len(self.ks)):
            self.down_gcns.append(GCN(input_dim, input_dim, nn.ELU(), self.drop_p))
            self.pools.append(Pool(self.ks[i], input_dim, self.drop_p))
        self.bottom_gcn = GCN(input_dim, input_dim, nn.ELU(), self.drop_p)
    def get_attn(self, reps, adj, mask=None):
        attn_scores = self.attn(reps).squeeze(2)
        if mask is not None:
            attn_scores = attn_scores + mask
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(2)  # (batch_size, len, 1)
        reps_attn = reps * attn_weights

        for i in range(len(self.ks)):
            if reps_attn.size()[1] > 2:
                reps_attn = self.down_gcns[i](adj, reps_attn)
                #adj, reps_attn, idx = self.pools[i](adj, reps_attn)
                adj, reps_attn = self.pools[i](adj, reps_attn)
        reps_attn = self.bottom_gcn(adj, reps_attn)

        #norm = torch.norm(reps_attn, p=2, dim=2, keepdim=True)
        #norm_weighted_reps = reps_attn / norm
        #attn_out = torch.mean(norm_weighted_reps, dim=1)
        attn_out = torch.mean(reps_attn, dim=1)  # (batch_size, hidden_dim)

        return attn_out, attn_weights

    def forward(self, reps, adj, mask=None):
        attn_out, attn_weights = self.get_attn(reps, adj, mask)

        return attn_out, attn_weights  # (batch_size, hidden_dim), (batch_size, len, 1)


class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, reps):
        return reps.mean(1)


class MaxAggregator(nn.Module):
    def __init__(self):
        super(MaxAggregator, self).__init__()

    def forward(self, reps):
        return torch.max(reps, dim=1)


class MultimodalGraphReadout(nn.Module):
    def __init__(self, m_dim, readout_t, readout_v, readout_a):
        super(MultimodalGraphReadout, self).__init__()
        self.readout_t = readout_t
        self.readout_v = readout_v
        self.readout_a = readout_a
        self.project_m = nn.Linear(m_dim, m_dim)

    def forward(self, hs_gnn, mask, adj_t, adj_a, adj_v):
        hs_t_, hs_v_, hs_a_ = torch.split(hs_gnn, hs_gnn.size(1)//3, dim=1)
        reps_t_, _ = self.readout_t(hs_t_, adj_t, mask)
        reps_v_, _ = self.readout_v(hs_v_, adj_v, mask)
        reps_a_, _ = self.readout_a(hs_a_, adj_a, mask)
        #print(reps_t_.size())
        reps_m = F.relu(self.project_m(torch.cat([reps_t_, reps_v_, reps_a_], dim=-1)))

        return reps_m