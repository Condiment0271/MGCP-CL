import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel
from models.gat import GAT
from models.aggregator import MeanAggregator, MaxAggregator, AttentionAggregator, MultimodalGraphReadout
import numpy as np
from sklearn.neighbors import NearestNeighbors
from util import config
from kmeans_pytorch import kmeans
from deslib.util.faiss_knn_wrapper import FaissKNNClassifier


def get_padding_mask(lengths, max_len=None):
    bsz = len(lengths)
    if not max_len:
        max_len = lengths.max()
    mask = torch.zeros((bsz, max_len))
    for i in range(bsz):
        index = torch.arange(int(lengths[i].item()), max_len)
        mask[i] = mask[i].index_fill_(0, index, -1e9)

    return mask


def KNN_Graph_to_adj(h):

    neighbors = config['knn_neighbors']
    # 定义 KNN 模型
    if h.size(1)*2//3 < neighbors:
        neighbors = h.size(1)*2//3
    '''
    if h.size(0) == 1:
        print(h.shape)
        num_nodes, vector_dim = h.shape

        # 将图结构重塑为二维数组以适应 KNN 模型
        # graph_reshaped = h_simple.reshape(batch_size * num_nodes, vector_dim)

        # 训练 KNN 模型
        knn_model.fit(h.cpu().detach().numpy())

        distances, indices = knn_model.kneighbors(h.cpu().detach().numpy(), n_neighbors=4)

        # 构建邻接矩阵
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(len(indices)):
            for j in range(1, len(indices[i])):
                adjacency_matrix[i, indices[i, j]] = 1

        # 输出邻接矩阵
        return adjacency_matrix
    else:
    '''
    # 遍历每个批次中的每个图结构
    h_adj = []
    for h_simple in h:
        num_nodes, vector_dim = h_simple.shape
        step = int(num_nodes/3)
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(step):
            adjacency_matrix[i, i + step] = 1
            adjacency_matrix[i + step, i] = 1
            adjacency_matrix[i, i + 2*step] = 1
            adjacency_matrix[i + 2*step, i] = 1
            adjacency_matrix[i + step, i + 2*step] = 1
            adjacency_matrix[i + 2*step, i + step] = 1
        #adjacency_matrix = torch.tensor(adjacency_matrix)
        for modality_index in range(3):
            # 获取当前模态的数据
            #.cpu().detach().numpy()
            single_modality = h_simple[modality_index * step: (modality_index + 1) * step, :]
            #single_modality = torch.tensor(single_modality).cuda()
            single_modality = single_modality.cpu().detach().numpy()
            other_modality = torch.cat((h_simple[:modality_index * step, :],h_simple[(modality_index + 1) * step:, :]), dim=0)
            #other_modality = torch.tensor(other_modality).cuda()
            other_modality = other_modality.cpu().detach().numpy()
            # 训练 KNN 模型
            knn_model = NearestNeighbors(n_neighbors=neighbors)
            #knn_model = FaissKNNClassifier(n_neighbors=neighbors)
            knn_model.fit(other_modality)

            distances, indices = knn_model.kneighbors(single_modality)
            distances = torch.tensor(distances)
            indices = torch.tensor(indices)
            # 构建当前模态的邻接矩阵
            for i in range(len(indices)):
                for j in range(neighbors):
                    source_index = i + modality_index * step
                    target_index = indices[i, j]
                    #print(source_index)
                    adjacency_matrix[source_index, target_index] = 1
        #knn_model.fit(h_simple.cpu().detach().numpy())

        #distances, indices = knn_model.kneighbors(h_simple.cpu().detach().numpy(), n_neighbors=neighbors)

        # 构建邻接矩阵

        #for i in range(len(indices)):
        #    for j in range(1, len(indices[i])):
        #        adjacency_matrix[i, indices[i, j]] = 1

        h_adj.append(torch.from_numpy(adjacency_matrix))

    h_adj_mul = torch.stack(h_adj)
    return h_adj_mul



class MultimodalGraphFusionNetwork(nn.Module):
    def __init__(self, config):
        super(MultimodalGraphFusionNetwork, self).__init__()
        dt, da, dv = config["t_size"], config["a_size"], config["v_size"]
        h = config["hidden_size"]
        m_dim = 3*h

        self.config = config
        self.h = h
        self.encoder_t = self._get_encoder(modality='t')
        self.encoder_v = self._get_encoder(modality='v')
        self.encoder_a = self._get_encoder(modality='a')

        self.gat_t = GAT(input_dim=h,
                        gnn_dim=h // config["num_gnn_heads"],
                        num_layers=config["num_gnn_layers"],
                        num_heads=config["num_gnn_heads"],
                        dropout=config["dropout_gnn"],
                        leaky_alpha=0.2)
        self.gat_v = GAT(input_dim=h,
                         gnn_dim=h // config["num_gnn_heads"],
                         num_layers=config["num_gnn_layers"],
                         num_heads=config["num_gnn_heads"],
                         dropout=config["dropout_gnn"],
                        leaky_alpha=0.2)
        self.gat_a = GAT(input_dim=h,
                         gnn_dim=h // config["num_gnn_heads"],
                         num_layers=config["num_gnn_layers"],
                         num_heads=config["num_gnn_heads"],
                         dropout=config["dropout_gnn"],
                        leaky_alpha=0.2)
        self.gat_m = GAT(input_dim=h,
                       gnn_dim=h // config["num_gnn_heads"],
                       num_layers=config["num_gnn_layers"],
                       num_heads=config["num_gnn_heads"],
                       dropout=config["dropout_gnn"],
                        leaky_alpha=0.2)

        self.project_t = nn.Linear(dt, h)
        self.project_v = nn.Linear(dv*2, h)
        self.project_a = nn.Linear(da*2, h)

        self.readout_t = AttentionAggregator(h)
        self.readout_v = AttentionAggregator(h)
        self.readout_a = AttentionAggregator(h)
        #self.readout_m = AttentionAggregator(h)
        self.readout_m = MultimodalGraphReadout(m_dim, self.readout_t, self.readout_v, self.readout_a)

        self.fc_out_mul = nn.Linear(m_dim, 1)
        self.fc_out_uni = nn.Linear(h, 1)
        self.dropout_mul = nn.Dropout(config["dropout"])
        self.dropout_uni = nn.Dropout(config["dropout"])
        self.dropout_t = nn.Dropout(config["dropout_t"])
        self.dropout_v = nn.Dropout(config["dropout_v"])
        self.dropout_a = nn.Dropout(config["dropout_a"])

    def _get_encoder(self, modality='t', *args):
        if modality == 't':
            return BertModel.from_pretrained(self.config["bert_path"])
        elif modality == 'v':
            return nn.LSTM(self.config["v_size"], self.config["v_size"], batch_first=True, bidirectional=True)
        elif modality == 'a':
            return nn.LSTM(self.config["a_size"], self.config["a_size"], batch_first=True, bidirectional=True)
        else:
            raise ValueError('modality should be t or v or a!')

    def _lstm_encode(self, inputs, lengths, lstm, h_size):
        batch_size, t, _ = inputs.size()
        h0 = torch.zeros(2, batch_size, h_size).to(inputs.device)
        c0 = torch.zeros(2, batch_size, h_size).to(inputs.device)

        pack = pack_padded_sequence(inputs, lengths.cpu(), batch_first=True)
        lstm.flatten_parameters()
        out, _ = lstm(pack, (h0, c0))
        out, lens = pad_packed_sequence(out, batch_first=True)

        memory = out.contiguous().view(batch_size * t, -1)
        index = lens - 1 + torch.arange(batch_size) * t
        if torch.cuda.is_available():
            index = index.cuda()

        last_h = torch.index_select(memory, 0, index)

        return out, last_h

    def _delete_edge(self, adj, ratio=0.1):
        bsz = adj.size(0)
        adj_del = adj.clone().contiguous().view(bsz, -1)
        for i in range(bsz):
            edges = torch.where(adj_del[i] == 1)[0]
            del_num = math.ceil(len(edges) * ratio)
            del_edges = random.sample(edges.cpu().numpy().tolist(), del_num)
            adj_del[i, del_edges] = 0
        adj_del = adj_del.reshape_as(adj)
        return adj_del

    def _add_edge(self, adj, ratio=0.1):
        bsz = adj.size(0)
        adj_add = adj.clone().contiguous().view(bsz, -1)
        for i in range(bsz):
            non_edges = torch.where(adj_add[i] == 1)[0]
            add_num = math.ceil(len(non_edges) * ratio)
            add_edges = random.sample(non_edges.cpu().numpy().tolist(), add_num)
            adj_add[i, add_edges] = 0
        adj_add = adj_add.reshape_as(adj)
        return adj_add

    def _delete_node(self, reps, adj, ratio=0.1):
        bsz, n, _ = adj.size()
        reps_del = reps.clone()
        adj_del = adj.clone()
        for i in range(bsz):
            del_num = math.ceil(n * ratio)
            del_nodes = random.sample(list(range(n)), del_num)
            adj_del[i, del_nodes, :] = 0
            adj_del[i, :, del_nodes] = 0
            reps_del[i, del_nodes] = 0
        return reps_del, adj_del

    def forward(self, text_tensor=None, video_tensor=None, audio_tensor=None, lengths=None,
                bert_sent_type=None, bert_sent_mask=None, adj_matrix=None):
        bsz, max_len, _ = video_tensor.size()

        # get padding mask
        mask = torch.zeros((bsz, max_len))
        for i in range(bsz):
            index = torch.arange(int(lengths[i].item()), max_len)
            mask[i] = mask[i].index_fill_(0, index, -1e9)
        mask = mask.to(text_tensor.device)

        # get unimodal adj
        adj_matrix_t = adj_matrix[:, :max_len, :max_len]
        adj_matrix_v = adj_matrix[:, max_len:2*max_len, max_len:2*max_len]
        adj_matrix_a = adj_matrix[:, 2*max_len:3*max_len, 2*max_len:3*max_len]

        # encode
        bert_output = self.encoder_t(input_ids=text_tensor,
                                     # token_type_ids=bert_sent_type,
                                     attention_mask=bert_sent_mask)
        hs_t = bert_output[0][:, 1:-1]
        hs_t = F.relu(self.project_t(hs_t))

        hs_v, last_v = self._lstm_encode(video_tensor, lengths, self.encoder_v, video_tensor.size(-1))
        hs_v = F.relu(self.project_v(hs_v))

        hs_a, last_a = self._lstm_encode(audio_tensor, lengths, self.encoder_a, audio_tensor.size(-1))
        hs_a = F.relu(self.project_a(hs_a))

        # multimodal graph
        hs = torch.cat([hs_t, hs_v, hs_a], dim=1)
        #h_mul_t, h_mul_v, h_mul_a = hierarchical_clustering(hs)
        adj_matrix_mul = KNN_Graph_to_adj(hs)

        hs_gnn, _ = self.gat_m(hs, adj_matrix_mul)
        hs_gnn = F.relu(hs_gnn + hs)
        reps_mul = self.readout_m(hs_gnn, mask, adj_matrix_t, adj_matrix_a, adj_matrix_v)

        # unimodal graphs
        hs_t_gnn, _ = self.gat_t(hs_t, adj_matrix_t)
        hs_v_gnn, _ = self.gat_v(hs_v, adj_matrix_v)
        hs_a_gnn, _ = self.gat_a(hs_a, adj_matrix_a)
        hs_t_gnn = F.relu(hs_t_gnn + hs_t)
        hs_v_gnn = F.relu(hs_v_gnn + hs_v)
        hs_a_gnn = F.relu(hs_a_gnn + hs_a)
        reps_t, _ = self.readout_t(hs_t_gnn, adj_matrix_t, mask)
        reps_v, _ = self.readout_v(hs_v_gnn, adj_matrix_v, mask)
        reps_a, _ = self.readout_a(hs_a_gnn, adj_matrix_a, mask)

        reps_uni = torch.cat([reps_t, reps_v, reps_a], dim=-1)

        reps_mul_arr = reps_mul.cpu().detach().numpy()
        file_path = "save/CL_features.txt"
        with open(file_path, 'a') as f:
            for row in reps_mul_arr:
                row_str = ' '.join([str(val) for val in row])
                f.write(row_str + '\n')

        output_mul = self.fc_out_mul(self.dropout_mul(reps_mul))
        output_mul_it = output_mul.cpu().detach().numpy()
        #output_mul_it = output_mul_it.item()
        file_path = "save/CL_features_labels.txt"
        with open(file_path, 'a') as f:
            for result in output_mul_it:
                f.write(str(result[0]) + '\n')
        # augmentation
        adj_matrix_aug1 = self._delete_edge(adj_matrix_mul, self.config["aug_ratio"])
        adj_matrix_aug1 = self._add_edge(adj_matrix_aug1, self.config["aug_ratio"])
        hs_gnn_aug1, _ = self.gat_m(hs, adj_matrix_aug1)
        hs_gnn_aug1 = F.relu(hs_gnn_aug1 + hs)
        reps_m_aug1 = self.readout_m(hs_gnn_aug1, mask, adj_matrix_t,adj_matrix_a,adj_matrix_v)

        adj_matrix_t_aug1 = self._delete_edge(adj_matrix_t, self.config["aug_ratio"])
        adj_matrix_v_aug1 = self._delete_edge(adj_matrix_v, self.config["aug_ratio"])
        adj_matrix_a_aug1 = self._delete_edge(adj_matrix_a, self.config["aug_ratio"])
        adj_matrix_t_aug1 = self._add_edge(adj_matrix_t_aug1, self.config["aug_ratio"])
        adj_matrix_v_aug1 = self._add_edge(adj_matrix_v_aug1, self.config["aug_ratio"])
        adj_matrix_a_aug1 = self._add_edge(adj_matrix_a_aug1, self.config["aug_ratio"])
        hs_t_gnn_aug1, _ = self.gat_t(hs_t, adj_matrix_t_aug1)
        hs_v_gnn_aug1, _ = self.gat_v(hs_v, adj_matrix_v_aug1)
        hs_a_gnn_aug1, _ = self.gat_a(hs_a, adj_matrix_a_aug1)
        hs_t_gnn_aug1 = F.relu(hs_t_gnn_aug1 + hs_t)
        hs_v_gnn_aug1 = F.relu(hs_v_gnn_aug1 + hs_v)
        hs_a_gnn_aug1 = F.relu(hs_a_gnn_aug1 + hs_a)
        reps_t_aug1, _ = self.readout_t(hs_t_gnn_aug1, adj_matrix_t, mask)
        reps_v_aug1, _ = self.readout_v(hs_v_gnn_aug1, adj_matrix_v, mask)
        reps_a_aug1, _ = self.readout_a(hs_a_gnn_aug1, adj_matrix_a, mask)

        reps_t_aug = torch.stack([reps_t_aug1, reps_t], dim=1)
        reps_v_aug = torch.stack([reps_v_aug1, reps_v], dim=1)
        reps_a_aug = torch.stack([reps_a_aug1, reps_a], dim=1)
        reps_m_aug = torch.stack([reps_m_aug1, reps_mul], dim=1)

        return output_mul.view(-1), F.normalize(reps_uni, dim=-1), F.normalize(reps_mul, dim=-1), F.normalize(reps_mul.unsqueeze(1), dim=-1), F.normalize(reps_t.unsqueeze(1), dim=-1), \
               F.normalize(reps_v.unsqueeze(1), dim=-1), F.normalize(reps_a.unsqueeze(1), dim=-1),\
               F.normalize(reps_m_aug, dim=-1), F.normalize(reps_t_aug, dim=-1), F.normalize(reps_v_aug, dim=-1),\
               F.normalize(reps_a_aug, dim=-1)
