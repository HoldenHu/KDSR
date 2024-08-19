import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import RGATLayer, GCNLayer, SAGELayer, GATLayer, KVAttentionLayer, HeteAttenLayer
from pooler import BatchedDiffPool
from torch.nn import Module, Parameter
import torch.nn.functional as F

import torch.nn.init as init
from torch.nn.init import xavier_uniform_, xavier_normal_

from utils import trans_to_cpu, trans_to_cuda
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances


class KDSR(Module):
    def __init__(self, config, iid_image_feature, iid_text_feature):
        '''
        iid_image_feature: [V_N D]
        iid_text_feature: [V_N D]
        node_embedding: [V_N+I_N+T_N D]
        '''
        super(KDSR, self).__init__()
        self.config = config
        self.batch_size = config['batch_size']
        self.num_item = config['num_node'][config['dataset']]
        self.dim = config['embedding_size']
        self.auxiliary_info = config['auxiliary_info']
        self.dropout_output = config['dropout_output']
        self.dropout_atten = config['dropout_atten']
        self.n_layer = config['n_layer']
        self.layer_norm_eps = 1e-12 # TODO
        self.last_n = config['last_n']
        self.cor_type = config['cor_type'] # cos/euc/dot/man
        self.cor_code_num = config['cor_code_num'] # K
        self.product_domain_D = config['product_domain_D'] # D
        self.quantization_method = config['quantization_method'] # 'kmeans'/'vq'
        self.tau = config['tau']
        
        self.v_s_weight = config['v_s_weight']
        self.v_c_weight = config['v_c_weight']
        self.t_s_weight = config['t_s_weight']
        self.t_c_weight = config['t_c_weight']

        self.use_CD = config['use_CD']
        self.use_CCA = config['use_CCA']

        # self.aggregator = config['aggregator']
        self.max_relid = 4

        # correlation calculation
        self.img_emb_project = nn.Linear(self.dim, self.dim)
        self.txt_emb_project = nn.Linear(self.dim, self.dim)
        self.img_cor_project = nn.Linear(self.dim, self.dim)
        self.txt_cor_project = nn.Linear(self.dim, self.dim)

        self.img_cor_param_layer = nn.Linear(self.dim, self.dim)
        self.img_cor_prediction_layer = nn.Linear(self.dim, self.cor_code_num)
        self.txt_cor_param_layer = nn.Linear(self.dim, self.dim)
        self.txt_cor_prediction_layer = nn.Linear(self.dim, self.cor_code_num)

        # Hierachical Aggregator 
        # self.local_agg_img = RGATLayer(self.dim, self.max_relid, self.config['alpha'], dropout=self.dropout_atten)
        # self.local_agg_txt = RGATLayer(self.dim, self.max_relid, self.config['alpha'], dropout=self.dropout_atten)
        self.local_agg_item = RGATLayer(self.dim, self.max_relid, self.config['alpha'], dropout=self.dropout_atten)
        
        # Modality Gate Fusion
        self.modality_projection = nn.Sequential(nn.Linear(self.dim, self.dim), nn.ReLU(True), nn.Linear(self.dim, 1))
        
        # Item representation & Position representation
        self.embedding = nn.Embedding(self.num_item, self.dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(150, self.dim)
        # self.node_type_embedding = nn.Embedding(4, self.dim)

        # Parameters
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_pos_type = nn.Parameter(torch.Tensor((len(self.auxiliary_info)+1) * self.dim, self.dim))

        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)

        self.projection = nn.Sequential(nn.Linear(self.dim, self.dim), nn.ReLU(True), nn.Linear(self.dim, 1)) # for gate
        self.fusion_layer = nn.Linear(self.dim*3, self.dim)

        self.dropout_out = nn.Dropout(self.dropout_output)
        # self.LayerNorm = nn.LayerNorm(self.dim, eps=self.layer_norm_eps) # can be used in merging pos embedding and hidden

        self.loss_function = nn.CrossEntropyLoss()

        self.apply(self._init_weights)
        init.normal_(self.w_1, mean=0.0, std=0.02)
        init.normal_(self.w_2, mean=0.0, std=0.02)
        init.normal_(self.w_pos_type, mean=0.0, std=0.02)
        # innitialize embedding
        if self.use_CCA:
            from sklearn.cross_decomposition import CCA
            cca = CCA(n_components=128)
            cca.fit(iid_image_feature, iid_text_feature)
            iid_image_feature, iid_text_feature = cca.transform(iid_image_feature, iid_text_feature)
        assert self.num_item == len(iid_image_feature) == len(iid_text_feature)
        self.image_embedding = nn.Embedding(iid_image_feature.shape[0], iid_image_feature.shape[1], padding_idx=0)
        self.image_embedding.weight.data.copy_(torch.from_numpy(iid_image_feature))
        self.text_embedding = nn.Embedding(iid_text_feature.shape[0], iid_text_feature.shape[1], padding_idx=0)
        self.text_embedding.weight.data.copy_(torch.from_numpy(iid_text_feature))

        if self.use_CD:
            self.image_cor_matrix = trans_to_cuda(torch.tensor(self.teacher_generate_cor(iid_image_feature))) # [N N]
            self.text_cor_matrix = trans_to_cuda(torch.tensor(self.teacher_generate_cor(iid_text_feature)))
            self.image_corcode_matrix = trans_to_cuda(torch.LongTensor(self.teacher_generate_cor_code(iid_image_feature))) # [N N]
            self.text_corcode_matrix = trans_to_cuda(torch.LongTensor(self.teacher_generate_cor_code(iid_text_feature))) # [N N]

        if config['use_AT']:
            slow_lr = config['lr'] / config['gamma']**len(config['schedule'])
            self.optimizer = torch.optim.Adam([
                {'params': self.image_embedding.weight, 'lr':slow_lr}, 
                {'params': self.text_embedding.weight, 'lr':slow_lr}, 
                {'params': [p for n, p in self.named_parameters() if n not in ['image_embedding.weight','text_embedding.weight']]}  # 其他参数使用默认的学习率
            ], lr=config['lr'], weight_decay=config['l2'])
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=config['lr'], weight_decay=config['l2'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config['lr_dc_step'], gamma=config['lr_dc'])
    
    def _init_weights(self, module):
        initializer_range = 0.02
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def teacher_generate_cor_code(self, iid_feature, normalize=False):
        split_arrays = np.split(iid_feature, self.product_domain_D, axis=1)
        sub_arrays = []
        for i, sub_array in enumerate(split_arrays):
            if self.cor_type == 'dot':
                sub_cos_cor = sub_array @ sub_array.T
            elif self.cor_type == 'cos':
                sub_cos_cor = cosine_similarity(sub_array) # [N, N]
            elif self.cor_type == 'euc': # euclidean_distances
                sub_cos_cor = euclidean_distances(sub_array)
            elif self.cor_type == 'man': # manhattan_distances
                sub_cos_cor = manhattan_distances(sub_array)
            sub_arrays.append(sub_cos_cor)
        correlation_vector = np.stack(sub_arrays, axis=2)
        flattened_data = correlation_vector.reshape(-1, self.product_domain_D)  # Reshape to (12102*12102, 8)
        if self.quantization_method == 'kmeans': # default
            print("Starting quantization ...")
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(n_clusters=self.cor_code_num, batch_size=10000)
            quantization_indices = kmeans.fit_predict(flattened_data)
            # Reshape quantization_indices back to (12102, 12102)
            quantization_indices = quantization_indices.reshape(iid_feature.shape[0], iid_feature.shape[0])
        elif self.quantization_method == 'vq':
            import faiss
            from scipy.cluster.vq import vq
            flattened_data = flattened_data.astype(np.float32)
            quantizer = faiss.IndexFlatL2(self.product_domain_D)  # Initialize a quantizer (self.product_domain_D is the dimension of the vectors)
            index = faiss.IndexIVFPQ(quantizer, self.product_domain_D, self.cor_code_num, self.product_domain_D, self.product_domain_D)  # self.product_domain_D is the bytes per vector
            index.train(flattened_data)
            index.add(flattened_data)
            # Get the quantized vectors
            codebook = index.quantizer.reconstruct_n(0, self.cor_code_num) # Now 'quantized_vectors' contains the K quantized vectors of shape (num_quantized_vectors, 8)
            quantization_indices, _ = vq(flattened_data, codebook)
            quantization_indices = quantization_indices.reshape(iid_feature.shape[0], iid_feature.shape[0])
        return quantization_indices

    def teacher_generate_cor(self, iid_feature, normalize=False):
        if self.cor_type == 'dot':
            cor_matrix = iid_feature @ iid_feature.T
        elif self.cor_type == 'cos':
            cor_matrix = cosine_similarity(iid_feature)
        elif self.cor_type == 'euc': # euclidean_distances
            cor_matrix = euclidean_distances(iid_feature)
        elif self.cor_type == 'man': # manhattan_distances
            cor_matrix = manhattan_distances(iid_feature)
        if normalize:
            sub_matrix = cor_matrix[1:, 1:]
            sub_matrix_normalized = (sub_matrix - torch.min(sub_matrix)) / (torch.max(sub_matrix) - torch.min(sub_matrix))
            cor_matrix[1:, 1:] = sub_matrix_normalized
        return cor_matrix
    
    def student_calculate_cor(self, nodes_h, normalize=False):
        B,N,D = nodes_h.size()
        if self.cor_type == 'dot':
            dot_product_list = []
            for i in range(B):
                batch_i = nodes_h[i]
                dot_product_i = torch.matmul(batch_i, batch_i.t())
                dot_product_list.append(dot_product_i)
            cor_matrix = torch.stack(dot_product_list) # [B N N]
        elif self.cor_type == 'cos':
            cos_similarity_list = []
            for i in range(B):
                batch_i = nodes_h[i]
                cos_similarity_i = F.cosine_similarity(batch_i.unsqueeze(1), batch_i.unsqueeze(0), dim=-1)
                cos_similarity_list.append(cos_similarity_i)
            cor_matrix = torch.stack(cos_similarity_list) # [B N N]
        elif self.cor_type == 'euc': # euclidean_distances
            euc_dist_list = []
            for i in range(B):
                batch_i = nodes_h[i]
                euc_dist_i = torch.cdist(batch_i, batch_i, p=2)  # compute euclidean distance
                euc_dist_list.append(euc_dist_i)
            cor_matrix = torch.stack(euc_dist_list) # [B N N]
        elif self.cor_type == 'man': # manhattan_distances
            man_dist_list = []
            for i in range(B):
                batch_i = nodes_h[i]
                man_dist_i = torch.cdist(batch_i, batch_i, p=1)  # compute manhattan distance
                man_dist_list.append(man_dist_i)
            cor_matrix = torch.stack(man_dist_list) # [B N N]
        return cor_matrix

    def calculate_teach_cor_loss(self, predict_cor, node_cor, cor_mask, T):
        ## Apply softmax with temperature T to both predictions
        predict_cor_soft = predict_cor / T
        node_cor_soft = cor_mask*node_cor / T
        ## Calculate L2 loss
        diff = torch.add(predict_cor_soft, -node_cor_soft)
        diff_squared = torch.sqrt(torch.pow(diff, 2) + 1e-8)
        global_teach_loss = (cor_mask * diff_squared).norm(dim=(1, 2)) # [B L L]
        ## Calculate Kullback-Leibler divergence loss
        # global_teach_loss = F.kl_div(input=predict_cor_soft, target=node_cor_soft, reduction='mean')
        # global_teach_loss = (cor_mask * loss).sum(dim=(1, 2)) # sum over last two dimensions
        return global_teach_loss.mean()
    
    def calculate_teach_corcode_loss(self, predict_logits, node_cor_codes, cor_mask):
        predict_logits = predict_logits.view(-1, predict_logits.shape[-1])  # now shape is [B*L*L, n]
        node_cor_codes = node_cor_codes.view(-1)  # now shape is [B*L*L]
        corcode_mask = cor_mask.view(-1).bool()  # now shape is [B*L*L]
        
        mask_expanded = cor_mask.unsqueeze(-1).repeat(1,1,1,predict_logits.size(-1))  # now shape is [B, L, L, n]
        mask_expanded = mask_expanded.view(-1, mask_expanded.shape[-1]).bool()  # now shape is [B*L*L, n]

        relevant_targets = torch.masked_select(node_cor_codes, corcode_mask)
        relevant_logits = torch.masked_select(predict_logits, mask_expanded)
        relevant_logits = relevant_logits.view(-1, predict_logits.shape[-1])
        teach_corcode_loss_v = F.cross_entropy(relevant_logits, relevant_targets) # ([B*L*L, n], [B*L*L]) => [1]
        return teach_corcode_loss_v

    def student_predict_cor_scores(self, h_nodes, adj, modality):
        if modality=='img':
            h_nodes = self.img_emb_project(h_nodes) # [B L D]
            # h_nodes = self.local_agg_img(h_nodes, adj) # [B L D]v # TODO: consider use it?
            h_nodes = self.img_cor_project(h_nodes)
        elif modality=='txt':
            h_nodes = self.txt_emb_project(h_nodes) # [B L D]
            # h_nodes = self.local_agg_txt(h_nodes, adj) # [B L D]v # TODO: consider use it?
            h_nodes = self.txt_cor_project(h_nodes)
        predict_cor = self.student_calculate_cor(h_nodes) # [B L L]
        return predict_cor

    def student_predict_corcodes_logits(self, h_nodes, modality):
        B = h_nodes.shape[0]
        N = h_nodes.shape[1]
        pair_cor_input = torch.abs((h_nodes.repeat(1, 1, N).view(B, N * N, self.dim) - h_nodes.repeat(1, N, 1)).view(B, N, N, self.dim))
        if modality=='img':
            pair_cor_input = self.img_cor_param_layer(pair_cor_input)
            pair_cor_input = F.relu(pair_cor_input)
            predict_logits = self.img_cor_prediction_layer(pair_cor_input)  # [B L L n]
        elif modality=='txt':
            pair_cor_input = self.txt_cor_param_layer(pair_cor_input)
            pair_cor_input = F.relu(pair_cor_input)
            predict_logits = self.txt_cor_prediction_layer(pair_cor_input)  # [B L L n]
        return predict_logits

    def forward(self, adj, nodes, node_msks, stage='train'):
        node_msks_float = node_msks.float()
        cor_mask = torch.bmm(node_msks_float.unsqueeze(2), node_msks_float.unsqueeze(1)) # [B L 1]*[B 1 L] => [B L L]
        # #####################
        h_nodes_emb = self.embedding(nodes)
        h_nodes = self.local_agg_item(h_nodes_emb, adj)
        # #####################
        h_image_nodes = self.image_embedding(nodes) # [B L D]
        # #####################
        h_text_nodes = self.text_embedding(nodes)
        # ##################### 
        
        # Fusion Layer
        input_h = torch.cat((h_nodes_emb.unsqueeze(-2), h_nodes.unsqueeze(-2), h_image_nodes.unsqueeze(-2), h_text_nodes.unsqueeze(-2)), -2) # [B n 3 D]
        # input_h = torch.cat((h_nodes_emb.unsqueeze(-2), h_nodes.unsqueeze(-2), h_image_nodes.unsqueeze(-2)), -2) # [B n 3 D]
        energy = self.modality_projection(input_h) # [B N 3 1]
        weights = torch.softmax(energy.squeeze(-1), dim=-1) # [B, N, 2]
        gate_output_h_nodes = (input_h * weights.unsqueeze(-1)).sum(dim=-2) # # (B, n, 3, D) * (B, n, 3, 1) -> (B, n, D)
        gate_output_h_nodes = self.dropout_out(gate_output_h_nodes)

        if self.use_CD:
            node_image_cor_scores = self.image_cor_matrix[nodes.unsqueeze(2), nodes.unsqueeze(1)] # teacher: [B L L]
            predict_img_cor = self.student_predict_cor_scores(h_image_nodes, adj, modality='img') # student: [B L L]
            teach_cor_loss_v = self.calculate_teach_cor_loss(predict_img_cor, node_image_cor_scores, cor_mask, self.tau) # [B L L]
            node_image_cor_codes = self.image_corcode_matrix[nodes.unsqueeze(2), nodes.unsqueeze(1)] # teacher: [B L L]
            predict_logits = self.student_predict_corcodes_logits(h_image_nodes, modality='img')  # [B L L n]
            teach_corcode_loss_v = self.calculate_teach_corcode_loss(predict_logits, node_image_cor_codes, cor_mask)

            node_text_cor_scores = self.text_cor_matrix[nodes.unsqueeze(2), nodes.unsqueeze(1)] # teacher: [B L L]
            predict_txt_cor = self.student_predict_cor_scores(h_text_nodes, adj, modality='txt') # student: [B L L]
            teach_cor_loss_t = self.calculate_teach_cor_loss(predict_txt_cor, node_text_cor_scores, cor_mask, self.tau) # [B L L]
            node_text_cor_codes = self.text_corcode_matrix[nodes.unsqueeze(2), nodes.unsqueeze(1)] # teacher: [B L L]
            predict_logits = self.student_predict_corcodes_logits(h_text_nodes, modality='txt')  # [B L L n]
            teach_corcode_loss_t = self.calculate_teach_corcode_loss(predict_logits, node_text_cor_codes, cor_mask)

            return gate_output_h_nodes * node_msks_float.unsqueeze(-1), teach_cor_loss_v, teach_corcode_loss_v, teach_cor_loss_t, teach_corcode_loss_t
        else:
            return gate_output_h_nodes * node_msks_float.unsqueeze(-1), 0, 0, 0, 0

    def get_sequence_representation(self, seq_hiddens, mask, pooling_method='last'):  # [B L D] => [B D]
        if pooling_method == 'last_attention': # keep the last k items
            batch_size = seq_hiddens.shape[0]
            len = seq_hiddens.shape[1]
            pos_emb = self.pos_embedding.weight[:len]
            pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
            nh = torch.matmul(torch.cat([pos_emb, seq_hiddens], -1), self.w_1)
            nh = torch.tanh(nh)

            last_n = 3 # TODO: this is a hyper-parameter
            gather_index = torch.sum(mask, dim = -1) - 1
            last_n_hiddens = [] # [B n D]
            for n in range(last_n):
                _gather_index = gather_index - n
                _gather_index[_gather_index<0] = 0
                _gather_index = _gather_index.view(-1, 1, 1).expand(-1, -1, nh.shape[-1])
                _hiddens = nh.gather(dim=1, index=_gather_index) # [B 1 D]
                last_n_hiddens.append(_hiddens.squeeze(1))
            last_n_hiddens = torch.stack(last_n_hiddens, dim=1) # [B n D]

            energy = self.projection(last_n_hiddens) # [B n 1]
            weights = torch.softmax(energy.squeeze(-1), dim=-1) # [B, n]
            hiddens = (last_n_hiddens * weights.unsqueeze(-1)).sum(dim=-2) # # (B, n, D) * (B, n, 1) -> (B, D)
        if pooling_method == 'attention':
            mask = mask.float().unsqueeze(-1)

            batch_size = seq_hiddens.shape[0]
            len = seq_hiddens.shape[1]
            pos_emb = self.pos_embedding.weight[:len]
            pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

            hs = torch.sum(seq_hiddens * mask, -2) / torch.sum(mask, 1) # sequence's representation [B D]
            hs = hs.unsqueeze(-2).repeat(1, len, 1) # [B L D]

            nh = torch.matmul(torch.cat([pos_emb, seq_hiddens], -1), self.w_1)
            nh = torch.tanh(nh)
            nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
            beta = torch.matmul(nh, self.w_2)
            beta = beta * mask
            hiddens = torch.sum(beta * seq_hiddens, 1)
        elif pooling_method == 'mean':
            mask = mask.float().unsqueeze(-1)
            hiddens = torch.sum(seq_hiddens * mask, -2) / torch.sum(mask, 1) # sequence's representation [B D]
        elif pooling_method == 'last_sum':
            last_n = self.last_n # TODO: this is a hyper-parameter
            gather_index = torch.sum(mask, dim = -1) - 1
            last_n_hiddens = [] # [B n D]
            for n in range(last_n):
                _gather_index = gather_index - n
                _gather_index[_gather_index<0] = 0
                _gather_index = _gather_index.view(-1, 1, 1).expand(-1, -1, seq_hiddens.shape[-1])
                _hiddens = seq_hiddens.gather(dim=1, index=_gather_index) # [B 1 D]
                last_n_hiddens.append(_hiddens.squeeze(1))
            last_n_hiddens = torch.stack(last_n_hiddens, dim=1) # [B n D]
            hiddens = torch.sum(last_n_hiddens, -2) # sequence's representation [B D]
        elif pooling_method == 'last':
            gather_index = torch.sum(mask, dim = -1) - 1
            gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, seq_hiddens.shape[-1])
            hiddens = seq_hiddens.gather(dim=1, index=gather_index)
            hiddens = hiddens.squeeze(1)
        return hiddens

    def compute_full_scores(self, node_hiddens, alias_item_inputs, item_seq_mask):
        '''
        Given node's hidden [B N D], predicting the scores [N] for all list of items
        '''
        mask = item_seq_mask.float().unsqueeze(-1) # [B L 1]
        # @_@: Items' hidden
        item_seq_hiddens = torch.gather(node_hiddens, 1, torch.unsqueeze(alias_item_inputs, dim=-1).expand(node_hiddens.shape[0], node_hiddens.shape[1], node_hiddens.shape[2]))
        item_seq_hiddens = mask * item_seq_hiddens
        item_hidden = self.get_sequence_representation(item_seq_hiddens, item_seq_mask, pooling_method=self.config['seq_pooling']) # [B L D] => [B D]
        item_emb = self.embedding.weight[1:self.num_item]  # (n_nodes+1) x latent_size 

        if self.config['modality_prediction']:
            # @_@: Images' hidden 
            img_hiddens = None
            img_emb = self.image_embedding.weight[1:self.num_item]
            # @_@: Texts' hidden        
            txt_hiddens = None
            txt_emb = self.text_embedding.weight[1:self.num_item]
            # @_@: linear layer
            emb = self.fusion_layer(torch.cat([item_emb, img_emb[1:], txt_emb[1:]], -1))
            hiddens = self.fusion_layer(torch.cat([item_hidden, img_hiddens, txt_hiddens], -1))
            scores = torch.matmul(hiddens, emb.transpose(1, 0))
        else:
            scores = torch.matmul(item_hidden, item_emb.transpose(1, 0))

        return scores
    

def model_process(model, data, stage='train'):
    adj, nodes, node_msks, alias_inputs, inputs_mask, targets, u_input = data
    adj = trans_to_cuda(adj).float()
    nodes, alias_inputs = trans_to_cuda(nodes).long(), trans_to_cuda(alias_inputs).long()
    node_msks, inputs_mask, targets, u_input = trans_to_cuda(node_msks).long(), trans_to_cuda(inputs_mask).long(), trans_to_cuda(targets).long(), trans_to_cuda(u_input).long()

    node_hidden, v_loss_1, v_loss_2, t_loss_1, t_loss_2 = model.forward(adj, nodes, node_msks, stage)
    scores = model.compute_full_scores(node_hidden, alias_inputs, inputs_mask) # the scores for all list of items

    return targets, scores, model.v_s_weight*v_loss_1, model.t_s_weight*v_loss_2, model.v_c_weight*t_loss_1, model.t_c_weight*t_loss_2


def train_and_test(model, train_loader, test_loader, topk=[20], logger=None):
    logger.info('start training.') if logger else print('start training.')
    model.train()
    total_loss, V_loss_1, V_loss_2, T_loss_1, T_loss_2 = 0.0, 0.0, 0.0, 0.0, 0.0
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores, v_loss_1, v_loss_2, t_loss_1, t_loss_2 = model_process(model, data, stage='train')
        loss = model.loss_function(scores, targets - 1)
        if model.use_CD:
            loss = loss + v_loss_1 + t_loss_1 + v_loss_2 + t_loss_2
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        V_loss_1 += v_loss_1
        V_loss_2 += v_loss_2
        T_loss_1 += t_loss_1
        T_loss_2 += t_loss_2
    loss_message = '\tLoss:\t%.3f \t V_loss_cor: %.3f \t V_loss_corcode: %.3f \t T_loss_cor: %.3f \t T_loss_corcode: %.3f' % (total_loss, V_loss_1, V_loss_2, T_loss_1, T_loss_2)
    logger.info(loss_message) if logger else print(loss_message)
    model.scheduler.step()

    logger.info('start predicting.') if logger else print('start predicting.')
    model.eval()
    hit, mrr, ndcg = [[] for k in topk], [[] for k in topk], [[] for k in topk]
    for data in test_loader:
        targets, scores, _, _, _, _ = model_process(model, data, stage='test')
        for i, k in enumerate(topk):
            hit_scores, mrr_scores, ndcg_scores = evaluate(targets, scores, k)
            hit[i] = hit[i]+hit_scores.tolist()
            mrr[i] = mrr[i]+mrr_scores.tolist()
            ndcg[i] = ndcg[i]+ndcg_scores.tolist()
    result = [[] for k in topk]
    for i, k in enumerate(topk):
        result[i].append(np.mean(hit[i]) * 100)
        result[i].append(np.mean(mrr[i]) * 100)
        result[i].append(np.mean(ndcg[i]) * 100)

    return result # [[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]]: [topk5, top20]


def evaluate(targets, scores, k):
    sorted_indices = scores.topk(k)[1]
    sorted_indices = trans_to_cpu(sorted_indices).detach()
    targets = trans_to_cpu(targets-1)

    hit_scores = hit_at_k(targets, sorted_indices, k)
    mrr_scores = mrr_at_k(targets, sorted_indices, k)
    ndcg_scores = ndcg_at_k(targets, sorted_indices, k)

    return hit_scores, mrr_scores, ndcg_scores

# 计算 NDCG
def ndcg_at_k(targets, sorted_indices, topk):
    k = min(topk, targets.shape[-1])
    ideal_dcg = torch.sum(1.0 / torch.log2(torch.arange(k) + 2))
    dcg = torch.sum(torch.where(sorted_indices[:, :k] == targets.unsqueeze(-1), 1.0 / torch.log2(torch.arange(k) + 2), torch.tensor(0.0)), dim=-1)
    return dcg / ideal_dcg # # [B]

# 计算 Hit
def hit_at_k(targets, sorted_indices, topk):
    k = min(topk, targets.shape[-1])
    hits = torch.sum(sorted_indices[:, :k] == targets.unsqueeze(-1), dim=-1).float()
    return hits # [B]

# 计算 MRR
def mrr_at_k(targets, sorted_indices, topk):
    k = min(topk, targets.shape[-1])
    indices = torch.arange(1, k + 1)
    rranks = torch.where(sorted_indices[:, :k] == targets.unsqueeze(-1), 1.0 / indices, torch.tensor(0.0))
    return torch.max(rranks, dim=-1)[0] # [B]
