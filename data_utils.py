import random
import numpy as np
import torch
from torch.utils.data import Dataset


def split_validation(train_set, valid_portion):
    train_set_x, train_set_image, train_set_text, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    valid_set_image = [train_set_image[s] for s in sidx[n_train:]]
    train_set_image = [train_set_image[s] for s in sidx[:n_train]]
    valid_set_text = [train_set_text[s] for s in sidx[n_train:]]
    train_set_text = [train_set_text[s] for s in sidx[:n_train]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_image, train_set_text, train_set_y), (valid_set_x, valid_set_image, valid_set_text, valid_set_y)


def get_item_num(train, test):
    i_max = max(max(train[-1]), max(test[-1]))
    for s in train[0]:
        for i in s:
            if i>i_max:
                i_max=i
    for s in test[0]:
        for i in s:
            if i>i_max:
                i_max=i
    return i_max+1

class Data(Dataset):
    def __init__(self, data, train_len=None):
        '''
        data[0]: [[a seq],[another seq], ...]
        data[1]: [target1, target2, ...]
        '''
        self.inverse_seq = False # TODO
        self.data_length = len(data[0])
        self.inputs = data[0]
        self.targets = data[-1]
        self.inputs, max_len = self._handle_data(data[0], train_len, self.inverse_seq)
        self.max_len = max_len        

    def _handle_data(self, inputData, train_len=None, reverse=False):
        len_data = [len(nowData) for nowData in inputData]
        if train_len is None:
            max_len = max(len_data)
        else:
            max_len = train_len
        # reverse the sequence
        us_pois = []
        for upois, le in zip(inputData, len_data):
            if reverse:
                _ = list(reversed(upois)) if le < max_len else list(reversed(upois[-max_len:]))
            else:
                _ = list(upois) if le < max_len else list(upois[:max_len])
            us_pois.append(_)
        return us_pois, max_len
    
    def __getitem__(self, index):
        u_input, target = self.inputs[index], self.targets[index]
        le = len(u_input) # real length of the inputs
        ## To Build Graph
        u_nodes = np.unique(u_input).tolist()
        ln = len(u_nodes)
        nodes = np.asarray(u_nodes + (self.max_len - len(u_nodes)) * [0])

        adj = np.zeros((self.max_len, self.max_len))
        for i in np.arange(le):
            item = u_input[i]
            item_idx = np.where(nodes == item)[0][0] # idx in nodes set
            adj[item_idx][item_idx] = 1 # self-loop
        for i in np.arange(le - 1):
            prev_item = u_input[i]
            next_item = u_input[i+1]
            u = np.where(nodes == prev_item)[0][0]
            v = np.where(nodes == next_item)[0][0]
            if u == v or adj[u][v] == 4:
                continue
            if adj[v][u] == 2:
                adj[u][v] = 4
                adj[v][u] = 4
            else:
                adj[u][v] = 2
                adj[v][u] = 3
        
        alias_inputs =[] 
        for item in u_input:
            item_idx = np.where(nodes == item)[0][0]
            alias_inputs.append(item_idx)

        alias_inputs = alias_inputs + [0] * (self.max_len-le)
        u_input = u_input + [0] * (self.max_len-le)
        us_msks = [1] * le + [0] * (self.max_len-le) if le < self.max_len else [1] * self.max_len        
        node_msks = [1] * ln + [0] * (self.max_len-ln) if ln < self.max_len else [1] * self.max_len        

        return [torch.tensor(adj), torch.tensor(nodes), torch.tensor(node_msks),
                torch.tensor(alias_inputs),
        		torch.tensor(us_msks), torch.tensor(target), torch.tensor(u_input)]

    def __len__(self):
        return self.data_length
