# Code reference from HGR-DTA(https://github.com/Zhaoyang-Chu/HGRL-DTA/)
import os
import pickle, argparse
import random

import numpy as np
from itertools import chain
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset, Batch

def argparser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset for use', default='davis')
    parser.add_argument('--cuda_id', type=int, help='Cuda for use', default=0)
    parser.add_argument('--num_epochs', type=int, help='Number of epochs to train', default=3)  # num_epochs = 200, when conducting the S2, S3 and S4 experiments
    parser.add_argument('--batch_size', type=int, help='Batch size of dataset', default=512)
    parser.add_argument('--lr', type=float, help='Initial learning rate to train', default=0.0002)
    parser.add_argument('--model', type=int, help='Model id', default=0)
    parser.add_argument('--fold', type=int, help='Fold of 5-CV', default=-100)
    parser.add_argument('--dropedge_rate', type=float, help='Rate of edge dropout', default=0.1)
    parser.add_argument('--seed', type=int, help='random seed', default=2)
    parser.add_argument('--num_pos', type=int, default=5)    # 3--kiba 10
    parser.add_argument('--pos_threshold', type=float, default=8.0)
    parser.add_argument('--drug_sim_k', type=int, help='Similarity topk of drug', default=2)
    parser.add_argument('--target_sim_k', type=int, help='Similarity topk of target', default=7)
    parser.add_argument('--tau', type=float, default=0.2)   #温度参数
    parser.add_argument('--lam', type=float, default=0.5)   #对比偏置项
    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS


# initialize the dataset
class DTADataset(InMemoryDataset):
    def __init__(self, root='/tmp', transform=None, pre_transform=None, drug_ids=None, target_ids=None, y=None):
        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.process(drug_ids, target_ids, y)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, drug_ids, target_ids, y):
        data_list = []
        for i in range(len(drug_ids)):
            DTA = DATA.Data(drug_id=torch.IntTensor([drug_ids[i]]), target_id=torch.IntTensor([target_ids[i]]), y=torch.FloatTensor([y[i]]))
            data_list.append(DTA)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class MyData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class GraphDataset(InMemoryDataset):
    def __init__(self, root='/tmp', transform=None, pre_transform=None, graphs_dict=None, dttype=None, seq = None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.dttype = dttype
        self.process(graphs_dict, seq)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, graphs_dict, seq):
        data_list = []
        index = 0
        for key in graphs_dict:
            size, features, edge_index = graphs_dict[key]
            GCNData = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                seq_x = torch.unsqueeze(torch.Tensor(seq[index]),0) )
            GCNData.__setitem__(f'{self.dttype}_size', torch.LongTensor([size]))
            index += 1
            data_list.append(GCNData)
        print('data数据', len(data_list), data_list[0])
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train(architecture, predictor, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, LR, epoch, TRAIN_BATCH_SIZE,affinity_graph, drug_pos, target_pos):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    architecture.train()
    predictor.train()
    LOG_INTERVAL = 10
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, chain(architecture.parameters(), predictor.parameters())), lr=LR, weight_decay=0)
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader)) #DataBatch(x=[2180,78),edge_index=[2,7000],seq_x=(68,384),drug_size=(68),batch=[2180],ptr=(69) drug graphs
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))#[DataBatch(x=[348715,54), edge_index=[2,2437717), seq_x=[442,768),target_size=[442),batch=[348715), ptr=[443))) target graphs
    for batch_idx, data in enumerate(train_loader): #data:DataBatch:512

        optimizer.zero_grad()
        ssl_loss,drug_embedding, target_embedding = architecture(affinity_graph.to(device), drug_graph_batchs,
                                                                  target_graph_batchs, drug_pos, target_pos)#[68,128],[442,128]

        output, _ = predictor(data.to(device), drug_embedding, target_embedding) #[512,1]
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))+ssl_loss
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * TRAIN_BATCH_SIZE, len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()
            ))



def predicting(architecture, predictor, device, loader, drug_graphs_DataLoader, target_graphs_DataLoader,affinity_graph, drug_pos, target_pos):
    architecture.eval()
    predictor.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))  # drug graphs
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))  # target graphs
    with torch.no_grad():
        for data in loader:
            _, drug_embedding, target_embedding = architecture(affinity_graph.to(device), drug_graph_batchs, target_graph_batchs, drug_pos, target_pos)
            output, _ = predictor(data.to(device), drug_embedding, target_embedding)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    print('Prediction end')
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def getLinkEmbeddings(architecture, predictor, device, loader, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph, drug_map=None, drug_map_weight=None, target_map=None, target_map_weight=None):
    architecture.eval()
    predictor.eval()
    affinity_graph.to(device)  # affinity graph
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))  # drug graphs
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))  # target graphs
    with torch.no_grad():
        link_embeddings_batch_list = []
        for data in loader:
            drug_embedding, target_embedding = architecture(
                affinity_graph, drug_graph_batchs, target_graph_batchs,
                drug_map=drug_map, drug_map_weight=drug_map_weight, target_map=target_map, target_map_weight=target_map_weight
            )
            _, link_embeddings_batch = predictor(data.to(device), drug_embedding, target_embedding)
            link_embeddings_batch_list.append(link_embeddings_batch.cpu().numpy())
    link_embeddings = np.concatenate(link_embeddings_batch_list, axis=0)
    return link_embeddings


def getEmbeddings(architecture, device, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph):
    architecture.eval()
    affinity_graph.to(device)  # affinity graph
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))  # drug graphs
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))  # target graphs
    with torch.no_grad():
        drug_embedding, target_embedding = architecture(affinity_graph, drug_graph_batchs, target_graph_batchs)
    return drug_embedding.cpu().numpy(), target_embedding.cpu().numpy()



def collate(data_list):
    batch = Batch.from_data_list(data_list)
    return batch


def read_data(dataset):
    dataset_path = 'data/' + dataset + '/'
    affinity = pickle.load(open(dataset_path + 'affinities', 'rb'), encoding='latin1')
    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]   #对亲和性数据取负对数-log10，并除以1e9
    affinity = np.asarray(affinity)
    return affinity

def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def minMaxNormalize(Y, Y_min=None, Y_max=None):
    if Y_min is None:
        Y_min = np.min(Y)
    if Y_max is None:
        Y_max = np.max(Y)
    normalize_Y = (Y - Y_min) / (Y_max - Y_min)
    return normalize_Y


def denseAffinityRefine(adj, k):
    refine_adj = np.zeros_like(adj)
    indexs1 = np.tile(np.expand_dims(np.arange(adj.shape[0]), 0), (k, 1)).transpose()
    indexs2 = np.argpartition(adj, -k, 1)[:, -k:]
    refine_adj[indexs1, indexs2] = adj[indexs1, indexs2]
    return refine_adj

