import os
import json, torch
import numpy as np
import torch.backends.cudnn
import torch.utils.data
import scipy.sparse as sp
from mpl_toolkits.axes_grid1 import make_axes_locatable
from model import (Predictor, MFCLDTA)
from metrics import model_evaluate
from torch_geometric import data as DATA
from utils import argparser, DTADataset, GraphDataset, collate, predicting, read_data, train, setup_seed,sparse_mx_to_torch_sparse_tensor,minMaxNormalize, denseAffinityRefine
import warnings

warnings.filterwarnings('ignore')

def process_data(affinity_mat, dataset, num_pos, pos_threshold):
    dataset_path = 'data/' + dataset + '/'

    train_file = json.load(open(dataset_path + 'train_set.txt'))
    train_index = []
    for i in range(len(train_file)):
        train_index += train_file[i]
    test_index = json.load(open(dataset_path + 'test_set.txt'))

    rows, cols = np.where(np.isnan(affinity_mat) == False)
    train_rows, train_cols = rows[train_index], cols[train_index]
    train_Y = affinity_mat[train_rows, train_cols]
    train_dataset = DTADataset(drug_ids=train_rows, target_ids=train_cols, y=train_Y)
    test_rows, test_cols = rows[test_index], cols[test_index]
    test_Y = affinity_mat[test_rows, test_cols]
    test_dataset = DTADataset(drug_ids=test_rows, target_ids=test_cols, y=test_Y)

    train_affinity_mat = np.zeros_like(affinity_mat)
    train_affinity_mat[train_rows, train_cols] = train_Y
    affinity_graph, drug_pos, target_pos = get_affinity_graph(dataset, train_affinity_mat, num_pos, pos_threshold)
    return train_dataset, test_dataset, affinity_graph, drug_pos, target_pos

def get_affinity_graph(dataset, adj, num_pos, pos_threshold):
    dataset_path = 'data/' + dataset + '/'
    num_drug, num_target = adj.shape[0], adj.shape[1]

    dt_ = adj.copy()
    dt_ = np.where(dt_ >= pos_threshold, 1.0, 0.0)
    dtd = np.matmul(dt_, dt_.T)
    dtd = dtd / dtd.sum(axis=-1).reshape(-1, 1)
    dtd = np.nan_to_num(dtd)
    dtd += np.eye(num_drug, num_drug)
    dtd = dtd.astype("float32")
    d_d = np.loadtxt(dataset_path + 'drug-drug-sim.txt', delimiter=',')
    dAll = dtd + d_d
    drug_pos = np.zeros((num_drug, num_drug))
    for i in range(len(dAll)):
        one = dAll[i].nonzero()[0]
        if len(one) > num_pos:
            oo = np.argsort(-dAll[i, one])
            sele = one[oo[:num_pos]]
            drug_pos[i, sele] = 1
        else:
            drug_pos[i, one] = 1
    drug_pos = sp.coo_matrix(drug_pos)
    drug_pos = sparse_mx_to_torch_sparse_tensor(drug_pos)

    td_ = adj.T.copy()
    td_ = np.where(td_ >= pos_threshold, 1.0, 0.0)
    tdt = np.matmul(td_, td_.T)
    tdt = tdt / tdt.sum(axis=-1).reshape(-1, 1)
    tdt = np.nan_to_num(tdt)
    tdt += np.eye(num_target, num_target)
    tdt = tdt.astype("float32")
    t_t = np.loadtxt(dataset_path + 'target-target-sim.txt', delimiter=',')
    tAll = tdt + t_t
    target_pos = np.zeros((num_target, num_target))
    for i in range(len(tAll)):
        one = tAll[i].nonzero()[0]
        if len(one) > num_pos:
            oo = np.argsort(-tAll[i, one])
            sele = one[oo[:num_pos]]
            target_pos[i, sele] = 1
        else:
            target_pos[i, one] = 1
    target_pos = sp.coo_matrix(target_pos)
    target_pos = sparse_mx_to_torch_sparse_tensor(target_pos)

    if dataset == "davis":
        adj[adj != 0] -= 5
        adj_norm = minMaxNormalize(adj, 0)
    elif dataset == "kiba":
        adj_refine = denseAffinityRefine(adj.T, 150)
        adj_refine = denseAffinityRefine(adj_refine.T, 40)
        adj_norm = minMaxNormalize(adj_refine, 0)
    adj_1 = adj_norm
    adj_2 = adj_norm.T
    adj = np.concatenate((
        np.concatenate((np.zeros([num_drug, num_drug]), adj_1), 1),
        np.concatenate((adj_2, np.zeros([num_target, num_target])), 1)
    ), 0)
    train_row_ids, train_col_ids = np.where(adj != 0)
    edge_indexs = np.concatenate((
        np.expand_dims(train_row_ids, 0),
        np.expand_dims(train_col_ids, 0)
    ), 0)
    edge_weights = adj[train_row_ids, train_col_ids]
    node_type_features = np.concatenate((
        np.tile(np.array([1, 0]), (num_drug, 1)),
        np.tile(np.array([0, 1]), (num_target, 1))
    ), 0)
    adj_features = np.zeros_like(adj)
    adj_features[adj != 0] = 1
    features = np.concatenate((node_type_features, adj_features), 1)
    affinity_graph = DATA.Data(x=torch.Tensor(features), adj=torch.Tensor(adj),
                               edge_index=torch.LongTensor(edge_indexs))
    affinity_graph.__setitem__("edge_weight", torch.Tensor(edge_weights))
    affinity_graph.__setitem__("num_drug", num_drug)
    affinity_graph.__setitem__("num_target", num_target)

    return affinity_graph, drug_pos, target_pos

def create_dataset_for_train_test(affinity, dataset, fold):
    # load dataset
    dataset_path = 'data/' + dataset + '/'

    train_fold_origin = json.load(open(dataset_path + 'train_set.txt'))
    train_folds = []
    for i in range(len(train_fold_origin)):
        if i != fold:
            train_folds += train_fold_origin[i]
    test_fold = json.load(open(dataset_path + 'test_set.txt')) if fold == -100 else train_fold_origin[fold]


    rows, cols = np.where(np.isnan(affinity) == False)
    train_rows, train_cols = rows[train_folds], cols[train_folds]

    train_Y = affinity[train_rows, train_cols]
    train_dataset = DTADataset(drug_ids=train_rows, target_ids=train_cols, y=train_Y)

    test_rows, test_cols = rows[test_fold], cols[test_fold]
    test_Y = affinity[test_rows, test_cols]
    test_dataset = DTADataset(drug_ids=test_rows, target_ids=test_cols, y=test_Y)

    return train_dataset, test_dataset


def train_test():
    FLAGS = argparser()
    dataset = FLAGS.dataset
    cuda_name = f'cuda:{FLAGS.cuda_id}'
    TRAIN_BATCH_SIZE = FLAGS.batch_size
    TEST_BATCH_SIZE = FLAGS.batch_size
    NUM_EPOCHS = FLAGS.num_epochs
    LR = FLAGS.lr
    tau = FLAGS.tau
    lam = FLAGS.lam
    num_pos=FLAGS.num_pos
    pos_threshold=FLAGS.pos_threshold
    Architecture = MFCLDTA
    fold = FLAGS.fold
    model_name = Architecture.__name__
    if fold != -100:
        model_name += f"-{fold}"

    print("Dataset:", dataset)
    print("Cuda name:", cuda_name)
    print("Epochs:", NUM_EPOCHS)
    print('batch size', TRAIN_BATCH_SIZE)
    print("Learning rate:", LR)
    print("Fold", fold)
    print("Model name:", model_name)
    print("Train and test") if fold == -100 else print("Fold of 5-CV:", fold)

    if os.path.exists(f"models/architecture/{dataset}/S1/cross_validation/") is False:
        os.makedirs(f"models/architecture/{dataset}/S1/cross_validation/")
    if os.path.exists(f"models/predictor/{dataset}/S1/cross_validation/") is False:
        os.makedirs(f"models/predictor/{dataset}/S1/cross_validation/")
    if os.path.exists(f"models/architecture/{dataset}/S1/test/") is False:
        os.makedirs(f"models/architecture/{dataset}/S1/test/")
    if os.path.exists(f"models/predictor/{dataset}/S1/test/") is False:
        os.makedirs(f"models/predictor/{dataset}/S1/test/")

    print("\ncreate dataset ......")

    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    affinity = read_data(dataset)

    train_data, test_data, affinity_graph, drug_pos, target_pos = process_data(affinity, dataset, num_pos,
                                                                               pos_threshold)
    drug_pos = drug_pos.to(device)
    target_pos = target_pos.to(device)
    print("create train_loader and test_loader ...")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                               collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)

    print("create drug_graphs_dict and target_graphs_dict ...")
    drug_graphs_dict = torch.load(f'data/{dataset}/drug_graph.pt')
    target_graphs_dict = torch.load(f'data/{dataset}/target_graph.pt')
    drug_seq_embedding = np.load(f'data/{dataset}/drugs_embedding.npy')
    target_seq_embedding = np.load(f'data/{dataset}/targets_embedding.npy')
    a=len(drug_graphs_dict)

    print("create drug_graphs_DataLoader and target_graphs_DataLoader ...")
    drug_graphs_Data = GraphDataset(graphs_dict=drug_graphs_dict, dttype="drug", seq=drug_seq_embedding)
    drug_graphs_DataLoader = torch.utils.data.DataLoader(drug_graphs_Data, shuffle=False, collate_fn=collate,
                                                         batch_size=len(drug_graphs_dict))

    target_graphs_Data = GraphDataset(graphs_dict=target_graphs_dict, dttype="target", seq=target_seq_embedding)
    target_graphs_DataLoader = torch.utils.data.DataLoader(target_graphs_Data, shuffle=False, collate_fn=collate,
                                                           batch_size=len(target_graphs_dict))
    #创建模型
    architecture = Architecture(tau,lam, ns_dims=[affinity_graph.num_drug + affinity_graph.num_target + 2, 512, 256],
                                 mg_init_dim=78, pg_init_dim=54,  embedding_dim=128,dropout_rate=0.2)
    architecture.to(device)


    predictor = Predictor(embedding_dim=architecture.output_dim)
    predictor.to(device)


    if fold != -100:
        best_result = [1000]

    print("start training ...")

    for epoch in range(NUM_EPOCHS):
        train(architecture, predictor, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, LR, epoch + 1, TRAIN_BATCH_SIZE,affinity_graph, drug_pos, target_pos)
        G, P = predicting(architecture, predictor, device, test_loader, drug_graphs_DataLoader,target_graphs_DataLoader,affinity_graph, drug_pos, target_pos)
        result = model_evaluate(G, P, dataset)
        print(result)

        if fold != -100 and result[0] < best_result[0]:
            best_result = result
            G = np.array(G)
            P = np.array(P)
            checkpoint_path = f"models/architecture/{dataset}/benchmark/cross_validation/{model_name}.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(architecture.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

            checkpoint_path = f"models/predictor/{dataset}/benchmark/cross_validation/{model_name}.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(predictor.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)
            epo=epoch
    if fold == -100:
        checkpoint_path = f"models/architecture/{dataset}/benchmark/test/{model_name}.pt"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(architecture.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

        checkpoint_path = f"models/predictor/{dataset}/benchmark/test/{model_name}.pt"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(predictor.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

        print('\npredicting for test data')

        G, P = predicting(architecture, predictor, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader,affinity_graph, drug_pos, target_pos)
        result = model_evaluate(G, P, dataset)
        print("reslut:", result)
    else:
        print(f"\nbest result for fold {fold} of cross validation:")
        print("The best result at epoch =",epo)
        print("reslut:", best_result)

if __name__ == '__main__':
    seed = 2
    setup_seed(seed)

    train_test()
