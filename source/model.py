import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Dropout, LSTM
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import DenseGCNConv, GCNConv, global_mean_pool as gep

vector_operations = {
    "cat": (lambda x, y: torch.cat((x, y), -1), lambda dim: 2 * dim),
    "add": (torch.add, lambda dim: dim),
    "sub": (torch.sub, lambda dim: dim),
    "mul": (torch.mul, lambda dim: dim),
    "combination1": (lambda x, y: torch.cat((x, y, torch.add(x, y)), -1), lambda dim: 3 * dim),
    "combination2": (lambda x, y: torch.cat((x, y, torch.sub(x, y)), -1), lambda dim: 3 * dim),
    "combination3": (lambda x, y: torch.cat((x, y, torch.mul(x, y)), -1), lambda dim: 3 * dim),
    "combination4": (lambda x, y: torch.cat((torch.add(x, y), torch.sub(x, y)), -1), lambda dim: 2 * dim),
    "combination5": (lambda x, y: torch.cat((torch.add(x, y), torch.mul(x, y)), -1), lambda dim: 2 * dim),
    "combination6": (lambda x, y: torch.cat((torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 2 * dim),
    "combination7": (
    lambda x, y: torch.cat((torch.add(x, y), torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 3 * dim),
    "combination8": (lambda x, y: torch.cat((x, y, torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 4 * dim),
    "combination9": (lambda x, y: torch.cat((x, y, torch.add(x, y), torch.mul(x, y)), -1), lambda dim: 4 * dim),
    "combination10": (lambda x, y: torch.cat((x, y, torch.add(x, y), torch.sub(x, y)), -1), lambda dim: 4 * dim),
    "combination11": (
    lambda x, y: torch.cat((x, y, torch.add(x, y), torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 5 * dim)
}


class LinearBlock(torch.nn.Module):
    def __init__(self, linear_layers_dim, dropout_rate=0, relu_layers_index=[], dropout_layers_index=[]):
        super(LinearBlock, self).__init__()

        self.layers = torch.nn.ModuleList()
        for i in range(len(linear_layers_dim) - 1):
            layer = Linear(linear_layers_dim[i], linear_layers_dim[i + 1])
            self.layers.append(layer)

        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x):
        output = x
        embeddings = [x]
        for layer_index in range(len(self.layers)):
            output = self.layers[layer_index](output)
            if layer_index in self.relu_layers_index:
                output = self.relu(output)
            if layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(output)
        return embeddings


class ResLinearBlock(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers=8, dropout_rate=0.2):
        super(ResLinearBlock, self).__init__()

        self.layer1 = Linear(input_dim, hidden_dim)
        self.layer2 = Linear(hidden_dim, output_dim)
        self.layers = torch.nn.ModuleList()
        for i in range(layers):
            layer = Linear(hidden_dim, hidden_dim)
            self.layers.append(layer)

        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_layers_index = [0, 1, 2, 3, 4, 5, 6, 7]
        self.dropout_layers_index = [0, 1, 2, 3, 4, 5, 6, 7]

    def forward(self, x):

        output = self.dropout(self.relu(self.layer1(x)))
        last_output = output
        for layer_index in range(len(self.layers)):
            if layer_index != 0 and layer_index % 2 == 0:
                output = output + last_output
            output = self.layers[layer_index](output)
            if layer_index in self.relu_layers_index:
                output = self.relu(output)
            if layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            if layer_index != 0 and layer_index % 2 == 0:
                last_output = output

        output = self.dropout(self.layer2(output))

        return output


class GCNBlock(torch.nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0, relu_layers_index=[], dropout_layers_index=[],
                 supplement_mode=None): #[78,78,156,312]
        super(GCNBlock, self).__init__()

        self.conv_layers = torch.nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            if supplement_mode is not None and i == 1:
                self.supplement_func, supplement_dim_func = vector_operations[supplement_mode]#
                conv_layer_input = supplement_dim_func(gcn_layers_dim[i])
            else:
                conv_layer_input = gcn_layers_dim[i]
            conv_layer = GCNConv(conv_layer_input, gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)

        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, edge_index, edge_weight, batch, supplement_x=None):
        output = x
        embeddings = [x]

        for conv_layer_index in range(len(self.conv_layers)):
            if supplement_x is not None and conv_layer_index == 1:
                output = self.supplement_func(output, supplement_x)

            output = self.conv_layers[conv_layer_index](output, edge_index, edge_weight)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(gep(output, batch))
        return embeddings


class GCNModel(torch.nn.Module):
    def __init__(self, layers_dim, supplement_mode=None):
        super(GCNModel, self).__init__()

        self.num_layers = len(layers_dim) - 1 #3
        self.graph_conv = GCNBlock(layers_dim, relu_layers_index=range(self.num_layers),
                                   supplement_mode=supplement_mode)

    def forward(self, graph_batchs, supplement_x=None):

        if supplement_x is not None:
            supplement_i = 0

            for graph_batch in graph_batchs:
                graph_batch.__setitem__('supplement_x',
                                        supplement_x[supplement_i: supplement_i + graph_batch.num_graphs])
                supplement_i += graph_batch.num_graphs

            embedding_batchs = list(map(lambda graph: self.graph_conv(graph.x, graph.edge_index, None, graph.batch,
                                                                      supplement_x=graph.supplement_x[
                                                                          graph.batch.int().cpu().numpy()]),
                                        graph_batchs))
        else:
            embedding_batchs = list(
                map(lambda graph: self.graph_conv(graph.x, graph.edge_index, None, graph.batch), graph_batchs))

        embeddings = []
        for i in range(self.num_layers + 1):
            embeddings.append(torch.cat(list(map(lambda embedding_batch: embedding_batch[i], embedding_batchs)), 0))

        return embeddings


class DenseGCNBlock(nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(DenseGCNBlock, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            conv_layer = DenseGCNConv(gcn_layers_dim[i], gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, adj):
        output = x
        embeddings = []
        for conv_layer_index in range(len(self.conv_layers)):
            output = self.conv_layers[conv_layer_index](output, adj, add_loop=False)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(torch.squeeze(output, dim=0))

        return embeddings


class DenseGCNModel(nn.Module):
    def __init__(self, layers_dim, edge_dropout_rate=0.):
        super(DenseGCNModel, self).__init__()

        self.edge_dropout_rate = edge_dropout_rate
        self.num_layers = len(layers_dim) - 1
        self.graph_conv = DenseGCNBlock(layers_dim, 0.1, relu_layers_index=list(range(self.num_layers)),
                                        dropout_layers_index=list(range(self.num_layers)))

    def forward(self, graph):
        xs, adj, num_d, num_t = graph.x, graph.adj, graph.num_drug, graph.num_target
        indexs = torch.where(adj != 0)
        edge_indexs = torch.cat((torch.unsqueeze(indexs[0], 0), torch.unsqueeze(indexs[1], 0)), 0)
        edge_indexs_dropout, edge_weights_dropout = dropout_adj(edge_index=edge_indexs, edge_attr=adj[indexs],
                                                                p=self.edge_dropout_rate, force_undirected=True,
                                                                num_nodes=num_d + num_t, training=self.training)
        adj_dropout = torch.zeros_like(adj)
        adj_dropout[edge_indexs_dropout[0], edge_indexs_dropout[1]] = edge_weights_dropout

        embeddings = self.graph_conv(xs, adj_dropout)

        return embeddings



class Contrast(nn.Module):
    def __init__(self, hidden_dim, output_dim, tau, lam):
        super(Contrast, self).__init__()

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim))
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)

        return sim_matrix

    def forward(self, za, zb, pos):
        za_proj = self.proj(za)
        zb_proj = self.proj(zb)
        matrix_a2b = self.sim(za_proj, zb_proj)
        matrix_b2a = matrix_a2b.t()

        matrix_a2b = matrix_a2b / (torch.sum(matrix_a2b, dim=1).view(-1, 1) + 1e-8)
        lori_a = -torch.log(matrix_a2b.mul(pos.to_dense()).sum(dim=-1)).mean()

        matrix_b2a = matrix_b2a / (torch.sum(matrix_b2a, dim=1).view(-1, 1) + 1e-8)
        lori_b = -torch.log(matrix_b2a.mul(pos.to_dense()).sum(dim=-1)).mean()
        a=torch.cat((za_proj, zb_proj), 1)
        b=torch.cat((zb_proj, za_proj), 1)
        return self.lam * lori_a + (1 - self.lam) * lori_b, (torch.cat((za_proj, zb_proj), 1)+torch.cat(( zb_proj,za_proj), 1))/2


class MFCLDTA(torch.nn.Module):
    def __init__(self,tau,lam,ns_dims, mg_init_dim=78, pg_init_dim=54,  embedding_dim=128,dropout_rate=0.2):
        super(MFCLDTA, self).__init__()

        self.drug_LSTMNet = nn.LSTM(input_size=384, hidden_size=128, num_layers=3, bidirectional=True, batch_first=True,
                                 dropout=0.2)
        self.target_LSTMNet = LSTM(input_size=768, hidden_size=128, num_layers=3, bidirectional=True, batch_first=True,
                                   dropout=0.2)


        drug_graph_dims = [mg_init_dim, mg_init_dim, mg_init_dim * 2, 256]
        target_graph_dims = [pg_init_dim, pg_init_dim, pg_init_dim * 2, 256]

        drug_output_dims = [drug_graph_dims[-1] , 1024, embedding_dim]
        target_output_dims = [target_graph_dims[-1] , 1024, embedding_dim]

        self.output_dim = embedding_dim

        self.drug_graph_conv = GCNModel(drug_graph_dims)
        self.target_graph_conv = GCNModel(target_graph_dims)

        self.drug_output_linear = LinearBlock(drug_output_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.target_output_linear = LinearBlock(target_output_dims, 0.2, relu_layers_index=[0],
                                                dropout_layers_index=[0, 1])
        self.embed_dim = 128
        self.n_filters = 32
        # drug sequence feature extractor-----------------------------------------------------------------------------------------------------
        self.embedding_xd = nn.Embedding(100, self.embed_dim)
        self.conv_smile1 = nn.Conv1d(in_channels=100, out_channels=2 * self.n_filters, kernel_size=3, padding=1)
        self.bn_conv_smile1 = nn.BatchNorm1d(2 * self.n_filters)
        self.conv_smile2 = nn.Conv1d(in_channels=2 * self.n_filters, out_channels=self.n_filters, kernel_size=3, padding=1)
        self.bn_conv_smile2 = nn.BatchNorm1d(self.n_filters)

        self.affinity_graph_conv = DenseGCNModel(ns_dims, dropout_rate)
        self.drug_contrast = Contrast(ns_dims[-1], embedding_dim, tau, lam)
        self.target_contrast = Contrast(ns_dims[-1], embedding_dim, tau, lam)

    def forward(self,  affinity_graph, drug_graph_batchs, target_graph_batchs, drug_pos, target_pos):
        num_d = affinity_graph.num_drug
        affinity_graph_embedding = self.affinity_graph_conv(affinity_graph)[-1]

        a = affinity_graph_embedding[:num_d]
        b = drug_graph_batchs[0].seq_x

        drug_seq_embedding, (dhn, dcn) = self.drug_LSTMNet(drug_graph_batchs[0].seq_x)
        target_seq_embedding, (tdn, tcn) = self.target_LSTMNet(target_graph_batchs[0].seq_x)

        drug_graph_embedding = self.drug_graph_conv(drug_graph_batchs)[-1]
        target_graph_embedding = self.target_graph_conv(target_graph_batchs)[-1]

        dru_loss1, drug_graph_embedding1  = self.drug_contrast(affinity_graph_embedding[:num_d], drug_graph_embedding,
                                                      drug_pos)

        dru_loss2, drug_graph_embedding2 = self.drug_contrast(drug_seq_embedding,affinity_graph_embedding[:num_d],
                                                              drug_pos)
        dru_loss3, drug_graph_embedding3 = self.drug_contrast(drug_graph_embedding, drug_seq_embedding,
                                                              drug_pos)
        dru_loss=(dru_loss1+dru_loss2+dru_loss3)/3


        tar_loss1, target_graph_embedding1  = self.target_contrast(affinity_graph_embedding[num_d:], target_graph_embedding,
                                                          target_pos)
        tar_loss2, target_graph_embedding2 = self.target_contrast(target_seq_embedding,affinity_graph_embedding[num_d:],
                                                                target_pos)
        tar_loss3, target_graph_embedding3 = self.target_contrast(target_graph_embedding,target_seq_embedding,
                                                                  target_pos)

        tar_loss=(tar_loss1+tar_loss2+tar_loss3)/3

        ssl_loss =tar_loss+dru_loss
        drug_output_embedding1 = self.drug_output_linear(drug_graph_embedding1)[-1]
        drug_output_embedding2 = self.drug_output_linear(drug_graph_embedding2)[-1]
        drug_output_embedding3 = self.drug_output_linear(drug_graph_embedding3)[-1]

        target_output_embedding1 = self.target_output_linear(target_graph_embedding1)[-1]
        target_output_embedding2 = self.target_output_linear(target_graph_embedding2)[-1]
        target_output_embedding3 = self.target_output_linear(target_graph_embedding3)[-1]

        drug_output_embedding =  (drug_output_embedding1+drug_output_embedding2+drug_output_embedding3)/3
        target_output_embedding =(target_output_embedding1+target_output_embedding2+target_output_embedding3)/3
        return ssl_loss,drug_output_embedding, target_output_embedding


class MFCLDTA_cold(torch.nn.Module):
    def __init__(self, mg_init_dim=78, pg_init_dim=54, embedding_dim=128):
        super(MFCLDTA_cold, self).__init__()
        print('DAS')

        drug_graph_dims = [mg_init_dim, mg_init_dim, mg_init_dim * 2, mg_init_dim * 4]
        target_graph_dims = [pg_init_dim, pg_init_dim, pg_init_dim * 2, pg_init_dim * 4]

        drug_output_dims = [drug_graph_dims[-1] + 256, 1024, embedding_dim]
        target_output_dims = [target_graph_dims[-1] + 256, 1024, embedding_dim]

        self.output_dim = embedding_dim

        self.drug_graph_conv = GCNModel(drug_graph_dims)
        self.target_graph_conv = GCNModel(target_graph_dims)

        self.drug_seq_linear = ResLinearBlock(384, 256, 256, dropout_rate= 0.6)
        self.target_seq_linear = ResLinearBlock(768, 256, 256, dropout_rate= 0.6)

        self.drug_output_linear = LinearBlock(drug_output_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.target_output_linear = LinearBlock(target_output_dims, 0.2, relu_layers_index=[0],
                                                dropout_layers_index=[0, 1])

    def forward(self, drug_graph_batchs, target_graph_batchs):

        drug_seq_embedding = self.drug_seq_linear(drug_graph_batchs[0].seq_x)
        target_seq_embedding = self.target_seq_linear(target_graph_batchs[0].seq_x)

        drug_graph_embedding = self.drug_graph_conv(drug_graph_batchs)[-1]
        target_graph_embedding = self.target_graph_conv(target_graph_batchs)[-1]

        drug_embedding = torch.cat((drug_seq_embedding, drug_graph_embedding), 1)
        target_embedding = torch.cat((target_seq_embedding, target_graph_embedding), 1)

        drug_output_embedding = self.drug_output_linear(drug_embedding)[-1]
        target_output_embedding = self.target_output_linear(target_embedding)[-1]

        return drug_output_embedding, target_output_embedding


class MFCLDTA_cold_drug(torch.nn.Module):
    def __init__(self, mg_init_dim=78, pg_init_dim=54, embedding_dim=128):
        super(MFCLDTA_cold_drug, self).__init__()

        self.target_LSTMNet = LSTM(input_size=768, hidden_size=128, num_layers=3, bidirectional=True, batch_first=True,
                                   dropout=0.2)

        drug_graph_dims = [mg_init_dim, mg_init_dim, mg_init_dim * 2, mg_init_dim * 4]
        target_graph_dims = [pg_init_dim, pg_init_dim, pg_init_dim * 2, pg_init_dim * 4]

        drug_output_dims = [drug_graph_dims[-1] + 256, 1024, embedding_dim]
        target_output_dims = [target_graph_dims[-1] + 256, 1024, embedding_dim]

        self.output_dim = embedding_dim

        self.drug_graph_conv = GCNModel(drug_graph_dims)
        self.target_graph_conv = GCNModel(target_graph_dims)

        self.drug_seq_linear = ResLinearBlock(384, 256, 256, dropout_rate=0.5)

        self.drug_output_linear = LinearBlock(drug_output_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.target_output_linear = LinearBlock(target_output_dims, 0.2, relu_layers_index=[0],
                                                dropout_layers_index=[0, 1])

    def forward(self, drug_graph_batchs, target_graph_batchs):
        drug_seq_embedding = self.drug_seq_linear(drug_graph_batchs[0].seq_x)
        target_seq_embedding, (tdn, tcn) = self.target_LSTMNet(target_graph_batchs[0].seq_x)

        drug_graph_embedding = self.drug_graph_conv(drug_graph_batchs)[-1]
        target_graph_embedding = self.target_graph_conv(target_graph_batchs)[-1]

        # 融合策略
        drug_embedding = torch.cat((drug_seq_embedding, drug_graph_embedding), 1)
        target_embedding = torch.cat((target_seq_embedding, target_graph_embedding), 1)

        drug_output_embedding = self.drug_output_linear(drug_embedding)[-1]
        target_output_embedding = self.target_output_linear(target_embedding)[-1]

        return drug_output_embedding, target_output_embedding


class MFCLDTA_cold_target(torch.nn.Module):
    def __init__(self, mg_init_dim=78, pg_init_dim=54, embedding_dim=128):
        super(MFCLDTA_cold_target, self).__init__()
        print('DAS')

        self.drug_LSTMNet = LSTM(input_size=384, hidden_size=128, num_layers=3, bidirectional=True, batch_first=True,
                                 dropout=0.2)

        drug_graph_dims = [mg_init_dim, mg_init_dim, mg_init_dim * 2, mg_init_dim * 4]
        target_graph_dims = [pg_init_dim, pg_init_dim, pg_init_dim * 2, pg_init_dim * 4]

        drug_output_dims = [drug_graph_dims[-1] + 256, 1024, embedding_dim]
        target_output_dims = [target_graph_dims[-1] + 256, 1024, embedding_dim]

        self.output_dim = embedding_dim

        self.drug_graph_conv = GCNModel(drug_graph_dims)
        self.target_graph_conv = GCNModel(target_graph_dims)


        self.target_seq_linear = ResLinearBlock(768, 256, 256, dropout_rate=0.3)

        self.drug_output_linear = LinearBlock(drug_output_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.target_output_linear = LinearBlock(target_output_dims, 0.2, relu_layers_index=[0],
                                                dropout_layers_index=[0, 1])

    def forward(self, drug_graph_batchs, target_graph_batchs):

        drug_seq_embedding, (dhn, dcn) = self.drug_LSTMNet(drug_graph_batchs[0].seq_x)
        target_seq_embedding = self.target_seq_linear(target_graph_batchs[0].seq_x)

        drug_graph_embedding = self.drug_graph_conv(drug_graph_batchs)[-1]
        target_graph_embedding = self.target_graph_conv(target_graph_batchs)[-1]


        drug_embedding = torch.cat((drug_seq_embedding, drug_graph_embedding), 1)
        target_embedding = torch.cat((target_seq_embedding, target_graph_embedding), 1)

        drug_output_embedding = self.drug_output_linear(drug_embedding)[-1]
        target_output_embedding = self.target_output_linear(target_embedding)[-1]

        return drug_output_embedding, target_output_embedding


class Predictor(torch.nn.Module):
    def __init__(self, embedding_dim=128, output_dim=1, prediction_mode="cat"):
        super(Predictor, self).__init__()
        print('Predictor Loaded')

        self.prediction_func, prediction_dim_func = vector_operations[prediction_mode]
        mlp_layers_dim = [prediction_dim_func(embedding_dim), 1024, 512, output_dim]

        self.mlp = LinearBlock(mlp_layers_dim, 0.1, relu_layers_index=[0, 1], dropout_layers_index=[0, 1])

    def forward(self, data, drug_embedding, target_embedding):
        drug_id, target_id, y = data.drug_id, data.target_id, data.y

        drug_feature = drug_embedding[drug_id.int().cpu().numpy()]
        target_feature = target_embedding[target_id.int().cpu().numpy()]

        concat_feature = self.prediction_func(drug_feature, target_feature)

        mlp_embeddings = self.mlp(concat_feature)
        link_embeddings = mlp_embeddings[-2]
        out = mlp_embeddings[-1]

        return out, link_embeddings
