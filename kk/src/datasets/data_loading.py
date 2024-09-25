import os
import numpy as np
import math
import pickle

import torch
import torch_geometric
from torch_geometric.data import download_url
from torch_geometric.datasets import (
    WikipediaNetwork,
    CitationFull,
)
import torch_geometric.transforms as transforms
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from src.datasets.directed_heterophilous_graphs import DirectedHeterophilousGraphDataset
from src.datasets.telegram import Telegram
from src.datasets.data_utils import get_mask
from src.utils.third_party import (
    load_snap_patents_mat,
    even_quantile_labels,
)
from src.datasets.synthetic import get_syn_dataset
from torch_geometric.utils import to_networkx
# Now the inverse operation of to_networkx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
import networkx as nx

def get_dataset(name: str, root_dir: str, homophily=None, undirected=False, self_loops=False, transpose=False,line=True):
    path = f"{root_dir}/"
    evaluator = None

    if name in ["chameleon", "squirrel"]:
        dataset = WikipediaNetwork(root=path, name=name, transform=transforms.NormalizeFeatures())
        dataset._data.y = dataset._data.y.unsqueeze(-1)
    elif name in ["ogbn-arxiv"]:
        dataset = PygNodePropPredDataset(name=name, transform=transforms.ToSparseTensor(), root=path)
        evaluator = Evaluator(name=name)
        split_idx = dataset.get_idx_split()
        dataset._data.train_mask = get_mask(split_idx["train"], dataset._data.num_nodes)
        dataset._data.val_mask = get_mask(split_idx["valid"], dataset._data.num_nodes)
        dataset._data.test_mask = get_mask(split_idx["test"], dataset._data.num_nodes)
    elif name in ["directed-roman-empire"]:
        dataset = DirectedHeterophilousGraphDataset(name=name, transform=transforms.NormalizeFeatures(), root=path)
    elif name == "snap-patents":
        dataset = load_snap_patents_mat(n_classes=5, root=path)
    elif name == "arxiv-year":
        # arxiv-year uses the same graph and features as ogbn-arxiv, but with different labels
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", transform=transforms.ToSparseTensor(), root=path)
        evaluator = Evaluator(name="ogbn-arxiv")
        y = even_quantile_labels(dataset._data.node_year.flatten().numpy(), nclasses=5, verbose=False)
        dataset._data.y = torch.as_tensor(y).reshape(-1, 1)
        # Tran, val and test masks are required during preprocessing. Setting them here to dummy values as
        # they are overwritten later for this dataset (see get_dataset_split function below)
        dataset._data.train_mask, dataset._data.val_mask, dataset._data.test_mask = 0, 0, 0
        # Create directory for this dataset
        os.makedirs(os.path.join(path, name.replace("-", "_"), "raw"), exist_ok=True)
    elif name == "syn-dir":
        dataset = get_syn_dataset(path)

    elif name in ["cora_ml", "citeseer_full"]:
        if name == "citeseer_full":
            name = "citeseer"
        dataset = CitationFull(path, name)
    elif name == "telegram":
        dataset = Telegram(path)
    else:
        raise Exception("Unknown dataset.")

    if undirected:
        dataset._data.edge_index = torch_geometric.utils.to_undirected(dataset._data.edge_index)
    if self_loops:
        dataset._data.edge_index, _ = torch_geometric.utils.add_self_loops(dataset._data.edge_index)
    if transpose:
        dataset._data.edge_index = torch.stack([dataset._data.edge_index[1], dataset._data.edge_index[0]])
    if line:
        print('===========================================================================================================')
        print('=============================== Creating Line Graph =====================================================')
        print('===========================================================================================================')
        original = to_networkx(dataset._data, to_undirected=False,to_multi=True)
        print('Original Graph: ')
        print(original)

        linegraph = nx.line_graph(original)
        # We add the self loops in term
        print('Line Graph: ')
        print('======================')
        print(f'Number of nodes: {linegraph.number_of_nodes()}')
        print(f'Number of edges: {linegraph.number_of_edges()}')
        print()
        # Now we parse to  edge_index the line graph in numpy
        line_edge_index  = from_networkx(linegraph).edge_index
        # Now we have that the original edge index is [2, 38378] and the line nodes is 38328, we need to remove them
        print('Original Edge Index: ')
        print('======================')
        print(dataset._data.edge_index.shape)
        print('Edge Index: ')
        print('======================')
        print(line_edge_index.shape)
        print()
        # Ahora para cada nodo del line graph, se le asigna la característica del nodo destino en el grafo original y el nodo origen
        # line_features = torch.zeros((linegraph.number_of_nodes(),2*dataset._data.num_features),dtype=torch.float32)
        # for i in range(linegraph.number_of_nodes()):
        #     line_features[i] = torch.cat([dataset._data.x[dataset._data.edge_index[0][i]],dataset._data.x[dataset._data.edge_index[1][i]]])
        #     #line_features[i] =  data.x[data.edge_index[1][i]]
        line_features = torch.zeros((linegraph.number_of_nodes(), dataset.num_features), dtype=dataset._data.x.dtype)
        edge_index_dst = dataset._data.edge_index[1]
        line_features = dataset._data.x[edge_index_dst]
        dataset._data.line_data = [line_features,line_edge_index]
        print('Line Data: ')
        print('======================')
        print(dataset._data)
        print('===========================================================================================================')

    return dataset, evaluator


def get_dataset_split(name, data, root_dir, split_number):
    if name in ["snap-patents", "chameleon", "squirrel", "telegram", "directed-roman-empire"]:
        return (
            data["train_mask"][:, split_number],
            data["val_mask"][:, split_number],
            data["test_mask"][:, split_number],
        )
    if name in ["ogbn-arxiv"]:
        # OGBN datasets have a single pre-assigned split
        return data["train_mask"], data["val_mask"], data["test_mask"]
    if name in ["arxiv-year"]:
        # Datasets from https://arxiv.org/pdf/2110.14446.pdf have five splits stored
        # in https://github.com/CUAI/Non-Homophily-Large-Scale/tree/82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6/data/splits
        num_nodes = data["y"].shape[0]
        github_url = f"https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data/splits/"
        split_file_name = f"{name}-splits.npy"
        local_dir = os.path.join(root_dir, name.replace("-", "_"), "raw")

        download_url(os.path.join(github_url, split_file_name), local_dir, log=False)
        splits = np.load(os.path.join(local_dir, split_file_name), allow_pickle=True)
        split_idx = splits[split_number % len(splits)]

        train_mask = get_mask(split_idx["train"], num_nodes)
        val_mask = get_mask(split_idx["valid"], num_nodes)
        test_mask = get_mask(split_idx["test"], num_nodes)

        return train_mask, val_mask, test_mask
    elif name in ["syn-dir", "cora_ml", "citeseer_full"]:
        # Uniform 50/25/25 split
        return set_uniform_train_val_test_split(split_number, data, train_ratio=0.5, val_ratio=0.25)


def set_uniform_train_val_test_split(seed, data, train_ratio=0.5, val_ratio=0.25):
    rnd_state = np.random.RandomState(seed)
    num_nodes = data.y.shape[0]

    # Some nodes have labels -1 (i.e. unlabeled), so we need to exclude them
    labeled_nodes = torch.where(data.y != -1)[0]
    num_labeled_nodes = labeled_nodes.shape[0]
    num_train = math.floor(num_labeled_nodes * train_ratio)
    num_val = math.floor(num_labeled_nodes * val_ratio)

    idxs = list(range(num_labeled_nodes))
    # Shuffle in place
    rnd_state.shuffle(idxs)

    train_idx = idxs[:num_train]
    val_idx = idxs[num_train : num_train + num_val]
    test_idx = idxs[num_train + num_val :]

    train_idx = labeled_nodes[train_idx]
    val_idx = labeled_nodes[val_idx]
    test_idx = labeled_nodes[test_idx]

    train_mask = get_mask(train_idx, num_nodes)
    val_mask = get_mask(val_idx, num_nodes)
    test_mask = get_mask(test_idx, num_nodes)

    return train_mask, val_mask, test_mask
