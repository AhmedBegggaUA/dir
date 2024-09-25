import os
import numpy as np
import math
import pickle
import torch.nn.functional as F
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
from src.homophily import get_node_homophily
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
        original = to_networkx(dataset._data, to_undirected=False,to_multi=False)
        print('Original Graph: ')
        print(original)
        # import networkx as nx
        # import numpy as np
        # from sklearn.metrics.pairwise import cosine_similarity

        # def surgical_linegraph_edge_replacement(G, node_features, replacement_ratio=0.1):
        #     """
        #     Reemplaza quirúrgicamente un ratio de aristas del grafo original con nuevas aristas
        #     sugeridas por el linegraph y las características de los nodos.
            
        #     :param G: Grafo dirigido original (networkx.DiGraph)
        #     :param node_features: Diccionario de características de nodos {node_id: feature_vector}
        #     :param replacement_ratio: Proporción de aristas a reemplazar
        #     :return: Grafo dirigido modificado
        #     """
            
        #     # Paso 1: Crear el linegraph
        #     L = nx.line_graph(G)
            
        #     # Paso 2: Calcular similitudes de características para todas las aristas potenciales en el linegraph
        #     edge_similarities = {}
        #     for edge in L.edges():
        #         # Aquí 'edge' representa una conexión entre dos aristas del grafo original
        #         (u1, v1), (u2, v2) = edge
        #         if v1 == u2:  # Aseguramos que las aristas están conectadas en el grafo original
        #             sim = cosine_similarity([node_features[u1]], [node_features[v2]])[0][0]
        #             edge_similarities[edge] = sim
                    
        #     # Paso 3: Identificar aristas potenciales basadas en el linegraph
        #     potential_edges = set((u1, v2) for ((u1, v1), (u2, v2)) in edge_similarities.keys())
        #     new_edges = potential_edges - set(G.edges())
            
        #     # Paso 4: Calcular la "fuerza" de las aristas existentes
        #     existing_edge_strength = {}
        #     for u, v in G.edges():
        #         neighborhood_sim = np.mean([
        #             edge_similarities.get(((u, v), (v, w)), 0)
        #             for w in G.successors(v)
        #         ])
        #         existing_edge_strength[(u, v)] = neighborhood_sim
            
        #     # Paso 5: Ordenar las aristas existentes por fuerza (ascendente) y las nuevas por similitud (descendente)
        #     existing_edges_sorted = sorted(existing_edge_strength.items(), key=lambda x: x[1])
        #     new_edges_sorted = sorted(
        #                     [
        #                         (u, v, max(
        #                             [edge_similarities.get(((w, u), (u, v)), 0) for w in G.predecessors(u)] or [-float('inf')]
        #                         ))
        #                         for u, v in new_edges
        #                     ],
        #                     key=lambda x: x[2],
        #                     reverse=True
        #                 )

            
        #     # Paso 6: Determinar el número de aristas a reemplazar
        #     num_edges_to_replace = int(replacement_ratio * G.number_of_edges())
            
        #     # Paso 7: Realizar el reemplazo quirúrgico
        #     edges_to_remove = [edge for edge, _ in existing_edges_sorted[:num_edges_to_replace]]
        #     edges_to_add = [edge[:2] for edge in new_edges_sorted[:num_edges_to_replace]]
            
        #     # Paso 8: Crear el grafo modificado
        #     G_modified = G.copy()
        #     G_modified.remove_edges_from(edges_to_remove)
        #     G_modified.add_edges_from(edges_to_add)
            
        #     return G_modified

        import networkx as nx
        import numpy as np
        from scipy.spatial.distance import cosine

        def linegraph_sparsification_directed(G, node_features, ratio=0.5, alpha=0.5, temperature=1.0):
            """
            Realiza la esparsificación de un grafo dirigido utilizando su linegraph y características de nodos.
            Incluye manejo de valores NaN y cero.
            
            :param G: Grafo dirigido original (networkx.DiGraph)
            :param node_features: Diccionario de características de nodos {node_id: feature_vector}
            :param ratio: Proporción de aristas a mantener
            :param alpha: Peso para combinar probabilidad topológica y similitud de características
            :param temperature: Controla la aleatoriedad en la selección final
            :return: Grafo dirigido esparsificado
            """
            
            # Paso 1: Crear el linegraph
            L = nx.line_graph(G)
            
            # Paso 2: Calcular probabilidades basadas en grado
            def sample_edges_degree(G):
                edges = list(G.edges)
                num_nodes = len(G.nodes)
                in_degree = dict(G.in_degree())
                out_degree = dict(G.out_degree())
                
                prob = [(0.5 / num_nodes) * (1.0 / (out_degree[edge[0]] + 1)) + 
                        (1.0 / (in_degree[edge[1]] + 1)) for edge in edges]
                return np.array(prob)
            
            prob_topological = sample_edges_degree(G)
            
            # Paso 3: Calcular similitud de coseno entre nodos conectados
            def cosine_similarity(features1, features2):
                if np.allclose(features1, features2):
                    return 1.0
                return max(0, 1 - cosine(features1, features2))  # Asegurar que no sea negativo
            
            edge_similarities = {}
            for u, v in G.edges():
                sim = cosine_similarity(node_features[u], node_features[v])
                edge_similarities[(u, v)] = sim
            
            # Normalizar similitudes
            similarities = list(edge_similarities.values())
            max_sim = max(similarities)
            min_sim = min(similarities)
            if max_sim != min_sim:
                for edge in edge_similarities:
                    edge_similarities[edge] = (edge_similarities[edge] - min_sim) / (max_sim - min_sim)
            else:
                for edge in edge_similarities:
                    edge_similarities[edge] = 1.0  # Si todas las similitudes son iguales, establecerlas en 1
            
            # Paso 4: Combinar probabilidades y similitudes
            #prob_final = alpha * prob_topological + (1 - alpha) * np.array([edge_similarities[edge] for edge in G.edges()])
            prob_final = prob_topological * np.array([edge_similarities[edge] for edge in G.edges()])
            # Manejar posibles NaN o infinitos
            prob_final = np.nan_to_num(prob_final, nan=0.0, posinf=1.0, neginf=0.0)
            prob_final = 1 - prob_final
            # Asegurarse de que las probabilidades no sean todas cero
            if np.all(prob_final == 0):
                prob_final = np.ones_like(prob_final)
            
            # Normalizar probabilidades finales
            prob_final /= prob_final.sum()
            # Damos la vuelta a las probabilidades para que sea más probable que se mantengan las aristas más importantes
            prob_final  = 1 - prob_final
            # Paso 5: Aplicar temperatura y seleccionar aristas
            prob_final = np.exp(np.log(prob_final + 1e-10) / temperature)  # Añadir pequeño valor para evitar log(0)
            prob_final /= prob_final.sum()
            
            num_edges_to_keep = int(ratio * G.number_of_edges())
            selected_edges = np.random.choice(G.number_of_edges(), size=num_edges_to_keep, replace=False, p=prob_final)
            
            # Paso 6: Crear el grafo esparsificado
            G_sparse = nx.DiGraph()
            G_sparse.add_nodes_from(G.nodes(data=True))
            for i, (u, v) in enumerate(G.edges()):
                if i in selected_edges:
                    G_sparse.add_edge(u, v)
            
            return G_sparse
        # Iteramos hasta 10 y vamos de 0.5 en 0.5
        i = 0
        max_homo = get_node_homophily(dataset._data.y, dataset._data.edge_index)
        best_rat = 1
        best_edge = None
        while i < 1:
            print('Iteration: ', i)
            G_sparse = linegraph_sparsification_directed(original, dataset._data.x.numpy(), ratio=i, alpha=0.2, temperature=1.0)
            #G_nigger = surgical_linegraph_edge_replacement(original, dataset._data.x.numpy())
            new_edge_index = from_networkx(G_sparse).edge_index
            print('Edge Index: ')
            print('======================')
            print(new_edge_index.shape)
            print()
            print('======================')
            print("Old node homophily: ", get_node_homophily(dataset._data.y, dataset._data.edge_index))
            print('======================')
            new_homo = get_node_homophily(dataset._data.y, new_edge_index)
            print("New node homophily: ", new_homo)
            if max_homo < new_homo:
                best_rat = i
                best_edge = new_edge_index

            i+=0.001
        print("Ratio ganador",best_rat)
        dataset._data.edge_index = best_edge
        
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
