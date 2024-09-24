import os
import numpy as np
import uuid

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint

from src.utils.utils import use_best_hyperparams, get_available_accelerator
from src.datasets.data_loading import get_dataset, get_dataset_split
from src.datasets.dataset import FullBatchGraphDataset
from src.model import get_model, LightingFullBatchModelWrapper
from src.utils.arguments import args

from torch_geometric.utils import to_networkx
# Now the inverse operation of to_networkx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
import networkx as nx

def run(args):
    torch.manual_seed(0)

    # Get dataset and dataloader
    dataset, evaluator = get_dataset(
        name=args.dataset,
        root_dir=args.dataset_directory,
        undirected=args.undirected,
        self_loops=args.self_loops,
        transpose=args.transpose,
    )
    data = dataset._data
    # train_masks, val_masks, test_masks = [], [], []
    # for num_run in range(args.num_runs):
    #     train_mask, val_mask, test_mask = get_dataset_split(args.dataset, data, args.dataset_directory, num_run)
    #     train_masks.append(train_mask.clone())
    #     val_masks.append(val_mask.clone())
    #     test_masks.append(test_mask.clone())
    # train_masks = torch.stack(train_masks)
    # val_masks = torch.stack(val_masks)
    # test_masks = torch.stack(test_masks)
    # print(train_masks.shape)
    # print(val_masks.shape)
    # print(test_masks.shape)

    print('===========================================================================================================')
    print('=============================== Creating Line Graph =====================================================')
    print('===========================================================================================================')
    original = to_networkx(data, to_undirected=False,to_multi=True)
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
    print(data.edge_index.shape)
    print('Edge Index: ')
    print('======================')
    print(line_edge_index.shape)
    print()
    # # Ahora para cada nodo del line graph, se le asigna la característica del nodo destino en el grafo original y el nodo origen
    # line_features = torch.zeros((linegraph.number_of_nodes(),data.num_features),dtype=torch.float32)
    # line_labels = torch.zeros((linegraph.number_of_nodes()),dtype=torch.long)
    # for i in range(linegraph.number_of_nodes()):
    #     line_features[i] = data.x[data.edge_index[1][i]]#torch.cat([data.x[data.edge_index[1][i]],data.x[data.edge_index[0][i]]])
    #     line_labels = data.y[data.edge_index[1]]
    # for num_run in range(args.num_runs):
    #     for i in range(linegraph.number_of_nodes()):
    #         line_train_masks[num_run][i] = train_masks[num_run][data.edge_index[1][i]]
    #         line_val_masks[num_run][i] = val_masks[num_run][data.edge_index[1][i]]
    #         line_test_masks[num_run][i] = test_masks[num_run][data.edge_index[1][i]]
    # # Inicializar las matrices para las características, etiquetas y máscaras
    line_features = torch.zeros((linegraph.number_of_nodes(), data.num_features), dtype=data.x.dtype)
    line_labels = torch.zeros((linegraph.number_of_nodes()), dtype=data.y.dtype)
    # line_train_masks = torch.zeros((args.num_runs, linegraph.number_of_nodes()), dtype=train_masks.dtype)
    # line_val_masks = torch.zeros((args.num_runs, linegraph.number_of_nodes()), dtype=train_masks.dtype)
    # line_test_masks = torch.zeros((args.num_runs, linegraph.number_of_nodes()), dtype=train_masks.dtype)

    # Asignar características y etiquetas para todos los nodos del line graph de manera vectorizada
    edge_indices = data.edge_index[1]
    line_features = data.x[edge_indices]
    line_labels = data.y[edge_indices]

    # Asignar máscaras de entrenamiento, validación y prueba de manera vectorizada
    # line_train_masks = train_masks[:, edge_indices]
    # line_val_masks = val_masks[:, edge_indices]
    # line_test_masks = test_masks[:, edge_indices]
    line_edge_index = line_edge_index.to(data.x.device)
    line_data = Data(x=line_features,edge_index=line_edge_index,y=line_labels)
    print('Line Data: ')
    print('======================')
    print(line_data)
    print('===========================================================================================================')
    data = line_data
    dataset._data = data
    print('Data: ')
    print('======================')
    print(dataset._data)
    print('===========================================================================================================')
    print('=============================== End Line Graph =====================================================')
    # train_masks = line_train_masks
    # val_masks = line_val_masks
    # test_masks = line_test_masks
    data_loader = DataLoader(FullBatchGraphDataset(data), batch_size=1, collate_fn=lambda batch: batch[0])
    val_accs, test_accs = [], []
    for num_run in range(args.num_runs):
        # Get train/val/test splits for the current run
        # train_mask = train_masks[num_run]
        # val_mask = val_masks[num_run]
        # test_mask = test_masks[num_run]    
        train_mask, val_mask, test_mask = get_dataset_split(args.dataset, data, args.dataset_directory, num_run)

        # Get model
        args.num_features, args.num_classes = data.num_features, dataset.num_classes
        model = get_model(args)
        lit_model = LightingFullBatchModelWrapper(
            model=model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            evaluator=evaluator,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

        # Setup Pytorch Lighting Callbacks
        early_stopping_callback = EarlyStopping(monitor="val_acc", mode="max", patience=args.patience)
        model_summary_callback = ModelSummary(max_depth=-1)
        if not os.path.exists(f"{args.checkpoint_directory}/"):
            os.mkdir(f"{args.checkpoint_directory}/")
        model_checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            dirpath=f"{args.checkpoint_directory}/{str(uuid.uuid4())}/",
        )

        # Setup Pytorch Lighting Trainer
        trainer = pl.Trainer(
            log_every_n_steps=1,
            max_epochs=args.num_epochs,
            callbacks=[
                early_stopping_callback,
                model_summary_callback,
                model_checkpoint_callback,
            ],
            profiler="simple" if args.profiler else None,
            accelerator=get_available_accelerator(),
            devices=[args.gpu_idx],
        )

        # Fit the model
        trainer.fit(model=lit_model, train_dataloaders=data_loader)

        # Compute validation and test accuracy
        val_acc = model_checkpoint_callback.best_model_score.item()
        test_acc = trainer.test(ckpt_path="best", dataloaders=data_loader)[0]["test_acc"]
        test_accs.append(test_acc)
        val_accs.append(val_acc)

    print(f"Test Acc: {np.mean(test_accs)} +- {np.std(test_accs)}")


if __name__ == "__main__":
    args = use_best_hyperparams(args, args.dataset) if args.use_best_hyperparams else args
    run(args)
