"""Standalone simple test to make sure that GNN learning works with globals
"""
from gnn import setup_graph_net
from gnn_dataset import GraphDictDataset, graph_batch_collate
from gnn_utils import train_model, get_model_predictions, visualize_graphs
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch


def create_data(num_samples=100, num_nodes=10, input_node_features=3,
                num_edge_features=1, edge_prob=0.25, infected_prob=0.2,
                friendly_prob=0.5, seed=0):
    """One-hop infection model

    Also, for the globals, count the number of infected people up to 5 
    """
    rng = np.random.RandomState(seed)

    graphs_input = []
    graphs_target = []
    
    for _ in range(num_samples):

        # Nodes
        node_features = rng.binomial(1, p=infected_prob, size=[num_nodes, input_node_features])
        node_targets = node_features[:, 0:1].copy() # initialize special nodes to be special

        graph_input = {}
        graph_input['nodes'] = node_features
        graph_input['n_node'] = np.array(node_features.shape[0])

        graph_target = {}
        graph_target['n_node'] = np.array(node_targets.shape[0])

        # Edges
        receivers, senders, edges = [], [], []

        for idx1 in range(num_nodes):
            for idx2 in range(num_nodes):
                # Create an edge by coin flip
                if rng.random() < edge_prob:
                    receivers.append(idx1)
                    senders.append(idx2)
                    edge_features = np.zeros(num_edge_features)
                    friendly = (rng.random() < friendly_prob)
                    edge_features[0] = friendly
                    edges.append(edge_features)
                    # Update outputs
                    if friendly and (node_features[idx2, 0] == 1):
                        node_targets[idx1, 0] = 1

        graph_target['nodes'] = node_targets

        n_edge = len(edges)
        edges = np.reshape(edges, [n_edge, num_edge_features])
        receivers = np.reshape(receivers, [n_edge]).astype(np.int64)
        senders = np.reshape(senders, [n_edge]).astype(np.int64)
        n_edge = np.reshape(n_edge, [1]).astype(np.int64)

        num_infections = np.sum(node_targets)
        target_global_value = min(num_infections, 4)
        target_globals = np.reshape(target_global_value, [1]).astype(np.int64)
        # target_globals = np.zeros(5, dtype=np.int64)
        # target_globals[target_global_value] = 1

        for graph in [graph_input, graph_target]:
            graph['receivers'] = receivers
            graph['senders'] = senders
            graph['n_edge'] = n_edge
            graph['edges'] = edges

        # Globals
        graph_input['globals'] = np.zeros(5, dtype=np.int64)
        graph_target['globals'] = target_globals

        graphs_input.append(graph_input)
        graphs_target.append(graph_target)

    return graphs_input, graphs_target

def run():
    """
    """
    train_graphs_input, train_graphs_target = create_data(seed=0)
    valid_graphs_input, valid_graphs_target = create_data(seed=1)
    eval_graphs_input, eval_graphs_target = create_data(num_samples=10, seed=2)

    node_color_fn = lambda graph,node,attrs : 'green' if attrs[0] == 1 else 'red'
    edge_color_fn = lambda graph,u,v,attrs : 'green' if attrs[0] == 1 else 'red'

    # for i in range(10):
    #     visualize_graphs(train_graphs_input[i], train_graphs_target[i], '/tmp/train_graph{}.png'.format(i),
    #         node_color_fn=node_color_fn, edge_color_fn=edge_color_fn)
    #     visualize_graphs(valid_graphs_input[i], valid_graphs_target[i], '/tmp/valid_graph{}.png'.format(i),
    #         node_color_fn=node_color_fn, edge_color_fn=edge_color_fn)

    graph_dataset = GraphDictDataset(train_graphs_input, train_graphs_target)
    graph_dataset_val = GraphDictDataset(valid_graphs_input, valid_graphs_target)
    graph_dataset_eval = GraphDictDataset(eval_graphs_input, eval_graphs_target)
    dataloader = DataLoader(graph_dataset, batch_size=16, shuffle=False, num_workers=3, 
        collate_fn=graph_batch_collate)
    dataloader_val = DataLoader(graph_dataset_val, batch_size=16, shuffle=False, num_workers=3, 
        collate_fn=graph_batch_collate)
    dataloader_eval = DataLoader(graph_dataset_eval, batch_size=16, shuffle=False, num_workers=3, 
        collate_fn=graph_batch_collate)
    dataloaders = {'train': dataloader, 'val': dataloader_val, 'eval' : dataloader_eval}

    model = setup_graph_net(graph_dataset, use_gpu=False, num_steps=3)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    global_criterion = nn.CrossEntropyLoss()
    model_weights = train_model(model, dataloaders, criterion=criterion, optimizer=optimizer, 
        use_gpu=False, num_epochs=201, global_criterion=global_criterion)
    model.load_state_dict(model_weights)

    # import torch
    # model.load_state_dict(torch.load('/tmp/model100.pt'))

    predictions = get_model_predictions(model, dataloader_eval)

    for i in range(len(eval_graphs_input)):
        visualize_graphs(eval_graphs_input[i], eval_graphs_target[i], '/tmp/eval_groundtruth_graph{}.png'.format(i),
            node_color_fn=node_color_fn, edge_color_fn=edge_color_fn)
        predicted_target = predictions[i]
        sigmoid = lambda x : 1/(1 + np.exp(-x))
        predicted_target['nodes'] = sigmoid(predicted_target['nodes']) > 0.5
        predicted_target['edges'] = eval_graphs_target[i]['edges'].copy()
        visualize_graphs(eval_graphs_input[i], predicted_target, '/tmp/eval_prediction_graph{}.png'.format(i),
            node_color_fn=node_color_fn, edge_color_fn=edge_color_fn)
        print("groundtruth globals:", eval_graphs_target[i]['globals'])
        print("predicted globals:", predicted_target['globals'])


if __name__ == "__main__":
    run()
