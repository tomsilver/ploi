"""Search guidance using a GNN.
"""

import pickle
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.optim
import torch.nn
import torch
import pddlgym
from pddlgym.structs import Predicate
from PLOI.gnn.gnn import setup_graph_net
from PLOI.gnn.gnn_dataset import GraphDictDataset, graph_batch_collate
from PLOI.gnn.gnn_utils import train_model, get_single_model_prediction
from PLOI.guidance import BaseSearchGuidance
from PLOI.planning import PlanningTimeout, PlanningFailure


class GNNSearchGuidance(BaseSearchGuidance):
    """Search guidance using a GNN.
    """
    def __init__(self, training_planner, num_train_problems, num_epochs,
                 criterion_name, bce_pos_weight, load_from_file,
                 load_dataset_from_file, dataset_file_prefix,
                 save_model_prefix, is_strips_domain):
        super().__init__()
        self._planner = training_planner
        self._num_train_problems = num_train_problems
        self._num_epochs = num_epochs
        self._criterion_name = criterion_name
        self._bce_pos_weight = bce_pos_weight
        self._load_from_file = load_from_file
        self._load_dataset_from_file = load_dataset_from_file
        self._dataset_file_prefix = dataset_file_prefix
        self._save_model_prefix = save_model_prefix
        self._is_strips_domain = is_strips_domain
        # Initialize other instance variables.
        self._model = None
        self._unary_types = None
        self._unary_predicates = None
        self._binary_predicates = None
        self._node_feature_to_index = None
        self._edge_feature_to_index = None
        self._last_processed_state = None
        self._last_object_scores = None
        self._num_node_features = None
        self._num_edge_features = None

    def train(self, train_env_name):
        model_outfile = self._save_model_prefix+"_{}.pt".format(train_env_name)
        print("Training search guidance {} in domain {}...".format(
            self.__class__.__name__, train_env_name))
        # Collect raw training data. Inputs are States, outputs are objects.
        training_data = self._collect_training_data(train_env_name)
        # Convert training data to graphs
        graphs_input, graphs_target = self._create_graph_dataset(training_data)
        # Use 10% for validation
        num_validation = max(1, int(len(graphs_input)*0.1))
        train_graphs_input = graphs_input[num_validation:]
        train_graphs_target = graphs_target[num_validation:]
        valid_graphs_input = graphs_input[:num_validation]
        valid_graphs_target = graphs_target[:num_validation]
        # Set up dataloaders
        graph_dataset = GraphDictDataset(train_graphs_input,
                                         train_graphs_target)
        graph_dataset_val = GraphDictDataset(valid_graphs_input,
                                             valid_graphs_target)
        dataloader = DataLoader(graph_dataset, batch_size=16, shuffle=False,
                                num_workers=3, collate_fn=graph_batch_collate)
        dataloader_val = DataLoader(graph_dataset_val, batch_size=16,
                                    shuffle=False, num_workers=3,
                                    collate_fn=graph_batch_collate)
        dataloaders = {"train": dataloader, "val": dataloader_val}
        # Set up model, loss, optimizer
        self._model = setup_graph_net(graph_dataset, use_gpu=False, num_steps=3)

        if not self._load_from_file or not os.path.exists(model_outfile):
            optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
            if self._criterion_name == "bce":
                pos_weight = self._bce_pos_weight*torch.ones([1])
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                raise Exception("Unrecognized criterion_name {}".format(
                    self._criterion_name))
            # Train model
            model_dict = train_model(self._model, dataloaders,
                                     criterion=criterion, optimizer=optimizer,
                                     use_gpu=False, num_epochs=self._num_epochs)
            torch.save(model_dict, model_outfile)
            self._model.load_state_dict(model_dict)
            print("Saved model to {}.".format(model_outfile))
        else:
            self._model.load_state_dict(torch.load(model_outfile))
            print("Loaded saved model from {}.".format(model_outfile))

    def seed(self, seed):
        torch.manual_seed(seed)

    def score_object(self, obj, state):
        if state != self._last_processed_state:
            # Create input graph from state
            graph, node_to_objects = self._state_to_graph(state)
            # Predict graph
            prediction = self._predict_graph(graph)
            # Derive object scores
            object_scores = {o: prediction["nodes"][n][0]
                             for n, o in node_to_objects.items()}
            self._last_object_scores = object_scores
            self._last_processed_state = state
        return self._last_object_scores[obj]

    def _collect_training_data(self, train_env_name):
        """Returns X, Y where X are States and Y are sets of objects
        """
        outfile = self._dataset_file_prefix + "_{}.pkl".format(train_env_name)
        if not self._load_dataset_from_file or not os.path.exists(outfile):
            inputs = []
            outputs = []
            env = pddlgym.make("PDDLEnv{}-v0".format(train_env_name))
            assert env.operators_as_actions
            for idx in range(min(self._num_train_problems, len(env.problems))):
                print("Collecting training data problem {}".format(idx),
                      flush=True)
                env.fix_problem_index(idx)
                state, _ = env.reset()
                try:
                    plan = self._planner(env.domain, state, timeout=60)
                except (PlanningTimeout, PlanningFailure):
                    print("Warning: planning failed, skipping: {}".format(
                        env.problems[idx].problem_fname))
                    continue
                inputs.append(state)
                objects_in_plan = {o for act in plan for o in act.variables}
                outputs.append(objects_in_plan)
            training_data = (inputs, outputs)

            with open(outfile, "wb") as f:
                pickle.dump(training_data, f)

        with open(outfile, "rb") as f:
            training_data = pickle.load(f)

        return training_data

    def _state_to_graph(self, state):
        """Create a graph from a State
        """
        assert self._node_feature_to_index is not None, "Must initialize first"
        all_objects = sorted(state.objects)
        node_to_objects = dict(enumerate(all_objects))
        objects_to_node = {v: k for k, v in node_to_objects.items()}
        num_objects = len(all_objects)

        G = self.wrap_goal_literal
        R = self.reverse_binary_literal

        graph_input = {}

        # Nodes: one per object
        graph_input["n_node"] = np.array(num_objects)
        input_node_features = np.zeros((num_objects, self._num_node_features))

        # Add features for types
        for obj_index, obj in enumerate(all_objects):
            type_index = self._node_feature_to_index[obj.var_type]
            input_node_features[obj_index, type_index] = 1

        # Add features for unary state literals
        for lit in state.literals:
            if lit.predicate.arity != 1:
                continue
            lit_index = self._node_feature_to_index[lit.predicate]
            assert len(lit.variables) == 1
            obj_index = objects_to_node[lit.variables[0]]
            input_node_features[obj_index, lit_index] = 1

        # Add features for unary goal literals
        for lit in state.goal.literals:
            if lit.predicate.arity != 1:
                continue
            lit_index = self._node_feature_to_index[G(lit.predicate)]
            assert len(lit.variables) == 1
            obj_index = objects_to_node[lit.variables[0]]
            input_node_features[obj_index, lit_index] = 1

        graph_input["nodes"] = input_node_features

        # Edges
        all_edge_features = np.zeros((num_objects, num_objects,
                                      self._num_edge_features))

        # Add edge features for binary state literals
        for bin_lit in state.literals:
            if bin_lit.predicate.arity != 2:
                continue
            for lit in [bin_lit, R(bin_lit)]:
                pred_index = self._edge_feature_to_index[lit.predicate]
                assert len(lit.variables) == 2
                obj0_index = objects_to_node[lit.variables[0]]
                obj1_index = objects_to_node[lit.variables[1]]
                all_edge_features[obj0_index, obj1_index, pred_index] = 1

        # Add edge features for binary goal literals
        for bin_lit in state.goal.literals:
            if bin_lit.predicate.arity != 2:
                continue
            for lit in [G(bin_lit), G(R(bin_lit))]:
                pred_index = self._edge_feature_to_index[lit.predicate]
                assert len(lit.variables) == 2
                obj0_index = objects_to_node[lit.variables[0]]
                obj1_index = objects_to_node[lit.variables[1]]
                all_edge_features[obj0_index, obj1_index, pred_index] = 1

        # Organize into expected representation
        adjacency_mat = np.any(all_edge_features, axis=2)
        receivers, senders, edges = [], [], []
        for sender, receiver in np.argwhere(adjacency_mat):
            edge = all_edge_features[sender, receiver]
            senders.append(sender)
            receivers.append(receiver)
            edges.append(edge)

        n_edge = len(edges)
        edges = np.reshape(edges, [n_edge, self._num_edge_features])
        receivers = np.reshape(receivers, [n_edge]).astype(np.int64)
        senders = np.reshape(senders, [n_edge]).astype(np.int64)
        n_edge = np.reshape(n_edge, [1]).astype(np.int64)

        graph_input["receivers"] = receivers
        graph_input["senders"] = senders
        graph_input["n_edge"] = n_edge
        graph_input["edges"] = edges

        # Globals
        graph_input["globals"] = None

        return graph_input, node_to_objects

    def _predict_graph(self, input_graph):
        """Predict the target graph given the input graph
        """
        assert self._model is not None, "Must train before calling predict"
        prediction = get_single_model_prediction(self._model, input_graph)
        # Apply sigmoids
        sigmoid = lambda x: 1/(1 + np.exp(-x))
        prediction["nodes"] = sigmoid(prediction["nodes"])
        # We're not predicting edges
        prediction["edges"] = input_graph["edges"].copy()
        return prediction

    @classmethod
    def wrap_goal_literal(cls, x):
        """Helper for converting a state to required input representation
        """
        if isinstance(x, Predicate):
            return Predicate("WANT"+x.name, x.arity, var_types=x.var_types,
                             is_negative=x.is_negative, is_anti=x.is_anti)
        new_predicate = cls.wrap_goal_literal(x.predicate)
        return new_predicate(*x.variables)

    @classmethod
    def reverse_binary_literal(cls, x):
        """Helper for converting a state to required input representation
        """
        if isinstance(x, Predicate):
            assert x.arity == 2
            return Predicate("REV"+x.name, x.arity, var_types=x.var_types,
                             is_negative=x.is_negative, is_anti=x.is_anti)
        new_predicate = cls.reverse_binary_literal(x.predicate)
        variables = [v for v in x.variables]
        assert len(variables) == 2
        return new_predicate(*variables[::-1])

    def _create_graph_dataset(self, training_data):
        # Initialize the graph features

        # First get the types and predicates
        self._unary_types = set()
        self._unary_predicates = set()
        self._binary_predicates = set()

        for state in training_data[0]:
            types = {o.var_type for o in state.objects}
            self._unary_types.update(types)
            for lit in set(state.literals) | set(state.goal.literals):
                arity = lit.predicate.arity
                assert arity == len(lit.variables)
                assert arity <= 2, "Arity > 2 predicates not yet supported"
                if arity == 0:
                    continue
                elif arity == 1:
                    self._unary_predicates.add(lit.predicate)
                elif arity == 2:
                    self._binary_predicates.add(lit.predicate)

        self._unary_types = sorted(self._unary_types)
        self._unary_predicates = sorted(self._unary_predicates)
        self._binary_predicates = sorted(self._binary_predicates)

        G = self.wrap_goal_literal
        R = self.reverse_binary_literal

        # Initialize node features
        self._node_feature_to_index = {}
        index = 0
        for unary_type in self._unary_types:
            self._node_feature_to_index[unary_type] = index
            index += 1
        for unary_predicate in self._unary_predicates:
            self._node_feature_to_index[unary_predicate] = index
            index += 1
        for unary_predicate in self._unary_predicates:
            self._node_feature_to_index[G(unary_predicate)] = index
            index += 1

        # Initialize edge features
        self._edge_feature_to_index = {}
        index = 0
        for binary_predicate in self._binary_predicates:
            self._edge_feature_to_index[binary_predicate] = index
            index += 1
        for binary_predicate in self._binary_predicates:
            self._edge_feature_to_index[R(binary_predicate)] = index
            index += 1
        for binary_predicate in self._binary_predicates:
            self._edge_feature_to_index[G(binary_predicate)] = index
            index += 1
        for binary_predicate in self._binary_predicates:
            self._edge_feature_to_index[G(R(binary_predicate))] = index
            index += 1

        self._num_node_features = len(self._node_feature_to_index)
        nnf = self._num_node_features
        assert max(self._node_feature_to_index.values()) == nnf-1
        self._num_edge_features = len(self._edge_feature_to_index)
        nef = self._num_edge_features
        assert max(self._edge_feature_to_index.values()) == nef-1

        # Process data
        num_training_examples = len(training_data[0])

        graphs_input = []
        graphs_target = []

        for i in range(num_training_examples):
            state = training_data[0][i]
            target_object_set = training_data[1][i]
            graph_input, node_to_objects = self._state_to_graph(state)
            graph_target = {
                "n_node": graph_input["n_node"],
                "n_edge": graph_input["n_edge"],
                "edges": graph_input["edges"],
                "senders": graph_input["senders"],
                "receivers": graph_input["receivers"],
                "globals": graph_input["globals"],
            }

            # Target nodes
            objects_to_node = {v: k for k, v in node_to_objects.items()}
            object_mask = np.zeros((len(node_to_objects), 1), dtype=np.int64)
            for o in target_object_set:
                obj_index = objects_to_node[o]
                object_mask[obj_index] = 1
            graph_target["nodes"] = object_mask

            graphs_input.append(graph_input)
            graphs_target.append(graph_target)

        return graphs_input, graphs_target
