import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm
from core.attention.extractors import AttentionExtractor, AttributeAttentionExtractor
import pickle
import os
import pathlib
from utils.result_collector import BinaryClassificationResultsAggregator


class AttentionGraphUtils(object):
    """
    Code adapted from https://github.com/samiraabnar/attention_flow
    """

    @staticmethod
    def get_adjmat(mat, input_text_units):
        n_layers, length, _ = mat.shape
        adj_mat = np.zeros(((n_layers + 1) * length, (n_layers + 1) * length))
        labels_to_index = {}
        for k in np.arange(length):
            labels_to_index[str(k) + "_" + input_text_units[k]] = k

        for i in np.arange(1, n_layers + 1):
            for k_f in np.arange(length):
                index_from = (i) * length + k_f
                label = "L" + str(i) + "_" + str(k_f)
                labels_to_index[label] = index_from
                for k_t in np.arange(length):
                    index_to = (i - 1) * length + k_t
                    adj_mat[index_from][index_to] = mat[i - 1][k_f][k_t]

        return adj_mat, labels_to_index

    @staticmethod
    def create_attention_graph(adjmat):
        A = adjmat
        G = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
        for i in np.arange(A.shape[0]):
            for j in np.arange(A.shape[1]):
                nx.set_edge_attributes(G, {(i, j): A[i, j]}, 'capacity')

        return G

    @staticmethod
    def compute_flows(G, labels_to_index, input_nodes, length):
        number_of_nodes = len(labels_to_index)
        flow_values = np.zeros((number_of_nodes, number_of_nodes))
        for key in labels_to_index:
            if key not in input_nodes:
                current_layer = int(labels_to_index[key] / length)
                pre_layer = current_layer - 1
                u = labels_to_index[key]
                for inp_node_key in input_nodes:
                    v = labels_to_index[inp_node_key]
                    flow_value = nx.maximum_flow_value(G, u, v, flow_func=nx.algorithms.flow.edmonds_karp)
                    flow_values[u][pre_layer * length + v] = flow_value
                flow_values[u] /= flow_values[u].sum()

        return flow_values

    @staticmethod
    def plot_attention_graph(G, adjmat, labels_to_index, n_layers, length, plot_top_scores=None, ax=None):

        top_score_selection_meths = ['percentile']
        if plot_top_scores is not None:
            assert plot_top_scores in top_score_selection_meths

        A = adjmat
        pos = {}
        label_pos = {}
        for i in np.arange(n_layers + 1):
            for k_f in np.arange(length):
                pos[i * length + k_f] = ((i + 0.4) * 2, length - k_f)
                label_pos[i * length + k_f] = (i * 2 - 1, length - k_f)

        index_to_labels = {}
        for key in labels_to_index:
            index_to_labels[labels_to_index[key]] = key.split("_")[-1]
            if labels_to_index[key] >= length:
                index_to_labels[labels_to_index[key]] = ''

        if ax is None:
            node_size = 50
            label_font_size = 18
        else:
            node_size = 20
            label_font_size = 12

        nx.draw_networkx_nodes(G, pos, node_color='green', node_size=node_size, ax=ax)  # , labels=index_to_labels, node_size=50)
        nx.draw_networkx_labels(G, pos=label_pos, labels=index_to_labels, font_size=label_font_size, ax=ax,
                                horizontalalignment='left')

        max_attentions = {}
        if plot_top_scores is not None:
            if plot_top_scores == 'percentile':
                for i in np.arange(1, n_layers + 1):
                    attn_map = A[i * length: (i + 1) * length, (i - 1) * length: i * length]
                    thr = np.quantile(attn_map, 0.8)
                    max_flat_idxs = np.where(attn_map >= thr)
                    max_idxs = list(zip(max_flat_idxs[0] + i * length, max_flat_idxs[1] + (i - 1) * length))
                    for max_idx in max_idxs:
                        max_attentions[max_idx] = A[max_idx[0], max_idx[1]]
            else:
                raise NotImplementedError()

        weights = []
        edges = []
        # 4 a. Iterate through the graph nodes to gather all the weights
        for (node1, node2, data) in G.edges(data=True):
            if len(max_attentions) > 0:
                if (node1, node2) in max_attentions:
                    weights.append(data['weight'])
                    edges.append((node1, node2))
            else:
                weights.append(data['weight'])  # we'll use this when determining edge thickness
                edges.append((node1, node2))

        nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color='darkblue', ax=ax)


class AttentionGraphExtractor(object):
    available_text_units = ['token', 'word', 'attr']

    def __init__(self, attn_extractor: AttentionExtractor, text_unit: str):

        assert isinstance(attn_extractor, AttentionExtractor), "Wrong data type for parameter 'attn_extractor'."
        assert isinstance(text_unit, str), "Wrong data type for parameter 'text_unit'."
        assert text_unit in AttentionGraphExtractor.available_text_units, f"No text unit {text_unit} found."

        self.attn_extractor = attn_extractor
        self.text_unit = text_unit

    @staticmethod
    def check_attn_graph_features(graph_attn_features: tuple, text_unit: str):
        assert isinstance(text_unit, str), "Wrong data type for parameter 'text_unit'."
        assert text_unit in AttentionGraphExtractor.available_text_units, f"No text unit {text_unit} found."

        if text_unit == 'token':
            AttentionExtractor.check_attn_features(graph_attn_features)
        elif text_unit == 'attr':
            AttributeAttentionExtractor.check_attn_features(graph_attn_features)
        else:
            raise NotImplementedError()

        err_msg = "Wrong graph attention features format."
        f = graph_attn_features[2]
        assert 'graph_attn' in f, err_msg
        assert isinstance(f['graph_attn'], dict), err_msg
        params = ['text_units', 'attns', 'flow']
        assert all([p in f['graph_attn'] for p in params]), err_msg
        assert isinstance(f['graph_attn']['text_units'], list), err_msg
        if f['graph_attn']['attns'] is not None:
            assert isinstance(f['graph_attn']['attns'], np.ndarray), err_msg
        if f['graph_attn']['flow'] is not None:
            assert isinstance(f['graph_attn']['flow'], np.ndarray), err_msg

    @staticmethod
    def check_batch_graph_attn_features(batch_graph_attn_features: list, text_unit: str):
        assert isinstance(batch_graph_attn_features, list), "Wrong data type for parameter 'batch_graph_attn_features'."
        assert len(batch_graph_attn_features) > 0, "Empty attention features."

        for graph_attn_features in batch_graph_attn_features:
            AttentionGraphExtractor.check_attn_graph_features(graph_attn_features, text_unit)

    def _get_attn_graph(self, left_entity: pd.Series, right_entity: pd.Series, features: dict):

        # check data types
        if self.text_unit == 'token':
            AttentionExtractor.check_attn_features((left_entity, right_entity, features))
        elif self.text_unit == 'attr':
            AttributeAttentionExtractor.check_attn_features((left_entity, right_entity, features))
        else:
            raise NotImplementedError()

        if self.text_unit == 'attr':
            text_units = [f'{prefix}{tu}' for prefix in ['l-', 'r-'] for tu in features["text_units"]]
            attentions_mat = features["attns"]

        elif self.text_unit == 'token':
            text_units = features['tokens']
            attentions_mat = features["attns"]
            attentions_mat = [att.detach().numpy() for att in attentions_mat]
            attentions_mat = np.asarray(attentions_mat)[:, 0]   # remove batch size

            # remove PAD tokens
            first_pad_token_idx = text_units.index('[PAD]')
            text_units = text_units[:first_pad_token_idx]
            attentions_mat = attentions_mat[:, :, :first_pad_token_idx, :first_pad_token_idx]

        else:
            raise NotImplementedError()

        # STEP 1: create the attention graph
        # get an attention map for each layer by averaging the heads' attention maps
        if attentions_mat is not None:
            att_mat = attentions_mat.sum(axis=1) / attentions_mat.shape[1]
            adj_mat, labels_to_index = AttentionGraphUtils.get_adjmat(mat=att_mat, input_text_units=text_units)
            G = AttentionGraphUtils.create_attention_graph(adj_mat)

            # STEP 2: compute the attention flow
            input_nodes = []
            for key in labels_to_index:
                if labels_to_index[key] < attentions_mat.shape[-1]:
                    input_nodes.append(key)

            flow_values = AttentionGraphUtils.compute_flows(G, labels_to_index, input_nodes, length=att_mat.shape[-1])
            # flow_G = AttentionGraphUtils.create_attention_graph(flow_values)

        else:
            att_mat = None
            flow_values = None

        new_features = features.copy()
        new_features['graph_attn'] = {
            'text_units': text_units,
            'attns': att_mat,
            'flow': flow_values,
        }

        return left_entity, right_entity, new_features

    def __len__(self):
        return len(self.attn_extractor)

    def __getitem__(self, idx):
        left_entity, right_entity, features = self.attn_extractor[idx]

        return self._get_attn_graph(left_entity, right_entity, features)

    def extract(self, out_file: str = None):
        attn_graph_features = []
        for i in tqdm(range(len(self))):
            features = self[i]
            attn_graph_features.append(features)

        if out_file:
            out_dir_path = out_file.split(os.sep)
            out_dir = os.sep.join(out_dir_path[:-1])
            pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
            with open(f'{out_file}.pkl', 'wb') as f:
                pickle.dump(attn_graph_features, f)

        return attn_graph_features


class AggregateAttributeAttentionGraph(object):

    def __init__(self, batch_graph_attns: list, target_categories: list = ['all']):
        AttentionGraphExtractor.check_batch_graph_attn_features(batch_graph_attns, 'attr')

        self.batch_graph_attns = [bga[2] for bga in batch_graph_attns]
        self.agg_metrics = ['mean']
        self.target_categories = target_categories

    def aggregate(self, metric: str):
        assert isinstance(metric, str), "Wrong data type for parameter 'metric'."
        assert metric in self.agg_metrics, f"Wrong metric: {metric} not in {self.agg_metrics}."

        attn_data = []
        flow_data = []
        text_units = None
        for graph_attns in self.batch_graph_attns:
            item = {
                'label': graph_attns['labels'].item(),
                'pred': graph_attns['preds'].item()
            }

            if text_units is None:
                text_units = graph_attns['graph_attn']['text_units']
            else:
                assert text_units == graph_attns['graph_attn']['text_units']

            attn_item = item.copy()
            attn_item['graph_attn'] = graph_attns['graph_attn']['attns']
            attn_data.append(attn_item)

            flow_item = item.copy()
            flow_item['graph_attn'] = graph_attns['graph_attn']['flow']
            flow_data.append(flow_item)

        attn_aggregator = BinaryClassificationResultsAggregator('graph_attn', target_categories=self.target_categories)
        grouped_attn, group_idxs, grouped_labels, grouped_preds = attn_aggregator.add_batch_data(attn_data)
        if metric == 'mean':
            attn_agg_data = attn_aggregator.aggregate(metric)
        else:
            raise NotImplementedError()

        flow_aggregator = BinaryClassificationResultsAggregator('graph_attn', target_categories=self.target_categories)
        grouped_flow, group_idxs, grouped_labels, grouped_preds = flow_aggregator.add_batch_data(flow_data)
        if metric == 'mean':
            flow_agg_data = flow_aggregator.aggregate(metric)
        else:
            raise NotImplementedError()

        out_grouped_data = {}
        out_agg_data = {}
        for cat in self.target_categories:

            if attn_agg_data[cat] is None:
                out_agg_data[cat] = None
                continue

            if metric == 'mean':
                attn_vals = attn_agg_data[cat]['mean']
                attn_errors = attn_agg_data[cat]['std']
                flow_vals = flow_agg_data[cat]['mean']
                flow_errors = flow_agg_data[cat]['std']
            else:
                raise NotImplementedError()

            out_grouped_data[cat] = {
                'text_units': text_units,
                'labels': grouped_labels[cat],
                'preds': grouped_preds[cat],
                'attn': grouped_attn[cat],
                'flow': grouped_flow[cat],
                'idxs': group_idxs[cat],
            }

            out_agg_data[cat] = {
                'text_units': text_units,
                'attn_vals': attn_vals,
                'attn_errors': attn_errors,
                'flow_vals': flow_vals,
                'flow_errors': flow_errors,
            }

        return out_grouped_data, out_agg_data
