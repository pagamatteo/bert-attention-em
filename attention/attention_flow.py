from attention.extractors import AttentionExtractor
from utils.general import get_dataset, get_model, get_sample
import os
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from transformers import BertConfig, BertForMaskedLM, BertTokenizer
import torch
import itertools
import seaborn as sns

PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')


def get_adjmat(mat, input_tokens):
    n_layers, length, _ = mat.shape
    adj_mat = np.zeros(((n_layers + 1) * length, (n_layers + 1) * length))
    labels_to_index = {}
    for k in np.arange(length):
        labels_to_index[str(k) + "_" + input_tokens[k]] = k

    for i in np.arange(1, n_layers + 1):
        for k_f in np.arange(length):
            index_from = (i) * length + k_f
            label = "L" + str(i) + "_" + str(k_f)
            labels_to_index[label] = index_from
            for k_t in np.arange(length):
                index_to = (i - 1) * length + k_t
                adj_mat[index_from][index_to] = mat[i - 1][k_f][k_t]

    return adj_mat, labels_to_index


def draw_attention_graph(adjmat, labels_to_index, n_layers, length):
    A = adjmat
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
    for i in np.arange(A.shape[0]):
        for j in np.arange(A.shape[1]):
            nx.set_edge_attributes(G, {(i, j): A[i, j]}, 'capacity')

    pos = {}
    label_pos = {}
    for i in np.arange(n_layers + 1):
        for k_f in np.arange(length):
            pos[i * length + k_f] = ((i + 0.4) * 2, length - k_f)
            label_pos[i * length + k_f] = (i * 2, length - k_f)

    index_to_labels = {}
    for key in labels_to_index:
        index_to_labels[labels_to_index[key]] = key.split("_")[-1]
        if labels_to_index[key] >= length:
            index_to_labels[labels_to_index[key]] = ''

    # # plt.figure(1,figsize=(20,12))
    #
    # nx.draw_networkx_nodes(G, pos, node_color='green', node_size=50)#, labels=index_to_labels, node_size=50)
    # nx.draw_networkx_labels(G, pos=label_pos, labels=index_to_labels, font_size=18)
    #
    # all_weights = []
    # # 4 a. Iterate through the graph nodes to gather all the weights
    # for (node1, node2, data) in G.edges(data=True):
    #     all_weights.append(data['weight'])  # we'll use this when determining edge thickness
    #
    # # 4 b. Get unique weights
    # unique_weights = list(set(all_weights))
    #
    # # 4 c. Plot the edges - one by one!
    # for weight in unique_weights:
    #     # 4 d. Form a filtered list with just the weight you want to draw
    #     weighted_edges = [(node1, node2) for (node1, node2, edge_attr) in G.edges(data=True) if
    #                       edge_attr['weight'] == weight]
    #     # 4 e. I think multiplying by [num_nodes/sum(all_weights)] makes the graphs edges look cleaner
    #
    #     w = weight  # (weight - min(all_weights))/(max(all_weights) - min(all_weights))
    #     width = w
    #     nx.draw_networkx_edges(G, pos, edgelist=weighted_edges, width=width, edge_color='darkblue')

    return G


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


def compute_joint_attention(att_mat, add_residual=True):
    if add_residual:
        residual_att = np.eye(att_mat.shape[1])[None, ...]
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1)[..., None]
    else:
        aug_att_mat = att_mat

    joint_attentions = np.zeros(aug_att_mat.shape)

    layers = joint_attentions.shape[0]
    joint_attentions[0] = aug_att_mat[0]
    for i in np.arange(1, layers):
        joint_attentions[i] = aug_att_mat[i].dot(joint_attentions[i - 1])

    return joint_attentions


def plot_attention_heatmap(att, s_position, t_positions, sentence):
    cls_att = np.flip(att[:, s_position, t_positions], axis=0)
    xticklb = input_tokens = list(
        itertools.compress(['<cls>'] + sentence.split(), [i in t_positions for i in np.arange(len(sentence) + 1)]))
    yticklb = [str(i) if i % 2 == 0 else '' for i in np.arange(att.shape[0], 0, -1)]
    ax = sns.heatmap(cls_att, xticklabels=xticklb, yticklabels=yticklb, cmap="YlOrRd")
    return ax


def convert_adjmat_tomats(adjmat, n_layers, l):
    mats = np.zeros((n_layers, l, l))

    for i in np.arange(n_layers):
        mats[i] = adjmat[(i + 1) * l:(i + 2) * l, i * l:(i + 1) * l]

    return mats


def get_gradient(sent1, sent2, model, tokenizer):
    """Return gradient of input (question) wrt to model output span prediction

    Args:
        question (str): text of input question
        context (str): text of question context/passage
        model (QA model): Hugging Face BERT model for QA transformers.modeling_tf_distilbert.TFDistilBertForQuestionAnswering, transformers.modeling_tf_bert.TFBertForQuestionAnswering
        tokenizer (tokenizer): transformers.tokenization_bert.BertTokenizerFast

    Returns:
          (tuple): (gradients, token_words, token_types, answer_text)
    """

    encoded_tokens = tokenizer.encode_plus(sent1, sent2, add_special_tokens=True, return_token_type_ids=True,
                                           return_tensors="pt")
    inputs_embeds = model.get_input_embeddings()(encoded_tokens['input_ids'])
    print(model(inputs_embeds=inputs_embeds, token_type_ids=encoded_tokens["token_type_ids"],
                attention_mask=encoded_tokens["attention_mask"]).keys())
    outputs = model(inputs_embeds=inputs_embeds,
                    token_type_ids=encoded_tokens["token_type_ids"],
                    attention_mask=encoded_tokens["attention_mask"])
    logits = outputs["logits"]

    y = torch.argmax(logits, axis=-1)
    torch.autograd.grad(y, inputs_embeds, retain_graph=True, create_graph=True)

    assert False


def explain_model(question, context, model, tokenizer, explain_method="gradient"):
    if explain_method == "gradient":
        return get_gradient(question, context, model, tokenizer)


if __name__ == '__main__':

    from transformers import AutoModel, AutoTokenizer

    conf = {
        'use_case': "Structured_Fodors-Zagats",
        'data_type': 'train',  # 'train', 'test', 'valid'
        'model_name': 'bert-base-uncased',
        'tok': 'sent_pair',  # 'sent_pair', 'attr', 'attr_pair'
        'label_col': 'label',
        'left_prefix': 'left_',
        'right_prefix': 'right_',
        'max_len': 128,
        'permute': False,
        'verbose': False,
    }

    sampler_conf = {
        'size': 2,
        'target_class': 'both',  # 'both', 0, 1
        'seeds': [42, 42],  # [42 -> class 0, 42 -> class 1]
    }

    fine_tune = 'simple'  # None, 'simple', 'advanced'

    dataset = get_dataset(conf)
    use_case = conf['use_case']
    tok = conf['tok']
    model_name = conf['model_name']

    if fine_tune is not None:
        model_path = os.path.join(MODELS_DIR, fine_tune, f"{use_case}_{tok}_tuned")
    else:
        model_path = None
    model = get_model(model_name, fine_tune, model_path)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)

    sent1 = "'` rex il ristorante \' \' 617 s. olive st. \' ` los angeles \' 213/627 -2300 italian 19'"
    sent2 = "'` rex il ristorante \' \' 617 s. olive st. \' ` los angeles \' 213-627-2300 ` nuova cucina italian \' 19'"

    gradients, tokens, token_types, answer = explain_model(sent1, sent2, model, tokenizer)

    # tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    # model = BertForMaskedLM.from_pretrained('bert-large-uncased',
    #                                         output_hidden_states=True,
    #                                         output_attentions=True)
    #
    # sent = "'` rex il ristorante \' \' 617 s. olive st. \' ` los angeles \' 213/627 -2300 " + tokenizer.mask_token + " 19' [SEP] '` rex il ristorante \' \' 617 s. olive st. \' ` los angeles \' 213-627-2300 ` nuova cucina italian \' 19'"
    # tokens = ['[cls]'] + tokenizer.tokenize(sent) + ['[sep]']
    # tf_input_ids = tokenizer.encode(sent)
    # input_ids = torch.tensor([tf_input_ids])
    # output = model(input_ids)[0]
    # predicted_target = torch.nn.Softmax()(output[0, 29])
    # words = tokenizer.decode([np.argmax(predicted_target.detach().numpy(), axis=-1)])
    # print("CIAO")

    # print(tokenizer.encode(['italian'])[1])
    #
    # sentences = {}
    # src = {}
    # targets = {}
    # sentences[2] = "'` rex il ristorante \' \' 617 s. olive st. \' ` los angeles \' 213/627 -2300 " + tokenizer.mask_token + " 19' [SEP] '` rex il ristorante \' \' 617 s. olive st. \' ` los angeles \' 213-627-2300 ` nuova cucina italian \' 19'"
    # src[2] = 7
    # targets[2] = (2, 5)
    #
    # ex_id = 2
    # sentence = sentences[ex_id]
    # tokens = ['[cls]'] + tokenizer.tokenize(sentence) + ['[sep]']
    # print(len(tokens), tokens)
    # tf_input_ids = tokenizer.encode(sentence)
    # print(tokenizer.decode(tf_input_ids))
    # input_ids = torch.tensor([tf_input_ids])
    # all_hidden_states, all_attentions = model(input_ids)[-2:]
    #
    # _attentions = [att.detach().numpy() for att in all_attentions]
    # attentions_mat = np.asarray(_attentions)[:, 0]
    # print(attentions_mat.shape)
    #
    # output = model(input_ids)[0]
    # predicted_target = torch.nn.Softmax()(output[0, src[ex_id]])
    #
    # print(np.argmax(output.detach().numpy()[0], axis=-1))
    # print(tokenizer.decode(np.argmax(output.detach().numpy()[0], axis=-1)))
    # print(tf_input_ids[src[ex_id]], tokenizer.decode([tf_input_ids[src[ex_id]]]))
    # print(tf_input_ids[targets[ex_id][0]], tokenizer.decode([tf_input_ids[targets[ex_id][0]]]),
    #       predicted_target[tf_input_ids[targets[ex_id][0]]])
    # print(tf_input_ids[targets[ex_id][1]], tokenizer.decode([tf_input_ids[targets[ex_id][1]]]),
    #       predicted_target[tf_input_ids[targets[ex_id][1]]])
    #
    # his_id = tokenizer.encode(['his'])[1]
    # her_id = tokenizer.encode(['her'])[1]
    #
    # print(his_id, her_id)
    # print("his prob:", predicted_target[his_id], "her prob:", predicted_target[her_id], "her?",
    #       predicted_target[her_id] > predicted_target[his_id])
    #
    # # plot_attention_heatmap(attentions_mat.sum(axis=1) / attentions_mat.shape[1], src[ex_id], t_positions=targets[ex_id],
    # #                        sentence=sentence)
    #
    # res_att_mat = attentions_mat.sum(axis=1) / attentions_mat.shape[1]
    # res_att_mat = res_att_mat + np.eye(res_att_mat.shape[1])[None, ...]
    # res_att_mat = res_att_mat / res_att_mat.sum(axis=-1)[..., None]
    #
    # res_adj_mat, res_labels_to_index = get_adjmat(mat=res_att_mat, input_tokens=tokens)
    #
    # res_G = draw_attention_graph(res_adj_mat, res_labels_to_index, n_layers=res_att_mat.shape[0],
    #                              length=res_att_mat.shape[-1])
    #
    # output_nodes = []
    # input_nodes = []
    # for key in res_labels_to_index:
    #     if 'L24' in key:
    #         output_nodes.append(key)
    #     if res_labels_to_index[key] < attentions_mat.shape[-1]:
    #         input_nodes.append(key)
    #
    # flow_values = compute_flows(res_G, res_labels_to_index, input_nodes, length=attentions_mat.shape[-1])
    # exit(1)
    #
    # flow_att_mat = convert_adjmat_tomats(flow_values, n_layers=attentions_mat.shape[0], l=attentions_mat.shape[-1])
    #
    # plot_attention_heatmap(flow_att_mat, src[ex_id], t_positions=targets[ex_id], sentence=sentence)
    # exit(1)

    conf = {
        'use_case': "Structured_Fodors-Zagats",
        'data_type': 'train',  # 'train', 'test', 'valid'
        'model_name': 'bert-base-uncased',
        'tok': 'sent_pair',  # 'sent_pair', 'attr', 'attr_pair'
        'label_col': 'label',
        'left_prefix': 'left_',
        'right_prefix': 'right_',
        'max_len': 128,
        'permute': False,
        'verbose': False,
    }

    sampler_conf = {
        'size': 2,
        'target_class': 'both',  # 'both', 0, 1
        'seeds': [42, 42],  # [42 -> class 0, 42 -> class 1]
    }

    fine_tune = 'simple'  # None, 'simple', 'advanced'

    dataset = get_dataset(conf)
    use_case = conf['use_case']
    tok = conf['tok']
    model_name = conf['model_name']

    if fine_tune is not None:
        model_path = os.path.join(MODELS_DIR, fine_tune, f"{use_case}_{tok}_tuned")
    else:
        model_path = None
    model = get_model(model_name, fine_tune, model_path)

    complete_sampler_conf = sampler_conf.copy()
    complete_sampler_conf['permute'] = conf['permute']
    sample = get_sample(dataset, complete_sampler_conf)

    attn_extr = AttentionExtractor(sample, model)
    for left_entity, right_entity, param in attn_extr:
        print(left_entity)
        print(right_entity)
        print(param.keys())
        attns = param["attns"]
        tokens = param["tokens"]
        _attentions = [att.detach().numpy() for att in attns]
        attentions_mat = np.asarray(_attentions)[:, 0]
        # average attention scores associated to the same layer
        att_mat = attentions_mat.sum(axis=1) / attentions_mat.shape[1]
        adj_mat, labels_to_index = get_adjmat(mat=att_mat, input_tokens=tokens)

        G = draw_attention_graph(adj_mat, labels_to_index, n_layers=attentions_mat.shape[0],
                                 length=attentions_mat.shape[-1])

        res_att_mat = attentions_mat.sum(axis=1) / attentions_mat.shape[1]
        res_att_mat = res_att_mat + np.eye(res_att_mat.shape[1])[None, ...]
        res_att_mat = res_att_mat / res_att_mat.sum(axis=-1)[..., None]

        res_adj_mat, res_labels_to_index = get_adjmat(mat=res_att_mat, input_tokens=tokens)

        res_G = draw_attention_graph(res_adj_mat, res_labels_to_index, n_layers=res_att_mat.shape[0],
                                     length=res_att_mat.shape[-1])

        input_nodes = []
        for key in labels_to_index:
            if labels_to_index[key] < attentions_mat.shape[-1]:
                input_nodes.append(key)
        flow_values = compute_flows(res_G, res_labels_to_index, input_nodes, length=attentions_mat.shape[-1])
        flow_G = draw_attention_graph(flow_values, labels_to_index, n_layers=attentions_mat.shape[0],
                                      length=attentions_mat.shape[-1])

        joint_attentions = compute_joint_attention(res_att_mat, add_residual=False)
        joint_att_adjmat, joint_labels_to_index = get_adjmat(mat=joint_attentions, input_tokens=tokens)

        G = draw_attention_graph(joint_att_adjmat, joint_labels_to_index, n_layers=joint_attentions.shape[0],
                                 length=joint_attentions.shape[-1])
        break
