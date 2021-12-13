import logging
import os
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoModel,
    set_seed,
)
from utils.general import get_dataset, get_sample
from pathlib import Path
from sklearn.metrics import f1_score
import pickle
import pandas as pd
from utils.test_utils import ConfCreator
import matplotlib.pyplot as plt
import matplotlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'attention')

"""
Code adapted from
https://github.com/huggingface/transformers/blob/master/examples/research_projects/bertology/run_bertology.py
"""


def get_results(conf: dict, use_cases: list):
    assert isinstance(conf, dict), "Wrong data type for parameter 'conf'."
    assert isinstance(use_cases, list), "Wrong data type for parameter 'use_cases'."
    assert len(use_cases) > 0, "Empty use case list."

    tester_params = {}
    tester_name = conf['tester']['tester']
    if tester_name == 'attr_tester':
        tester_param = {
            'permute': conf['permute'],
            'model_attention_grid': (12, 12),
        }
        tester_param.update(conf['tester']['tester_params'])
        tester_params[tester_name] = tester_param

    elif tester_name == 'attr_pattern_tester':
        tester_params[tester_name] = conf['tester']['tester_params']

    else:
        raise ValueError("Wrong tester name.")

    results = {}
    for use_case in use_cases:

        out_path = os.path.join(RESULTS_DIR, use_case)

        extractor_name = conf['extractor']['attn_extractor']
        extractor_params = '_'.join([f'{x[0]}={x[1]}' for x in conf['extractor']['attn_extr_params'].items()])
        if tester_name == 'attr_pattern_tester':
            tester_name = 'attr_patt_tester'
        tester_params = '_'.join([f'{x[0]}={x[1]}' for x in conf['tester']['tester_params'].items()])
        template_file_name = 'ANALYSIS_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_AVG.pickle'.format(use_case, conf['data_type'],
                                                                                        extractor_name, tester_name,
                                                                                        conf['fine_tune_method'],
                                                                                        conf['permute'], conf['tok'],
                                                                                        conf['size'], extractor_params,
                                                                                        tester_params)

        res_file = os.path.join(out_path, template_file_name)
        with open(res_file, 'rb') as f:
            res = pickle.load(f)
        results[use_case] = res

    return results


def entropy(p):
    """Compute the entropy of a probability distribution"""
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)


def print_2d_tensor(tensor):
    """Print a 2D tensor"""
    logger.info("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
    for row in range(len(tensor)):
        if tensor.dtype != torch.long:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data))
        else:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))


def compute_heads_importance(model, eval_dataloader, imp_norm_params, device, compute_entropy=True,
                             compute_importance=True, head_mask=None, actually_pruned=False):
    """This method shows how to compute:
    - head attention entropy
    - head importance scores according to http://arxiv.org/abs/1905.10650
    """

    assert 'dont_normalize_importance_by_layer' in imp_norm_params
    assert 'dont_normalize_global_importance' in imp_norm_params

    # Prepare our tensors
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(device)
    attn_entropy = torch.zeros(n_layers, n_heads).to(device)

    if head_mask is None:
        head_mask = torch.ones(n_layers, n_heads).to(device)

    head_mask.requires_grad_(requires_grad=True)
    # If actually pruned attention multi-head, set head mask to None to avoid shape mismatch
    if actually_pruned:
        head_mask = None

    preds = None
    labels = None
    tot_tokens = 0.0

    for step, inputs in enumerate(tqdm(eval_dataloader, desc="Iteration")):
        select_inputs = {}
        for k, v in inputs.items():
            if k not in ['sent1', 'sent2']:
                select_inputs[k] = v.to(device)

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        outputs = model(**select_inputs, head_mask=head_mask)
        loss, logits, all_attentions = (
            outputs[0],
            outputs[1],
            outputs[-1],
        )  # Loss and logits are the first, attention the last
        loss.backward()  # Back-propagate to populate the gradients in the head mask

        if compute_entropy:
            for layer, attn in enumerate(all_attentions):
                masked_entropy = entropy(attn.detach()) * inputs["attention_mask"].float().unsqueeze(1)
                attn_entropy[layer] += masked_entropy.sum(-1).sum(0).detach()

        if compute_importance:
            head_importance += head_mask.grad.abs().detach()

        # Also store our logits/labels if we want to compute metrics afterwards
        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, inputs["labels"].detach().cpu().numpy(), axis=0)

        tot_tokens += inputs["attention_mask"].float().detach().sum().data

    # Normalize
    attn_entropy /= tot_tokens
    head_importance /= tot_tokens
    # Layer-wise importance normalization
    if not imp_norm_params['dont_normalize_importance_by_layer']:
        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

    if not imp_norm_params['dont_normalize_global_importance']:
        head_importance = (head_importance - head_importance.min()) / (head_importance.max() - head_importance.min())

    # logger.info("Attention entropies")
    # print_2d_tensor(attn_entropy)
    # logger.info("Head importance scores")
    # print_2d_tensor(head_importance)
    # logger.info("Head ranked by importance scores")
    # head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long, device=device)
    # head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(
    #     head_importance.numel(), device=device
    # )
    # head_ranks = head_ranks.view_as(head_importance)
    # print_2d_tensor(head_ranks)

    return attn_entropy, head_importance, preds, labels


def get_data_and_model(conf, batch_size):
    dataset = get_dataset(conf)
    sample = get_sample(dataset, conf)

    class Sampler(object):
        def __init__(self, d):
            self.d = d

        def __len__(self):
            return len(self.d)

        def __getitem__(self, item):
            return self.d[item][2]

    dataloader = DataLoader(Sampler(sample), batch_size=batch_size)

    # Get model
    if conf['fine_tune_method'] is not None:
        model_path = os.path.join(MODELS_DIR, conf['fine_tune_method'], f"{conf['use_case']}_{conf['tok']}_tuned")
        model = AutoModelForSequenceClassification.from_pretrained(model_path, output_attentions=True)
    else:
        model = AutoModel.from_pretrained(conf['model_name'], output_attentions=True)

    return dataloader, model


def apply_pruning_or_masking(model, dataloader, prune, head_mask, device):
    if head_mask is not None:
        if prune:
            heads_to_prune = dict(
                (layer, (1 - head_mask[layer].long()).nonzero().squeeze(1).tolist()) for layer in range(len(head_mask))
            )

            assert sum(len(h) for h in heads_to_prune.values()) == (1 - head_mask.long()).sum().item()
            model.prune_heads(heads_to_prune)
            head_mask = None

    preds = None
    labels = None
    num_params = sum(p.numel() for p in model.parameters())

    start_time = datetime.now()
    for step, inputs in enumerate(tqdm(dataloader, desc="Iteration")):
        select_inputs = {}
        for k, v in inputs.items():
            if k not in ['sent1', 'sent2', 'labels']:
                select_inputs[k] = v.to(device)

        with torch.no_grad():
            outputs = model(**select_inputs, head_mask=head_mask)
            logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                labels = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                labels = np.append(labels, inputs["labels"].detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)
    score = f1_score(y_pred=preds, y_true=labels)

    elapsed_time = (datetime.now() - start_time).total_seconds()

    return {'num_params': num_params, 'score': score, 'time': elapsed_time}


def prune_or_mask_heads(model, dataloader, prune, heads_by_metric, amounts, device):
    """This method shows how to prune head (remove heads weights) based on
    the head importance scores as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """

    def get_topk_heads(head_grid, amount, grid_shape):
        sorted_head_idxs = head_grid.view(-1).sort(descending=True)[1]

        new_head_mask = torch.ones(grid_shape)

        count = 1
        while count <= amount:
            new_head_mask = new_head_mask.view(-1)
            new_head_mask[sorted_head_idxs[count - 1]] = 0.0
            new_head_mask = new_head_mask.view(grid_shape)

            # check if all the heads of a layer have been masked
            # in this case unmask the current head and select in the next step a new head to mask
            if (new_head_mask.sum(1) == 0).sum() > 0:   # there is a layer where all the heads have been masked
                new_head_mask = new_head_mask.view(-1)
                new_head_mask[sorted_head_idxs[count - 1]] = 1.0
                new_head_mask = new_head_mask.view(grid_shape)
                amount += 1

            count += 1

        return new_head_mask.clone().detach()

    results = []

    # Get original model effectiveness and efficiency
    original_scores = apply_pruning_or_masking(model=model, dataloader=dataloader, prune=False, head_mask=None,
                                               device=device)
    original_scores['num_to_mask'] = 0
    original_scores['perc_to_mask'] = 0
    original_scores['speed_up'] = None
    results.append(original_scores)
    logger.info(original_scores)

    for amount in amounts:
        logger.info(f"Heads to prune/mask: {amount}")

        new_head_mask = get_topk_heads(heads_by_metric, amount, heads_by_metric.shape)
        # sorted_head_idxs = heads_by_metric.view(-1).sort(descending=True)[1]
        # topk_head_idxs = sorted_head_idxs[:amount]
        #
        # new_head_mask = torch.ones_like(heads_by_metric)
        # new_head_mask = new_head_mask.view(-1)
        # new_head_mask[topk_head_idxs] = 0.0
        # new_head_mask = new_head_mask.view_as(heads_by_metric)
        # new_head_mask = new_head_mask.clone().detach()

        pruned_or_masked_scores = apply_pruning_or_masking(model=model, dataloader=dataloader, prune=prune,
                                                           head_mask=new_head_mask, device=device)
        pruned_or_masked_scores['num_to_mask'] = amount
        pruned_or_masked_scores['perc_to_mask'] = amount / heads_by_metric.numel()
        pruned_or_masked_scores['speed_up'] = original_scores['time'] / pruned_or_masked_scores['time'] * 100
        logger.info(pruned_or_masked_scores)

        results.append(pruned_or_masked_scores)

    return pd.DataFrame(results)


def get_maa_res(conf):
    train_conf = conf.copy()
    train_conf['data_type'] = 'train'
    res = get_results(train_conf, [train_conf['use_case']])
    heads_by_metric = res[train_conf['use_case']]['all'].get_results()['match_attr_attn_loc']
    heads_by_metric = torch.from_numpy(heads_by_metric)

    return heads_by_metric


def get_pruning_or_masking_effect(conf, prune_or_mask_conf, batch_size, device, save, seed=42):
    params = ['method', 'amounts', 'prune']
    assert all([p in prune_or_mask_conf for p in params])
    set_seed(seed)
    torch.manual_seed(seed)

    prune_or_mask_methods = prune_or_mask_conf['method']
    prune = prune_or_mask_conf['prune']
    amounts = prune_or_mask_conf['amounts']

    # Loop over use cases
    for idx in range(len(conf['use_case'])):

        uc = conf['use_case'][idx]
        uc_results = []

        for prune_or_mask_method in prune_or_mask_methods:

            # Get data and model
            uc_conf = conf.copy()
            uc_conf['use_case'] = uc
            dataloader, model = get_data_and_model(conf=uc_conf, batch_size=batch_size)

            # Get heads by metric
            if prune_or_mask_method == 'importance':

                # Get importance of heads
                precomputed_importance_path = os.path.join(RESULTS_DIR, uc, f'{conf["data_type"]}_head_importance.npy')

                if os.path.exists(precomputed_importance_path):
                    print("Load precomputed head importance...")
                    head_importance = torch.from_numpy(np.load(precomputed_importance_path))

                else:
                    print("Compute head importance...")
                    imp_params = ['dont_normalize_importance_by_layer', 'dont_normalize_global_importance']
                    imp_norm_params = {k: v for k, v in prune_or_mask_conf.items() if k in imp_params}
                    _, head_importance, _, _ = compute_heads_importance(model, dataloader, imp_norm_params, device,
                                                                        compute_entropy=False)
                    with open(precomputed_importance_path, 'wb') as f:
                        np.save(f, head_importance.numpy())

                heads_by_metric = head_importance

            elif prune_or_mask_method == 'maa':

                assert 'precomputed' in prune_or_mask_conf

                # Load precomputed maa results
                if prune_or_mask_conf['precomputed']:
                    heads_by_metric = get_maa_res(uc_conf)
                    heads_by_metric[heads_by_metric == 0.0] = -1

                else:
                    raise NotImplementedError()

            elif prune_or_mask_method == 'random':
                n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
                heads_by_metric = torch.rand(n_layers, n_heads)

            else:
                raise ValueError("Wrong prune or mask method.")

            results = prune_or_mask_heads(model=model, dataloader=dataloader, prune=prune,
                                          heads_by_metric=heads_by_metric,
                                          amounts=amounts, device=device)
            results['method'] = prune_or_mask_method
            results['prune'] = prune
            uc_results.append(results)

        uc_tab_results = pd.concat(uc_results)

        if save:
            save_path = os.path.join(RESULTS_DIR, uc, f'head_prune_mask.csv')
            uc_tab_results.to_csv(save_path)


def get_masking_results(use_cases):

    all_res = []
    use_case_map = ConfCreator().use_case_map
    for uc in use_cases:
        uc_res_path = os.path.join(RESULTS_DIR, uc, f'head_prune_mask.csv')
        uc_res = pd.read_csv(uc_res_path)
        uc_res['use_case'] = use_case_map[uc]
        all_res.append(uc_res)

    res = pd.concat(all_res)

    return res


def plot_masking_results(results, save_path=None):

    plot_table = results.pivot_table(values='score', index=['use_case', 'num_to_mask'], columns=['method'])

    # ncols = 6
    # nrows = 2
    # figsize = (21, 6)
    ncols = 4
    nrows = 3
    figsize = (16, 8)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=True)
    axes = axes.flat

    # loop over the use cases
    for idx, use_case in enumerate(ConfCreator().use_case_map.values()):
        ax = axes[idx]
        use_case_plot_data = plot_table[plot_table.index.get_level_values(0) == use_case]
        use_case_plot_data = use_case_plot_data.reset_index(level=0, drop=True)
        # no_mask_score = use_case_plot_data.iloc[0]
        use_case_plot_data = use_case_plot_data.iloc[1:-1]
        use_case_plot_data.index = range(4)

        use_case_plot_data.plot(ax=ax, marker='o', legend=False, rot=0)
        # ax.axhline(y=no_mask_score[0], color='r', linestyle='-')

        ax.set_title(use_case, fontsize=16)

        if idx % ncols == 0:
            ax.set_ylabel('F1 score', fontsize=16)
        ax.set_xlabel('# pruned heads', fontsize=16)
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)
        positions = range(4)
        labels = [5, 10, 20, 50]
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(positions))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(labels))

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(.68, 0.01), ncol=3, fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.5)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    use_cases = ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"]

    conf = {
        'use_case': ["Structured_DBLP-ACM", "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"],
        'data_type': 'test',
        'model_name': 'bert-base-uncased',
        'tok': 'sent_pair',  # 'sent_pair', 'attr', 'attr_pair'
        'label_col': 'label',
        'left_prefix': 'left_',
        'right_prefix': 'right_',
        'max_len': 128,
        'permute': False,
        'verbose': False,
        'return_offset': False,
        'size': None,
        'target_class': 'both',  # 'both', 0, 1
        'seeds': [42, 42],  # [42 -> class 0, 42 -> class 1]
        'fine_tune_method': 'simple',  # None, 'simple'
        'extractor': {
            'attn_extractor': 'attr_extractor',  # 'attr_extractor', 'word_extractor'
            'attn_extr_params': {'special_tokens': True, 'agg_metric': 'max'},
        },
        'tester': {
            'tester': 'attr_tester',  # 'attr_tester', 'attr_pattern_tester'
            'tester_params': {'ignore_special': True}
        },
    }

    # prune or mask params
    prune_or_mask_params = {
        'importance': {
            'dont_normalize_importance_by_layer': True,  # don't normalize importance score by layers
            'dont_normalize_global_importance': False,  # don't normalize all importance scores between 0 and 1
        },
        'maa': {
            'precomputed': True
        },
        'random': {}
    }

    prune_or_mask_method = ['importance', 'maa', 'random']  # 'importance', 'maa', 'random'
    prune_or_mask_conf = {
        'method': prune_or_mask_method,
        'amounts': [5, 10, 20, 50, 100],  # amounts of heads to prune/mask
        'prune': True,  # prune -> remove heads from the model, mask -> set heads to zero from the model
    }
    for m in prune_or_mask_method:
        prune_or_mask_conf.update(prune_or_mask_params[m])

    # batch size
    batch_size = 1

    # device
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    # save
    save = True

    # RUN EXPERIMENT
    # get_pruning_or_masking_effect(conf=conf, prune_or_mask_conf=prune_or_mask_conf, batch_size=batch_size,
    #                               device=device, save=save)

    # PLOT RESULTS
    res = get_masking_results(use_cases=use_cases)
    save_path = os.path.join(RESULTS_DIR, 'PLOT_masking.pdf')
    plot_masking_results(res, save_path=save_path)

    print(":)")
