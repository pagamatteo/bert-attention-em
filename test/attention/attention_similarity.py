from utils.general import get_dataset, get_model, get_sample
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification
from tqdm import tqdm
from utils.test_utils import ConfCreator
import seaborn as sns

"""
Code adapted from https://github.com/text-machine-lab/dark-secrets-of-BERT/blob/master/visualize_attention.ipynb
"""

PROJECT_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'attention')


def get_use_case_entity_pairs(conf, sampler_conf):
    dataset = get_dataset(conf)

    complete_sampler_conf = sampler_conf.copy()
    complete_sampler_conf['permute'] = conf['permute']
    sample = get_sample(dataset, complete_sampler_conf)

    return sample


def get_pt_ft_attention_sim(conf, sampler_conf, precomputed=False, save=False):

    uc_sim_maps = {}
    for uc in conf['use_case']:
        print("\n\n", uc)
        uc_conf = conf.copy()
        uc_conf['use_case'] = uc

        save_path = os.path.join(RESULTS_DIR, uc, 'attn_pt_ft_similarity.npy')
        compute = True
        if precomputed:
            if os.path.exists(save_path):
                print("Loading precomputed similarity map.")
                avg_uc_sim = np.load(save_path)
                compute = False
            else:
                print("No precomputed results are available.")
                compute = True

        if compute is True:

            print("Computing similarity map...")
            # Get data
            encoded_dataset = get_use_case_entity_pairs(conf=uc_conf, sampler_conf=sampler_conf)

            # Get pre-trained model
            pt_model = AutoModel.from_pretrained(conf['model_name'], output_attentions=True)

            # Get fine-tuned model
            ft_model_path = os.path.join(MODELS_DIR, 'simple', f"{uc}_{conf['tok']}_tuned")
            ft_model = AutoModelForSequenceClassification.from_pretrained(ft_model_path, output_attentions=True)

            n_layers, n_heads = pt_model.config.num_hidden_layers, pt_model.config.num_attention_heads

            uc_sims = []
            for encoded_row in tqdm(encoded_dataset):
                features = encoded_row[2]
                input_ids = features['input_ids'].unsqueeze(0)
                attention_mask = features['attention_mask'].unsqueeze(0)
                token_type_ids = features['token_type_ids'].unsqueeze(0)
                with torch.no_grad():
                    pt_attn = pt_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)["attentions"]
                    ft_attn = ft_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)["attentions"]

                    pt_attn = torch.cat(pt_attn).view(n_layers, n_heads, -1)
                    ft_attn = torch.cat(ft_attn).view(n_layers, n_heads, -1)

                    sim_map = torch.nn.functional.cosine_similarity(pt_attn, ft_attn, dim=-1).detach().numpy()
                    uc_sims.append(sim_map)

            uc_sims = np.stack(uc_sims, axis=-1)
            avg_uc_sim = np.mean(uc_sims, axis=-1)
            if save:
                np.save(save_path, avg_uc_sim)

        uc_sim_maps[uc] = avg_uc_sim

    sim_maps = {'all': np.mean(np.stack(list(uc_sim_maps.values()), axis=-1), axis=-1)}
    sim_maps.update(uc_sim_maps)

    return sim_maps


def plot_attention_sim(att_sim, ax=None, title=None, show_xlabel=True, show_ylabel=True, cbar=False, cbar_ax=None):

    show = False
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 9))
        show = True

    sns.heatmap(att_sim, cmap='Blues_r', vmin=0, vmax=1, ax=ax, cbar=cbar, cbar_ax=cbar_ax)
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=16)
    if show_xlabel:
        ax.set_xlabel('Head', fontsize=14)
    if show_ylabel:
        ax.set_ylabel('Layer', fontsize=14)
    ax.yaxis.set_tick_params(rotation=0, labelsize=14)
    ax.xaxis.set_tick_params(rotation=0, labelsize=14)
    ax.set_xticklabels(range(1, att_sim.shape[0] + 1))
    ax.set_yticklabels(range(1, att_sim.shape[1] + 1))

    if show:
        plt.show()


def plot_attention_sim_maps(sim_maps, save_path=None):

    use_case_map = ConfCreator().use_case_map

    ncols = 6
    nrows = 2
    figsize = (18, 6)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=True, sharex=True)
    axes = axes.flat
    # colorbar_ax = fig.add_axes([1.02, .377, .03, .27])
    colorbar_ax = fig.add_axes([1.01, .113, .025, .82])

    # loop over the use cases
    for idx, use_case in enumerate(sim_maps):
        ax = axes[idx]
        uc_sim_map = sim_maps[use_case]

        if idx % ncols == 0:
            show_ylabel = True
        else:
            show_ylabel = False

        if idx // ncols == nrows - 1:
            show_xlabel = True
        else:
            show_xlabel = False

        cbar = False
        cbar_ax = None
        # if idx // ncols == 1 and idx % ncols == ncols - 1:
        if idx == len(sim_maps) - 1:
            cbar = True
            cbar_ax = colorbar_ax

        plot_attention_sim(uc_sim_map, ax=ax, title=use_case_map[use_case], show_xlabel=show_xlabel,
                           show_ylabel=show_ylabel, cbar=cbar, cbar_ax=cbar_ax)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.15)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    use_cases = ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"]

    conf = {
        'use_case': use_cases,
        'data_type': 'train',  # 'train', 'test', 'valid'
        'model_name': 'bert-base-uncased',
        'tok': 'sent_pair',  # 'sent_pair', 'attr', 'attr_pair'
        'label_col': 'label',
        'left_prefix': 'left_',
        'right_prefix': 'right_',
        'max_len': 128,
        'permute': False,
        'verbose': False,
        'return_offset': False,
    }

    sampler_conf = {
        'size': None,
        'target_class': 'both',  # 'both', 0, 1
        'seeds': [42, 42],  # [42 -> class 0, 42 -> class 1]
    }

    # Compute similarity maps
    sim_maps = get_pt_ft_attention_sim(conf, sampler_conf, precomputed=True, save=True)

    # Plot the average similarity map over the entire benchmark
    all_sim_map = sim_maps['all']
    del sim_maps['all']
    plot_attention_sim(all_sim_map)

    # Plot the similarity maps for each use case
    plot_save_path = os.path.join(RESULTS_DIR, 'PLOT_attention_pt_ft_similarity.pdf')
    plot_attention_sim_maps(sim_maps, save_path=plot_save_path)


