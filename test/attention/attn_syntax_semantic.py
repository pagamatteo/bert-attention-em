from utils.general import get_dataset, get_sample
from utils.nlp import get_similar_word_pairs
import os
import pandas as pd
from pathlib import Path
from tqdm import trange, tqdm
from utils.test_utils import ConfCreator
import matplotlib.pyplot as plt
import pickle
import gensim
import numpy as np
from core.attention.extractors import AttributeAttentionExtractor, WordAttentionExtractor, AttentionExtractor

PROJECT_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'attention')
FAST_TEXT_PATH = os.path.join(PROJECT_DIR, 'data', 'wiki-news-300d-1M.vec', 'wiki-news-300d-1M.vec')


def get_use_case_entity_pairs(conf, sampler_conf):
    dataset = get_dataset(conf)

    complete_sampler_conf = sampler_conf.copy()
    complete_sampler_conf['permute'] = conf['permute']
    sample = get_sample(dataset, complete_sampler_conf)

    return sample


def load_saved_attn_data(use_case, conf, sampler_conf, fine_tune, attn_params):
    tok = conf['tok']
    size = sampler_conf['size']
    extractor_name = attn_params['attn_extractor']
    params = '_'.join([f'{x[0]}={x[1]}' for x in attn_params['attn_extr_params'].items()])
    out_fname = f"ATTN_{use_case}_{tok}_{size}_{fine_tune}_{extractor_name}_{params}"
    data_path = os.path.join(RESULTS_DIR, use_case, out_fname)
    uc_attn = pickle.load(open(f"{data_path}.pkl", "rb"))
    if extractor_name == 'attr_extractor':
        AttributeAttentionExtractor.check_batch_attn_features(uc_attn)
    elif extractor_name == 'word_extractor':
        WordAttentionExtractor.check_batch_attn_features(uc_attn)
    elif extractor_name == 'token_extractor':
        AttentionExtractor.check_batch_attn_features(uc_attn)
    else:
        raise NotImplementedError()
    return uc_attn


def get_top_word_pairs_by_attn(attns, tokenization='sent_pair'):
    attn_data = {}
    for record_id, attn in enumerate(attns):
        attn_features = attn[2]
        attn_values = attn_features['attns']
        attn_text_units = attn_features['text_units']
        label = int(attn_features['labels'].item())
        pred = int(attn_features['preds'].item()) if attn_features['preds'] is not None else None

        # find the [SEP] token used to delimit the two entities
        sep_idxs = np.where(np.array(attn_text_units) == '[SEP]')[0]

        # filter out truncated rows
        if len(sep_idxs) % 2 != 0 or attn_values is None:
            print("Skip truncated row.")
            continue

        if tokenization == 'sent_pair':
            entity_delimit = attn_text_units.index('[SEP]')  # get first occurrence of the [SEP] token

        else:  # attr-pair
            # in the attr-pair tokenization the [SEP] token is also used to delimit the attributes
            entity_delimit = sep_idxs[(len(sep_idxs) // 2) - 1]

        # get an average attention map for each layer by averaging all the heads referring to the same layer
        attn_values = attn_values.mean(axis=1)

        # ignore the special tokens and related attention weights
        left_idxs = list(range(entity_delimit + 1))
        right_idxs = list(range(entity_delimit, attn_values.shape[1]))
        left_idxs = left_idxs[1:]  # remove [CLS]
        left_idxs = sorted(list(set(left_idxs).difference(set(sep_idxs))))  # remove [SEP]s
        right_idxs = sorted(list(set(right_idxs).difference(set(sep_idxs))))  # remove [SEP]s
        valid_idxs = np.array(left_idxs + right_idxs)
        attn_values = attn_values[:, valid_idxs, :][:, :, valid_idxs]
        valid_attn_text_units = list(np.array(attn_text_units)[valid_idxs])
        left_words = list(np.array(attn_text_units)[left_idxs])
        right_words = list(np.array(attn_text_units)[right_idxs])

        assert attn_values.shape[1] == len(valid_attn_text_units)
        original_left = [w for attr in attn[0] for w in str(attr).split()]
        original_right = [w for attr in attn[1] for w in str(attr).split()]
        assert all([left_words[i] == original_left[i] for i in range(len(left_words))])
        assert all([right_words[i] == original_right[i] for i in range(len(right_words))])

        for layer in range(attn_values.shape[0]):
            layer_attn_values = attn_values[layer]

            attn_scores = np.maximum(layer_attn_values[:len(left_idxs), len(left_idxs):],
                                     layer_attn_values[len(left_idxs):, :len(left_idxs)].T)

            thr = np.quantile(attn_scores, 0.8)
            row_idxs, col_idxs = np.where(attn_scores >= thr)
            words = [(attn_scores[i, j], (left_words[i], right_words[j])) for (i, j) in zip(row_idxs, col_idxs)]

            top_words = list(sorted(words, key=lambda x: x[0], reverse=True))
            if len(top_words) > int(attn_scores.size * 0.2):
                new_topk = int(attn_scores.size * 0.2)
                top_words = top_words[:new_topk]
            else:
                new_topk = len(top_words)

            if layer not in attn_data:
                attn_data[layer] = {record_id: top_words}
            else:
                attn_data[layer][record_id] = top_words

    return attn_data


def get_attn_to_similarity(conf, sampler_conf, fine_tune, attn_params, sim_type, sim_metric, sim_thrs, sim_op_eq,
                           sem_emb_model=None):
    attn_to_sim = []
    for uc in conf['use_case']:
        print("\n\n", uc)
        uc_conf = conf.copy()
        uc_conf['use_case'] = uc

        # Get data
        encoded_dataset = get_use_case_entity_pairs(conf=uc_conf, sampler_conf=sampler_conf)

        # Get similar words
        pair_of_entities = [(row[0], row[1]) for row in encoded_dataset]
        top_word_pairs_by_similarity = get_similar_word_pairs(pair_of_entities=pair_of_entities, sim_type=sim_type,
                                                              metric=sim_metric, thrs=sim_thrs, op_eq=sim_op_eq,
                                                              sem_emb_model=sem_emb_model, word_min_len=None)

        # Get top word pairs by BERT attention weights
        uc_attn = load_saved_attn_data(uc, conf, sampler_conf, fine_tune, attn_params)
        top_word_pairs_by_attn = get_top_word_pairs_by_attn(uc_attn)

        # Compute the percentage of word pairs shared between top word pairs by similarity and attention
        # loop over similarity thresholds
        skips = 0
        for thr in top_word_pairs_by_similarity:
            print(f"THR: {thr}")
            thr_sim_word_pairs = top_word_pairs_by_similarity[thr]
            sim_record_ids = thr_sim_word_pairs['idxs']
            sim_word_pairs = [[(p[0], p[1]) for p in p_list] for p_list in thr_sim_word_pairs['pairs']]

            # loop over BERT layers
            for layer in top_word_pairs_by_attn:
                print(f"\tLAYER: {layer}")
                layer_attn_word_pairs = top_word_pairs_by_attn[layer]

                overlaps = []
                weights = []
                for k in trange(len(sim_record_ids)):
                    record_id = sim_record_ids[k]
                    if record_id in layer_attn_word_pairs:
                        attn_word_pairs = [p[1] for p in layer_attn_word_pairs[record_id]]
                        weight = min(len(sim_word_pairs[k]), len(attn_word_pairs))
                        overlap = len(set(attn_word_pairs).intersection(set(sim_word_pairs[k]))) / weight
                        overlaps.append(overlap)
                        weights.append(weight)
                    else:
                        skips += 1

                layer_overlap = np.average(overlaps, weights=weights)

                attn_to_sim.append({
                    'use_case': uc,
                    'layer': layer,
                    'overlap': layer_overlap,
                    'num_records': len(overlaps)
                })

        print(f"SKIPS: {skips}")

    return attn_to_sim


def _res_to_df(res, res_type):
    res_tab = pd.DataFrame(res)
    res_tab['use_case'] = res_tab['use_case'].map(use_case_map)
    if res_type is None:
        method = 'pre-trained'
    else:
        method = 'fine-tuned'
    res_tab['method'] = method

    return res_tab


def load_results(res_type, tok, res_metric):

    # get pt results
    model_type = None
    with open(os.path.join(RESULTS_DIR, f'attn_to_{res_type}_{tok}_{model_type}_{res_metric}.pkl'), 'rb') as f:
        pt_res = pickle.load(f)
    pt_res = _res_to_df(pt_res, model_type)

    # get ft results
    model_type = 'simple'
    with open(os.path.join(RESULTS_DIR, f'attn_to_{res_type}_{tok}_{model_type}_{res_metric}.pkl'), 'rb') as f:
        ft_res = pickle.load(f)
    ft_res = _res_to_df(ft_res, model_type)

    res = pd.concat([pt_res, ft_res])

    return res


def plot_results(plot_data, plot_type, save_path=None):
    plt.figure(figsize=(6, 2.5))

    for method in ['pre-trained', 'fine-tuned']:
        method_data = plot_data[plot_data['method'] == method]
        if plot_type == 'by_use_case':
            method_pivot_data = method_data.pivot_table(index='layer', columns=['use_case'], values='overlap')
            method_pivot_data = method_pivot_data[use_case_map.values()]
        elif plot_type == 'by_layer':
            method_pivot_data = method_data.pivot_table(index='use_case', columns=['layer'], values='overlap')
            method_pivot_data.columns = range(1, len(method_pivot_data.columns) + 1)
        else:
            raise NotImplementedError()
        method_pivot_data = method_pivot_data * 100
        method_plot_stats = method_pivot_data.describe()
        medians = method_plot_stats.loc['50%', :].values
        percs_25 = method_plot_stats.loc['25%', :].values
        percs_75 = method_plot_stats.loc['75%', :].values

        plot_stats = {
            'x': range(len(method_pivot_data.columns)),
            'y': medians,
            'yerr': [medians - percs_25, percs_75 - medians],
        }

        if plot_type == 'by_use_case':
            plt.errorbar(**plot_stats, alpha=.75, fmt='.', capsize=5, label=method)
        elif plot_type == 'by_layer':
            plt.errorbar(**plot_stats, alpha=.75, fmt='.-', capsize=5, label=method)
        else:
            raise NotImplementedError()

    ax = plt.gca()
    ax.set_ylim(0, 100)
    plt.xticks(rotation=0)
    plt.legend(ncol=2, loc='best')
    plt.ylabel("Freq. (%)")
    if plot_type == 'by_use_case':
        plt.xlabel('Datasets')
        plt.xticks(range(len(method_pivot_data.columns)), method_pivot_data.columns)
    elif plot_type == 'by_layer':
        plt.xlabel('Layers')
        plt.xticks(range(len(method_pivot_data.columns)), list(method_pivot_data.columns))
    else:
        raise NotImplementedError()
    plt.tight_layout()

    if save_path is not None:
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

    fine_tune = 'simple'  # None, 'simple', 'advanced'

    attn_params = {
        'attn_extractor': 'word_extractor',  # 'attr_extractor', 'word_extractor', 'token_extractor'
        'attn_extr_params': {'special_tokens': True, 'agg_metric': 'mean'},
    }

    use_case_map = ConfCreator().use_case_map

    sim_type = 'syntax'
    compute = False

    if sim_type == 'syntax':
        sim_metric = 'jaccard'

        if compute is True:
            attn_to_syntax = get_attn_to_similarity(conf, sampler_conf, fine_tune, attn_params, sim_type=sim_type,
                                                    sim_metric=sim_metric, sim_thrs=[0.7], sim_op_eq=False)
            out_fname = os.path.join(RESULTS_DIR, f'attn_to_{sim_type}_{conf["tok"]}_{fine_tune}_{sim_metric}.pkl')

            with open(out_fname, 'wb') as f:
                pickle.dump(attn_to_syntax, f)

        else:
            res = load_results(res_type=sim_type, tok=conf['tok'], res_metric=sim_metric)
            plot_results(res, plot_type='by_use_case', save_path=os.path.join(RESULTS_DIR, f'PLOT_attn_to_{sim_type}_by_use_case.pdf'))
            plot_results(res, plot_type='by_layer', save_path=os.path.join(RESULTS_DIR, f'PLOT_attn_to_{sim_type}_by_layer.pdf'))

    elif sim_type == 'semantic':
        sim_metric = 'cosine'

        if compute is True:
            fasttext_model = gensim.models.KeyedVectors.load_word2vec_format(FAST_TEXT_PATH, binary=False,
                                                                             encoding='utf8')
            attn_to_semantic = get_attn_to_similarity(conf, sampler_conf, fine_tune, attn_params, sim_type=sim_type,
                                                      sim_metric=sim_metric, sim_thrs=[0.7], sim_op_eq=False,
                                                      sem_emb_model=fasttext_model)
            out_fname = os.path.join(RESULTS_DIR, f'attn_to_{sim_type}_{conf["tok"]}_{fine_tune}_{sim_metric}.pkl')

            with open(out_fname, 'wb') as f:
                pickle.dump(attn_to_semantic, f)

        else:
            res = load_results(res_type=sim_type, tok=conf['tok'], res_metric=sim_metric)
            plot_results(res, plot_type='by_use_case', save_path=os.path.join(RESULTS_DIR, f'PLOT_attn_to_{sim_type}_by_use_case.pdf'))
            plot_results(res, plot_type='by_layer', save_path=os.path.join(RESULTS_DIR, f'PLOT_attn_to_{sim_type}_by_layer.pdf'))

    print(":)")
