from core.attention.extractors import AttentionExtractor, AttributeAttentionExtractor
from utils.general import get_dataset, get_model, get_sample
import os
from pathlib import Path
from core.attention.attention_flow import AttentionGraphExtractor, AttentionGraphUtils, AggregateAttributeAttentionGraph
import pickle
from multiprocessing import Process
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'attention_flow')


def run_attn_flow_experiment(conf, sampler_conf, fine_tune, attn_flow_param, models_dir, res_dir):
    dataset = get_dataset(conf)
    use_case = conf['use_case']
    tok = conf['tok']
    model_name = conf['model_name']

    if fine_tune is not None:
        model_path = os.path.join(models_dir, fine_tune, f"{use_case}_{tok}_tuned")
    else:
        model_path = None
    model = get_model(model_name, fine_tune, model_path)

    complete_sampler_conf = sampler_conf.copy()
    complete_sampler_conf['permute'] = conf['permute']
    sample = get_sample(dataset, complete_sampler_conf)

    if attn_flow_param['text_unit'] == 'attr':
        attn_extr = AttributeAttentionExtractor(sample, model)
        text_unit = 'attr'
    elif attn_flow_param['text_unit'] == 'token':
        attn_extr = AttentionExtractor(sample, model)
        text_unit = 'token'
    else:
        raise NotImplementedError()

    attn_graph_extr = AttentionGraphExtractor(attn_extr, text_unit)
    out_fname = f"{use_case}_{tok}_{sampler_conf['size']}_{fine_tune}_{text_unit}"
    out_file = os.path.join(res_dir, use_case, out_fname)
    attn_graph_extr.extract(out_file=out_file)


def load_saved_attn_graph_data(use_case, conf, sampler_conf, fine_tune, attn_flow_param, res_dir):
    tok = conf['tok']
    size = sampler_conf['size']
    text_unit = attn_flow_param['text_unit']
    out_fname = f"{use_case}_{tok}_{size}_{fine_tune}_{text_unit}"
    data_path = os.path.join(res_dir, use_case, out_fname)
    uc_attn_graph = pickle.load(open(f"{data_path}.pkl", "rb"))
    AttentionGraphExtractor.check_batch_graph_attn_features(uc_attn_graph, text_unit=text_unit)
    return uc_attn_graph


# def plot_attn_graph(att_mat, text_units, plot_attn_graph, plot_attn_flow, plot_title):
#     adj_mat, labels_to_index = AttentionGraphUtils.get_adjmat(mat=att_mat,
#                                                               input_text_units=text_units)
#
#     if plot_attn_graph:
#         G = AttentionGraphUtils.create_attention_graph(adj_mat)
#         AttentionGraphUtils.plot_attention_graph(G, adj_mat, labels_to_index,
#                                                  n_layers=att_mat.shape[0],
#                                                  length=att_mat.shape[-1], title=plot_title)
#
#     if plot_attn_flow:
#         flow_G = AttentionGraphUtils.create_attention_graph(flow_res)
#         AttentionGraphUtils.plot_attention_graph(flow_G, flow_res, labels_to_index,
#                                                  n_layers=att_mat.shape[0],
#                                                  length=att_mat.shape[-1],
#                                                  title=plot_title.replace('graph', 'flow'))

def plot_attention_graph(att_mat, text_units, flow_res, plot_attn_graph, plot_attn_flow, top_selection_method,
                         plot_title, ax=None, save_path=None):
    adj_mat, labels_to_index = AttentionGraphUtils.get_adjmat(mat=att_mat,
                                                              input_text_units=text_units)

    if plot_attn_graph:
        target_res = adj_mat
        target = 'graph'

    if plot_attn_flow:
        target_res = flow_res
        plot_title = plot_title.replace("graph", "flow")
        target = 'flow'

    G = AttentionGraphUtils.create_attention_graph(target_res)

    if ax is None:
        plt.figure(1, figsize=(20, 12))

    AttentionGraphUtils.plot_attention_graph(G, target_res, labels_to_index, n_layers=att_mat.shape[0],
                                             length=att_mat.shape[-1], plot_top_scores=top_selection_method, ax=ax)

    if ax is None:
        plt.title(plot_title)
        plt.gca().axis('off')

        if save_path is not None:
            save_path += f'_{target}.png'
            plt.savefig(save_path, bbox_inches='tight')

        plt.show()
    else:
        ax.set_title(plot_title)


def plot_benchmark(use_cases, conf, sampler_conf, fine_tune, attn_flow_param, plot_params, res_dir):

    assert isinstance(use_cases, list)
    assert len(use_cases) > 0
    assert plot_params['group_by_cat'] is True
    assert plot_params['agg'] is True
    assert plot_params['target_categories'] is not None
    assert isinstance(plot_params['target_categories'], list)
    assert len(plot_params['target_categories']) > 0

    # load saved results and aggregate them
    res = {}
    for use_case in use_cases:
        uc_attn_graph = load_saved_attn_graph_data(use_case, conf, sampler_conf, fine_tune, attn_flow_param,
                                                   RESULTS_DIR)
        aggregator = AggregateAttributeAttentionGraph(uc_attn_graph, target_categories=plot_params['target_categories'])
        _, uc_agg_attn_graph = aggregator.aggregate('mean')

        for cat in uc_agg_attn_graph:
            if cat not in res:
                res[cat] = {use_case: uc_agg_attn_graph[cat]}
            else:
                res[cat][use_case] = uc_agg_attn_graph[cat]

    if plot_params['plot_attn_graph'] is True and plot_params['plot_attn_flow'] is True:
        items_to_plot = [(True, False), (False, True)]
    else:
        items_to_plot = [(plot_params['plot_attn_graph'], plot_params['plot_attn_flow'])]

    for item_to_plot in items_to_plot:
        for cat in res:
            cat_res = res[cat]

            nrows = 6
            ncols = 2
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 18))
            axes = axes.flat
            suptitle = f'average {cat}'
            if item_to_plot[0] is True:
                target = 'attention graph'
                suptitle += f' {target}'
            else:
                target = 'attention flow'
                suptitle += f' {target}'
            fig.suptitle(suptitle)

            for idx, use_case in enumerate(cat_res):
                uc_cat_res = cat_res[use_case]
                ax = axes[idx]

                att_mat = uc_cat_res['attn_vals']
                text_units = uc_cat_res['text_units']
                flow_res = uc_cat_res['flow_vals']

                plot_attention_graph(att_mat, text_units, flow_res, item_to_plot[0], item_to_plot[1],
                                     plot_params['top_selection_method'], plot_title=use_case, ax=ax)
                ax.axis('off')

            plt.tight_layout()
            plt.subplots_adjust(wspace=-0.08, hspace=0.05)

            out_fname = f"{target}_{conf['tok']}_{sampler_conf['size']}_{fine_tune}_{cat}.png"
            out_file = os.path.join(res_dir, out_fname)
            plt.savefig(out_file, bbox_inches='tight')

            plt.show()


def plot_use_cases_sequentially(use_cases, conf, sampler_conf, fine_tune, attn_flow_param, plot_params, res_dir):

    # loop over the use cases
    for use_case in use_cases:

        out_fname = f"{use_case}_{conf['tok']}_{sampler_conf['size']}_{fine_tune}"
        out_file = os.path.join(res_dir, use_case, out_fname)

        # retrieved saved results
        uc_attn_graph = load_saved_attn_graph_data(use_case, conf, sampler_conf, fine_tune, attn_flow_param,
                                                   RESULTS_DIR)

        if plot_params['group_by_cat']:
            # group data and optionally aggregate it
            assert plot_params['target_categories'] is not None
            assert isinstance(plot_params['target_categories'], list)
            assert len(plot_params['target_categories']) > 0
            aggregator = AggregateAttributeAttentionGraph(uc_attn_graph,
                                                          target_categories=plot_params['target_categories'])
            uc_grouped_attn_graph, uc_agg_attn_graph = aggregator.aggregate('mean')

            # plot aggregated results
            if plot_params['agg']:

                # loop over the data categories
                for cat in uc_agg_attn_graph:
                    att_mat = uc_agg_attn_graph[cat]['attn_vals']
                    text_units = uc_agg_attn_graph[cat]['text_units']
                    flow_res = uc_agg_attn_graph[cat]['flow_vals']
                    plot_title = f'{use_case} average {cat} attention graph'
                    out_file += f'_avg_{cat}'

                    plot_attention_graph(att_mat, text_units, flow_res, plot_params['plot_attn_graph'],
                                         plot_params['plot_attn_flow'], plot_params['top_selection_method'], plot_title,
                                         save_path=out_file)

            else:       # plot raw results by grouping them by data category
                for cat in uc_grouped_attn_graph:
                    attn_graph_by_cat = uc_grouped_attn_graph[cat]
                    for i in range(len(attn_graph_by_cat['attn'])):
                        att_mat = attn_graph_by_cat['attn'][i]
                        text_units = attn_graph_by_cat['text_units']
                        flow_res = attn_graph_by_cat['flow'][i]
                        label = attn_graph_by_cat['labels'][i]
                        pred = attn_graph_by_cat['preds'][i]
                        idx = attn_graph_by_cat['idxs'][i]
                        plot_title = f'{use_case} {cat} record#{idx} (label: {label}, pred: {pred}) attention graph'

                        plot_attention_graph(att_mat, text_units, flow_res, plot_params['plot_attn_graph'],
                                             plot_params['plot_attn_flow'], plot_params['top_selection_method'],
                                             plot_title)

        else:
            for idx, (l, r, f) in enumerate(uc_attn_graph):
                att_mat = f['graph_attn']['attns']
                text_units = f['graph_attn']['text_units']
                flow_res = f['graph_attn']['flow']
                label = f['labels'].item()
                pred = f['preds'].item()
                plot_title = f'{use_case} record#{idx} (label: {label}, pred: {pred}) attention graph'

                if att_mat is None or flow_res is None:
                    print(f"Skip record {idx}.")
                    continue

                if plot_params['target_idx'] is not None:
                    if idx != plot_params['target_idx']:
                        continue

                plot_attention_graph(att_mat, text_units, flow_res, plot_params['plot_attn_graph'],
                                     plot_params['plot_attn_flow'], plot_params['top_selection_method'], plot_title)


if __name__ == '__main__':

    use_cases = ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"]
    # use_cases = ["Structured_Fodors-Zagats"]

    conf = {
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
        'size': 50,
        'target_class': 'both',  # 'both', 0, 1
        'seeds': [42, 42],  # [42 -> class 0, 42 -> class 1]
    }

    fine_tune = 'simple'  # None, 'simple', 'advanced'

    attn_flow_param = {
        'text_unit': 'attr'
    }

    # experiment = 'compute_attn_flow', 'plot_attn_flow'
    experiment = 'plot_attn_flow'
    plot_params = {
        'plot_attn_graph': True,
        'plot_attn_flow': False,
        'group_by_cat': True,
        'agg': True,
        'top_selection_method': 'percentile',
        'target_categories': ['all', 'all_pos', 'all_neg'],   # ['all', 'all_pos', 'all_neg'],
        'target_idx': None,
    }

    if experiment == 'compute_attn_flow':
        # # no multi process
        # for use_case in use_cases:
        #     print(use_case)
        #
        #     uc_conf = conf.copy()
        #     uc_conf['use_case'] = use_case
        #     run_attn_flow_experiment(uc_conf, sampler_conf, fine_tune, attn_flow_param, MODELS_DIR, RESULTS_DIR)

        processes = []
        for use_case in use_cases:
            uc_conf = conf.copy()
            uc_conf['use_case'] = use_case
            p = Process(target=run_attn_flow_experiment,
                        args=(uc_conf, sampler_conf, fine_tune, attn_flow_param, MODELS_DIR, RESULTS_DIR,))
            processes.append(p)

        for p in processes:
            p.start()

        for p in processes:
            p.join()

    elif experiment == 'plot_attn_flow':

        assert plot_params['plot_attn_graph'] or plot_params['plot_attn_flow']

        plot_benchmark(use_cases, conf, sampler_conf, fine_tune, attn_flow_param, plot_params, RESULTS_DIR)
        # plot_use_cases_sequentially(use_cases, conf, sampler_conf, fine_tune, attn_flow_param, plot_params, RESULTS_DIR)
