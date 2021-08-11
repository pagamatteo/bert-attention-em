import os
import time

import pandas as pd
from transformers import AutoTokenizer
from utils.general import get_dataset, get_model, get_sample
from pathlib import Path
from explanation.gradient.plot_gradients import plot_grads, plot_batch_grads, plot_multi_use_case_grads,\
    plot_top_grad_stats
from explanation.gradient.extractors import EntityGradientExtractor, AggregateAttributeGradient
from explanation.gradient.analyzers import TopKGradientAnalyzer
import logging
from multiprocessing import Process
import pickle

PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
RESULT_DIR = os.path.join(PROJECT_DIR, 'results', 'gradient_analysis')


def run_gradient_test(conf, sampler_conf, fine_tune, grad_params, models_dir, res_dir):
    assert isinstance(grad_params, dict), "Wrong data type for parameter 'grad_params'."
    params = ['text_unit', 'special_tokens']  # , 'agg', 'agg_target_cat']
    assert all([p in grad_params for p in params])

    dataset = get_dataset(conf)
    tok = conf['tok']
    model_name = conf['model_name']
    use_case = conf['use_case']

    grad_text_unit = grad_params['text_unit']
    grad_special_tokens = grad_params['special_tokens']
    # grad_agg = grad_params['agg']
    # grad_agg_target_cat = grad_params['agg_target_cat']

    if fine_tune is not None:
        model_path = os.path.join(models_dir, fine_tune, f"{use_case}_{tok}_tuned")
    else:
        model_path = None
    model = get_model(model_name, fine_tune, model_path)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)

    complete_sampler_conf = sampler_conf.copy()
    complete_sampler_conf['permute'] = conf['permute']
    sample = get_sample(dataset, complete_sampler_conf)

    # [END] MODEL AND DATA LOADING

    entity_grad_extr = EntityGradientExtractor(
        model,
        tokenizer,
        grad_text_unit,
        special_tokens=grad_special_tokens,
        show_progress=True
    )
    out_fname = f"{use_case}_{tok}_{sampler_conf['size']}_{fine_tune}_{grad_text_unit}_{grad_special_tokens}_ALL"
    out_dir = os.path.join(res_dir, use_case, out_fname)
    # save grads data
    entity_grad_extr.extract(sample, sample.max_len, out_path=out_dir)

    # if grad_agg:        # aggregate gradient data
    #     if grad_text_unit == 'attrs':
    #         aggregator = AggregateAttributeGradient(grads_data, target_categories=grad_agg_target_cat)
    #         agg_grads = aggregator.aggregate(grad_agg)
    #     else:
    #         raise NotImplementedError("No aggregation for non-attrs gradients.")


def load_saved_grads_data(use_case, conf, sampler_conf, fine_tune, grad_conf):
    tok = conf['tok']
    size = sampler_conf['size']
    text_unit = grad_conf['text_unit']
    special_tokens = grad_conf['special_tokens']
    out_fname = f"{use_case}_{tok}_{size}_{fine_tune}_{text_unit}_{special_tokens}_ALL"
    data_path = os.path.join(RESULT_DIR, use_case, out_fname)
    uc_grad = pickle.load(open(f"{data_path}.pkl", "rb"))
    return uc_grad


if __name__ == '__main__':

    # [BEGIN] PARAMS

    use_cases = ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"]
    # use_cases = ["Structured_Fodors-Zagats"]
    # use_cases = ["Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
    #              "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"]

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

    grad_conf = {
        'text_unit': 'words',
        'special_tokens': True,
        'agg': None,  # 'mean'
        'agg_target_cat': ['all', 'all_pos', 'all_neg', 'all_pred_pos', 'all_pred_neg', 'tp', 'tn', 'fp', 'fn']
    }

    # 'compute_grad', 'plot_grad', 'topk_word_grad', 'topk_word_grad_by_attr', 'topk_word_grad_by_attr_similarity'
    experiment = 'topk_word_grad_by_attr_similarity'

    # [END] PARAMS

    start = time.time()

    if experiment == 'compute_grad':
        # no multi process
        for use_case in use_cases:
            print(use_case)

            if not use_case == 'Structured_DBLP-ACM':
                continue

            uc_conf = conf.copy()
            uc_conf['use_case'] = use_case
            run_gradient_test(uc_conf, sampler_conf, fine_tune, grad_conf, MODELS_DIR, RESULT_DIR)

        # processes = []
        # for use_case in use_cases:
        #     uc_conf = conf.copy()
        #     uc_conf['use_case'] = use_case
        #     p = Process(target=run_gradient_test,
        #                 args=(uc_conf, sampler_conf, fine_tune, grad_conf, MODELS_DIR, RESULT_DIR,))
        #     processes.append(p)
        #
        # for p in processes:
        #     p.start()
        #
        # for p in processes:
        #     p.join()

    elif experiment == 'plot_grad':
        if grad_conf['text_unit'] == 'attrs':
            grad_agg_metric = 'max'
            out_plot_name = os.path.join(RESULT_DIR,
                                         f"grad_{conf['tok']}_{sampler_conf['size']}_{grad_conf['text_unit']}_{grad_agg_metric}.pdf")
            plot_multi_use_case_grads(conf, sampler_conf, fine_tune, grad_conf, use_cases, RESULT_DIR,
                                      grad_agg_metrics=[grad_agg_metric], plot_type='box', out_plot_name=out_plot_name)
            # plot_multi_use_case_grads(conf, sampler_conf, fine_tune, grad_conf, use_cases, RESULT_DIR,
            #                           grad_agg_metrics=['max', 'avg'], plot_type='error')
        else:
            for use_case in use_cases:
                uc_grad = load_saved_grads_data(use_case, conf, sampler_conf, fine_tune, grad_conf)
                out_plot_name = os.path.join(RESULT_DIR, use_case,
                                             f"grad_{grad_conf['text_unit']}_{grad_conf['special_tokens']}")
                plot_batch_grads(uc_grad, 'all', title_prefix=use_case, out_plot_name=out_plot_name,
                                 ignore_special=True)

    elif experiment in ['topk_word_grad', 'topk_word_grad_by_attr', 'topk_word_grad_by_attr_similarity']:
        stats_data = []
        for use_case in use_cases:
            uc_grad = load_saved_grads_data(use_case, conf, sampler_conf, fine_tune, grad_conf)
            if experiment == 'topk_word_grad':
                analyzer = TopKGradientAnalyzer(uc_grad, topk=5)
                analyzed_data = analyzer.analyze('pos', by_attr=False, target_categories=['all', 'all_pos', 'all_neg'])
            else:
                uc_conf = conf.copy()
                uc_conf['use_case'] = use_case
                dataset = get_dataset(uc_conf)
                uc_sampler_conf = sampler_conf.copy()
                uc_sampler_conf['permute'] = uc_conf['permute']
                sample = get_sample(dataset, uc_sampler_conf)

                if experiment == 'topk_word_grad_by_attr':
                    analyzer = TopKGradientAnalyzer(uc_grad, topk=3)
                    analyzed_data = analyzer.analyze('pos', by_attr=True, target_categories=['all'], entities=sample)
                else:
                    topk_range = [1, 3, 5, 10]
                    # topk_range = [3]
                    target_categories = ['all_pos']
                    analyzed_data = {}
                    for topk in topk_range:
                        analyzer = TopKGradientAnalyzer(uc_grad, topk=topk)
                        single_analyzed_data = analyzer.analyze('sim', by_attr=True,
                                                                target_categories=target_categories, entities=sample)
                        analyzed_data.update(single_analyzed_data)

                    if len(target_categories) == 1 and len(topk_range) > 1:
                        analyzed_data_table = pd.concat(list(analyzed_data.values()), axis=0)
                        analyzed_data_table.index = list(analyzed_data.keys())
                        analyzed_data = {
                            target_categories[0]: analyzed_data_table
                        }

            if len(stats_data) == 0:
                for key in analyzed_data:
                    stats_data.append({use_case: analyzed_data[key]})
            else:
                for i, key in enumerate(analyzed_data):
                    stats_data[i][use_case] = analyzed_data[key]

        out_plot_name = os.path.join(RESULT_DIR, f"word_grad_analysis_{conf['tok']}_{sampler_conf['size']}.pdf")
        if experiment == 'topk_word_grad_by_attr':
            plot_top_grad_stats(stats_data, out_plot_name=out_plot_name)
        else:
            plot_top_grad_stats(stats_data, out_plot_name=out_plot_name, stacked=False, share_legend=False)

    else:
        raise ValueError("Experiment not found.")

    end = time.time()
    print(end - start)
