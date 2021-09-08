import os
import pickle
import copy
import numpy as np
from pathlib import Path
import matplotlib.image as mpimg
import pandas as pd
import itertools
import matplotlib.pyplot as plt

from utils.result_collector import TestResultCollector
from utils.plot import plot_results, plot_benchmark_results, plot_agg_results, plot_comparison, plot_images_grid
from utils.test_utils import ConfCreator
from utils.general import get_testers


PROJECT_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'attention')


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

    testers = get_testers(tester_params)

    assert len(testers) == 1
    tester = testers[0]

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

    # if len(results) > 0:
    #     if len(results) == 1:
    #         results = list(results.values())[0]

    return results, tester


def cmp_results(res1: dict, res2: dict):
    def check_data(res):
        assert isinstance(res, dict)
        for cat in res:
            if res[cat] is not None:
                assert isinstance(res[cat], TestResultCollector)

    check_data(res1)
    check_data(res2)
    res1 = copy.deepcopy(res1)
    res2 = copy.deepcopy(res2)

    cmp_res = {}
    for cat in res1:
        cat_res1 = None
        if cat in res1:
            cat_res1 = res1[cat]

        cat_res2 = None
        if cat in res2:
            cat_res2 = res2[cat]

        if cat_res1 is None or cat_res2 is None:
            print(f"Skip {cat} results.")
            continue

        out_cat_res = copy.deepcopy(cat_res1)
        out_cat_res.transform_collector(cat_res2, lambda x, y: x - y)
        cmp_res[cat] = out_cat_res

    return cmp_res


def cmp_benchmark_results(results1, results2):
    bench_cmp_res = {}
    for use_case in list(results1):
        res1 = results1[use_case]
        res2 = results2[use_case]

        cmp_res = cmp_results(res1, res2)
        bench_cmp_res[use_case] = cmp_res

    return bench_cmp_res


def aggregate_results(results: dict, agg_fns: list, result_ids: list):
    assert isinstance(results, dict)
    assert isinstance(agg_fns, list)
    assert len(agg_fns) > 0
    assert isinstance(result_ids, list)
    assert len(results) > 0

    new_agg_fns = []
    for agg_fn in agg_fns:
        if agg_fn == 'row_mean':
            agg_fn = lambda x: x.mean(1)
        elif agg_fn == 'row_std':
            agg_fn = lambda x: x.std(1)
        else:
            raise ValueError("Wrong value for the aggregate function.")
        new_agg_fns.append(agg_fn)

    agg_cat_results = {}
    out_use_cases = []

    # aggregate the results
    for use_case in results:  # loop over the results
        use_case_res = results[use_case]
        assert isinstance(use_case_res, dict)
        out_use_cases.append(use_case)

        for cat in use_case_res:  # loop over category results

            target_res = {}
            for idx, agg_fn in enumerate(new_agg_fns):
                cat_res = copy.deepcopy(use_case_res[cat])
                assert isinstance(cat_res, TestResultCollector)

                agg_target_res = {}
                for result_id in result_ids:  # aggregate the result ids
                    res = cat_res.get_result(result_id)
                    agg_target_res[result_id] = agg_fn(res).reshape((-1, 1))
                target_res[agg_fns[idx]] = agg_target_res

            if cat not in agg_cat_results:
                agg_cat_results[cat] = [target_res]
            else:
                agg_cat_results[cat].append(target_res)

    agg_results = {}  # concat aggregated results
    for cat in agg_cat_results:
        agg_cat_res = agg_cat_results[cat]
        assert isinstance(agg_cat_res, list)
        assert len(agg_cat_res) > 0

        current_cat_res = {}
        for use_case_res in agg_cat_res:
            if len(current_cat_res) == 0:
                current_cat_res = copy.deepcopy(use_case_res)
            else:
                for agg_metric in current_cat_res:

                    for result_id in current_cat_res[agg_metric]:
                        current_cat_res[agg_metric][result_id] = np.concatenate([current_cat_res[agg_metric][result_id],
                                                                                 use_case_res[agg_metric][result_id]],
                                                                                axis=1)

        agg_results[cat] = current_cat_res

    return agg_results


def cmp_agg_results(res1, res2, target_cats):
    cmp_res = {}
    for cat in res1:

        if cat not in target_cats:
            continue

        cat_res1 = None
        if cat in res1:
            cat_res1 = res1[cat]

        cat_res2 = None
        if cat in res2:
            cat_res2 = res2[cat]

        if cat_res1 is None or cat_res2 is None:
            print(f"Skip {cat} results.")
            continue

        out_cat_res = copy.deepcopy(cat_res1)
        for metric in cat_res2:
            for res_id in cat_res2[metric]:
                out_cat_res[metric][res_id] -= cat_res2[metric][res_id]
        cmp_res[cat] = out_cat_res

    return cmp_res


def use_case_analysis(conf: dict, plot_params: list, categories: list, agg_fns: list = None,
                                  target_agg_result_ids: list = None, plot_type: str = 'simple', save_path: str = None):

    assert isinstance(conf, dict), "Wrong data type for parameter 'conf'."
    assert isinstance(plot_params, list), "Wrong data type for parameter 'plot_params'."
    assert len(plot_params) > 0, "Wrong value for parameter 'plot_params'."
    assert isinstance(categories, list), "Wrong data type for parameter 'categories'."
    assert len(categories) > 0, "Wrong value for parameter 'categories'."
    if agg_fns is not None:
        assert isinstance(agg_fns, list), "Wrong data type for parameter 'agg_fns'."
        assert len(agg_fns) > 0, "Empty aggregation functions."
    if target_agg_result_ids is not None:
        assert isinstance(target_agg_result_ids, list), "Wrong data type for parameter 'target_agg_result_ids'."
        assert len(target_agg_result_ids) > 0, "Empty target aggregated results."

    res, tester = get_results(conf, [conf['use_case']])

    if agg_fns is not None:
        res = aggregate_results(res, agg_fns, target_agg_result_ids)
        display_uc = [conf_creator.use_case_map[conf['use_case']]]
        plot_agg_results(res, target_cats=categories, xticks=display_uc, vmin=-0.5, vmax=0.5, agg=False,
                         plot_type=plot_type, save_path=save_path)

    else:
        plot_results(res, tester, target_cats=categories, plot_params=plot_params, plot_type=plot_type,
                     save_path=save_path)


def use_case_comparison_analysis(confs: list, plot_params: list, categories: list, compared_methods: list,
                                 agg_fns: list = None, target_agg_result_ids: list = None, only_diff: bool = True):
    assert isinstance(confs, list), "Wrong data type for parameter 'confs'."
    assert len(confs) > 1, "Wrong value for parameter 'confs'."

    for conf_idx1 in range(len(confs) - 1):
        conf1 = confs[conf_idx1]
        res1, tester = get_results(conf1, [conf1['use_case']])

        for conf_idx2 in range(conf_idx1 + 1, len(confs)):
            conf2 = confs[conf_idx2]
            res2, tester = get_results(conf2, [conf2['use_case']])

            cmp_vals = []
            for param in conf1:
                if conf1[param] != conf2[param]:
                    cmp_vals.append(conf1[param])
                    cmp_vals.append(conf2[param])
                    break

            assert list(res1.keys()) == list(res2.keys())
            assert len(res1) == 1

            if agg_fns is not None:
                res1 = aggregate_results(res1, agg_fns, target_agg_result_ids)
                res2 = aggregate_results(res2, agg_fns, target_agg_result_ids)

                cmp_res = cmp_agg_results(res1, res2, target_cats=categories)
                display_uc = [conf_creator.use_case_map[conf2['use_case']]]

                if only_diff:
                    res1 = res2 = res1_name = res2_name = None
                else:
                    res1_name = compared_methods[0]
                    res2_name = compared_methods[1]

                plot_agg_results(cmp_res, target_cats=categories, title_prefix=f'{cmp_vals[0]} vs {cmp_vals[1]}',
                                 xticks=display_uc, vmin=-0.5, vmax=0.5, res1=res1, res2=res2, res1_name=res1_name,
                                 res2_name=res2_name)

            else:
                res1 = list(res1.values())[0]
                res2 = list(res2.values())[0]
                cmp_res = cmp_results(res1, res2)
                plot_comparison(res1, res2, cmp_res, tester, cmp_vals, target_cats=categories, plot_params=plot_params)


def benchmark_analysis(conf: dict, plot_params: list, categories: list, agg_fns: list = None,
                       target_agg_result_ids: list = None):
    assert isinstance(conf, dict), "Wrong data type for parameter 'conf'."
    assert isinstance(plot_params, list), "Wrong data type for parameter 'plot_params'."
    assert len(plot_params) > 0, "Empty plot params."
    assert isinstance(categories, list), "Wrong data type for parameter 'categories'."
    assert len(categories) > 0, "Empty categories."
    if agg_fns is not None:
        assert isinstance(agg_fns, list), "Wrong data type for parameter 'agg_fns'."
        assert len(agg_fns) > 0, "Empty aggregation functions."
    if target_agg_result_ids is not None:
        assert isinstance(target_agg_result_ids, list), "Wrong data type for parameter 'target_agg_result_ids'."
        assert len(target_agg_result_ids) > 0, "Empty target aggregated results."

    use_cases = conf['use_case']
    assert isinstance(use_cases, list), "Wrong type for configuration use_case param."
    assert len(use_cases) > 0, "Empty use case list."
    res, tester = get_results(conf, use_cases)

    if agg_fns is not None:
        res = aggregate_results(res, agg_fns, target_agg_result_ids)
        display_uc = [conf_creator.use_case_map[uc] for uc in conf_creator.conf_template['use_case']]
        plot_agg_results(res, target_cats=categories, xticks=display_uc, vmin=-0.5, vmax=0.5, agg=True)
    else:
        plot_benchmark_results(res, tester, use_cases, target_cats=categories, plot_params=plot_params)


def benchmark_comparison_analysis(confs: list, plot_params: list, categories: list, compared_methods: list,
                                  agg_fns: list = None,  target_agg_result_ids: list = None, only_diff: bool = True,
                                  save_path: str = None):
    assert isinstance(confs, list), "Wrong data type for parameter 'confs'."
    assert len(confs) > 1, "Wrong value for parameter 'confs'."

    for conf in confs:
        assert isinstance(conf['use_case'], list), "Wrong type for configuration use_case param."
        assert len(conf['use_case']) > 0, "Empty use case list."
    for idx in range(len(confs) - 1):
        assert confs[idx]['use_case'] == confs[idx + 1]['use_case'], "Use cases not equal."
    use_cases = confs[0]['use_case']

    assert isinstance(compared_methods, list)
    assert len(compared_methods) == 2
    assert all([isinstance(p, str) for p in compared_methods])

    for conf_idx1 in range(len(confs) - 1):
        conf1 = confs[conf_idx1]
        res1, tester = get_results(conf1, use_cases)

        for conf_idx2 in range(conf_idx1 + 1, len(confs)):
            conf2 = confs[conf_idx2]
            res2, tester = get_results(conf2, use_cases)

            cmp_vals = []
            for param in conf1:
                if conf1[param] != conf2[param]:
                    cmp_vals.append(conf1[param])
                    cmp_vals.append(conf2[param])
                    break

            print(conf1)
            print(conf2)

            if agg_fns is not None:
                res1 = aggregate_results(res1, agg_fns, target_agg_result_ids)
                res2 = aggregate_results(res2, agg_fns, target_agg_result_ids)

                cmp_res = cmp_agg_results(res1, res2, target_cats=categories)
                display_uc = [conf_creator.use_case_map[uc] for uc in conf_creator.conf_template['use_case']]

                if only_diff:
                    res1 = res2 = res1_name = res2_name = None
                else:
                    res1_name = compared_methods[0]
                    res2_name = compared_methods[1]

                plot_agg_results(cmp_res, target_cats=categories, title_prefix=f'{cmp_vals[0]} vs {cmp_vals[1]}',
                                 xticks=display_uc, agg=True, vmin=-0.5, vmax=0.5, res1=res1, res2=res2,
                                 res1_name=res1_name, res2_name=res2_name, save_path=save_path)

            else:
                cmp_res = cmp_benchmark_results(res1, res2)
                plot_benchmark_results(cmp_res, tester, use_cases, target_cats=categories,
                                       title_prefix=f'{cmp_vals[0]} {cmp_vals[1]}', plot_params=plot_params, vmin=-0.5,
                                       vmax=0.5, save_path=save_path)


def plot_pattern_freq(stats, save_path: str = None):

    ncols = 4
    nrows = 3
    if len(stats) == 1:
        ncols = 1
        nrows = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10), sharey=True)
    if len(stats) > 1:
        axes = axes.flat

    max_val = 0
    for uc in stats:
        uc_stats = stats[uc]
        if uc_stats.values.max() > max_val:
            max_val = uc_stats.values.max()

    for idx, uc in enumerate(stats):
        if len(stats) > 1:
            ax = axes[idx]
        else:
            ax = axes

        use_case_stats = stats[uc]

        if len(stats) > 1:
            use_case_stats.plot(kind='bar', ax=ax, legend=False, rot=0)
        else:
            use_case_stats.plot(kind='bar', ax=ax, legend=True, rot=0)

        ax.set_title(uc, fontsize=18)
        if idx % ncols == 0:
            ax.set_ylabel("Freq. (%)", fontsize=20)
        # ax.set_xlabel("", fontsize=20)
        ax.xaxis.set_tick_params(labelsize=18)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.set_yticks(np.arange(0, max_val, 5))

    if len(stats) > 1:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.74, 0.08), ncol=4, fontsize=20)
    plt.subplots_adjust(wspace=0.01, hspace=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # plt.tight_layout()
    plt.show()


def extract_pattern_data_by_conf(data, target_layer, target_head, target_pattern, target_pattern_metric, target_metric):
    assert isinstance(data, dict)

    layer_comb = [target_layer]
    head_combs = [target_head] if target_head != 'q_all' else ['q0', 'q1', 'q2', 'q3']
    if target_metric is None:
        pattern_comb = [target_pattern] if target_pattern != 'all' else ['vertical', 'diag', 'match']
        pattern_or_metric = [(p, target_pattern_metric) for p in pattern_comb]
    else:
        pattern_or_metric = [target_metric]

    conf_combs = (layer_comb, head_combs, pattern_or_metric)
    confs = list(itertools.product(*conf_combs))
    out_data = {}
    for data_conf in confs:
        if isinstance(data_conf[2], tuple):
            sub_key = f'{data_conf[2][0]}_{data_conf[2][1]}'
        else:
            sub_key = data_conf[2]
        data_key = f'{data_conf[0]}_{data_conf[1]}_{sub_key}'
        if 'freq' in data_key:
            try:
                item = data[data_key]
            except KeyError:
                item = data[data_key.replace('freq', 'existence')]
            if len(item.reshape(-1)) == 1:
                item = item.reshape(-1)[0]
            out_data[data_key] = item

    return out_data


def plot_pattern_freq_stats(stats: dict, target_layer: str, target_head: str, target_pattern: str = None,
                            target_pattern_metric: str = None, target_metric: str = None,
                            comparison: bool = False, save_path: str = None, legend: str = None):
    assert isinstance(stats, dict)
    assert isinstance(target_layer, str)
    assert target_layer in ['layers', 'tot', 'avg']
    assert isinstance(target_head, str)
    assert target_head in ['q_all', 'q0', 'q1', 'q2', 'q3', 'all']
    patterns = ['vertical', 'diag', 'match']
    if target_pattern is not None:
        assert isinstance(target_pattern, str)
        assert target_pattern in ['all'] + patterns
    if target_pattern_metric is not None:
        assert isinstance(target_pattern_metric, str)
        assert target_pattern_metric in ['freq', 'locs']
    if target_pattern is not None:
        assert target_pattern_metric is not None
    if target_pattern_metric is not None:
        assert target_pattern is not None
    if target_metric is not None:
        assert isinstance(target_metric, str)
        assert target_metric in ['avg', 'entropy']
    if target_pattern is not None:
        assert target_metric is None
    else:
        assert target_metric is not None
    if target_metric is not None:
        assert target_pattern is None
    else:
        assert target_pattern is not None
    assert isinstance(comparison, bool)
    if legend is not None:
        assert isinstance(legend, str)
    assert legend in ['pattern']

    # prepare data for plot
    plot_data = {}
    if comparison:
        for uc in stats:
            uc_plot_data = {}
            uc_stats = stats[uc]
            for method in uc_stats:
                method_collector = uc_stats[method]
                assert isinstance(method_collector, TestResultCollector)
                method_stats = method_collector.get_results()
                method_stats = extract_pattern_data_by_conf(method_stats, target_layer, target_head, target_pattern,
                                                            target_pattern_metric, target_metric)
                uc_plot_data[method] = method_stats
            uc_plot_data = pd.DataFrame.from_dict(uc_plot_data, orient='index')

            if target_pattern is not None and target_pattern == 'all':
                uc_plot_data.columns = patterns

            plot_data[uc] = uc_plot_data.T

    else:
        for uc in stats:
            uc_collector = stats[uc]
            assert isinstance(uc_collector, TestResultCollector)
            uc_stats = uc_collector.get_results()
            uc_stats = extract_pattern_data_by_conf(uc_stats, target_layer, target_head, target_pattern,
                                                    target_pattern_metric, target_metric)
            uc_plot_data = pd.DataFrame([uc_stats])

            if target_pattern is not None and target_pattern == 'all':
                uc_plot_data.columns = patterns

            plot_data[uc] = uc_plot_data.T

    if target_layer == 'layers':
        raise NotImplementedError()
    else:
        if target_head == 'q_all' or target_pattern_metric == 'locs':
            raise NotImplementedError()

        plot_pattern_freq(plot_data, save_path=save_path)


if __name__ == '__main__':

    use_cases = ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"]

    conf = {
        'use_case': "Structured_Beer",  # when analysis_target = 'benchmark' this field will be set with all use cases
        'data_type': 'train',
        'permute': False,
        'model_name': 'bert-base-uncased',
        'tok': 'sent_pair',
        'size': None,
        'fine_tune_method': None,       # None, 'simple'
        'extractor': {
            'attn_extractor': 'attr_extractor',  # 'attr_extractor', 'word_extractor'
            'attn_extr_params': {'special_tokens': True, 'agg_metric': 'mean'},
        },
        'tester': {
            'tester': 'attr_pattern_tester',    # 'attr_tester', 'attr_pattern_tester'
            'tester_params': {'ignore_special': True}
        },
    }
    conf_creator = ConfCreator()
    conf_creator.validate_conf(conf)
    use_case_map = ConfCreator().use_case_map

    experiment = 'pattern_freq'     # 'pattern_freq', 'pattern'

    if experiment == 'pattern':
        analysis_target = 'benchmark'    # 'use_case', 'benchmark'
        analysis_type = 'comparison'    # 'simple', 'comparison', 'multi'

        # plot_params = ['match_attr_attn_loc', 'match_attr_attn_over_mean',
        #                'avg_attr_attn', 'attr_attn_3_last', 'attr_attn_last_1',
        #                'attr_attn_last_2', 'attr_attn_last_3',
        #                'avg_attr_attn_3_last', 'avg_attr_attn_last_1',
        #                'avg_attr_attn_last_2', 'avg_attr_attn_last_3']
        plot_params = ['attr_attn_3_last', 'match_attr_attn_loc', 'match_attr_attn_over_mean',
                       'avg_attr_attn', 'attr_attn_last_1',
                       'attr_attn_last_2', 'attr_attn_last_3']
        plot_params = ['match_attr_attn_over_mean']
        # plot_params = ['match_attr_attn_loc']

        # aggregation
        # agg_fns = None
        # target_agg_result_ids = None
        agg_fns = ['row_mean', 'row_std']
        target_agg_result_ids = ['match_attr_attn_loc']

        categories = ['all']

        extractor_name = conf['extractor']['attn_extractor']
        tester_name = conf['tester']['tester']
        agg_metric = conf['extractor']['attn_extr_params']['agg_metric']

        if analysis_target == 'use_case':

            if analysis_type == 'simple':
                use_case_analysis(conf, plot_params, categories, agg_fns, target_agg_result_ids)

            elif analysis_type == 'comparison':
                comparison_param = 'fine_tune_method'  # 'tok', 'fine_tune_method'

                if comparison_param == 'fine_tune_method':
                    compared_methods = ['Pre-training', 'Fine-tuning']
                elif comparison_param == 'tok':
                    compared_methods = ['Attr-pair', 'Sent-pair']
                else:
                    raise ValueError("Wrong comparison param.")

                confs = conf_creator.get_confs(conf, [comparison_param])
                use_case_comparison_analysis(confs, plot_params, categories, compared_methods, agg_fns,
                                             target_agg_result_ids, only_diff=False)

            else:
                raise NotImplementedError()

        elif analysis_target == 'benchmark':

            bench_conf = conf.copy()
            bench_conf['use_case'] = conf_creator.conf_template['use_case']

            if analysis_type == 'simple':

                benchmark_analysis(bench_conf, plot_params, categories, agg_fns, target_agg_result_ids)

            elif analysis_type == 'multi':

                assert agg_fns is None
                assert target_agg_result_ids is None
                assert plot_params == ['match_attr_attn_over_mean']
                assert len(categories) == 1

                template_file_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(conf['data_type'], extractor_name, tester_name,
                                                                            conf['fine_tune_method'], conf['permute'],
                                                                            conf['tok'],
                                                                            conf['size'], analysis_target, analysis_type,
                                                                            agg_metric)

                confs = conf_creator.get_confs(conf, ['use_case'])
                imgs = []
                for conf in confs:
                    uc = conf['use_case']
                    out_file = os.path.join(RESULTS_DIR, uc, f'PLOT_{uc}_{template_file_name}')
                    Path(os.path.join(RESULTS_DIR, uc)).mkdir(parents=True, exist_ok=True)
                    use_case_analysis(conf, plot_params, categories, agg_fns, target_agg_result_ids, plot_type='advanced',
                                      save_path=out_file)
                    imgs.append(mpimg.imread(f'{out_file}_{plot_params[0]}.png'))

                save_path = os.path.join(RESULTS_DIR, f'GRID_PLOT_{template_file_name}_{plot_params[0]}.pdf')
                plot_images_grid(imgs, nrows=3, ncols=4, save_path=save_path)

            elif analysis_type == 'comparison':

                comparison_param = 'tok'   # 'tok', 'fine_tune_method'

                template_file_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(conf['data_type'], extractor_name, tester_name,
                                                                         conf['permute'],  conf['size'], agg_metric,
                                                                         analysis_target, analysis_type, comparison_param)

                if comparison_param == 'fine_tune_method':
                    compared_methods = ['Pre-training', 'Fine-tuning']
                    template_file_name = f'{template_file_name}_{conf["tok"]}'
                elif comparison_param == 'tok':
                    compared_methods = ['Attr-pair', 'Sent-pair']
                    template_file_name = f'{template_file_name}_{conf["fine_tune_method"]}'
                else:
                    raise ValueError("Wrong comparison param.")

                bench_confs = conf_creator.get_confs(bench_conf, [comparison_param])

                save_path = os.path.join(RESULTS_DIR, f'PLOT_LOC_{template_file_name}_{plot_params[0]}.pdf')
                benchmark_comparison_analysis(bench_confs, plot_params, categories, compared_methods, agg_fns,
                                              target_agg_result_ids, only_diff=False, save_path=save_path)

            else:
                raise NotImplementedError()

        else:
            raise NotImplementedError()

    elif experiment == 'pattern_freq':
        analysis_type = 'comparison'    # 'simple', 'comparison'
        conf['use_case'] = use_cases
        categories = ['all']
        assert len(categories) == 1

        extractor_name = conf['extractor']['attn_extractor']
        tester_name = conf['tester']['tester']
        agg_metric = conf['extractor']['attn_extr_params']['agg_metric']

        cat = categories[0]
        if analysis_type == 'simple':
            res, _ = get_results(conf, use_cases)
            res = {use_case_map[uc]: res[uc][cat] for uc in res}
            plot_pattern_freq_stats(res, 'tot', 'all', target_pattern='all', target_pattern_metric='freq',
                                    legend='pattern')

        elif analysis_type == 'comparison':
            comparison = 'tune_tok'  # 'tune', 'tok', 'tune_tok'

            if comparison == 'tune':
                new_conf = conf.copy()
                new_conf['fine_tune_method'] = None
                pretrain_res, _ = get_results(new_conf, use_cases)
                pretrain_res = {uc: pretrain_res[uc][cat] for uc in pretrain_res}

                new_conf['fine_tune_method'] = 'simple'
                tuned_res, _ = get_results(new_conf, use_cases)
                tuned_res = {uc: tuned_res[uc][cat] for uc in tuned_res}

                plot_res = {}
                for uc in pretrain_res:
                    uc_pretrain_res = pretrain_res[uc]
                    uc_tuned_res = tuned_res[uc]
                    plot_res[use_case_map[uc]] = {'pretrain': uc_pretrain_res, 'tune': uc_tuned_res}

                plot_pattern_freq_stats(plot_res, 'tot', 'all', target_pattern='all', target_pattern_metric='freq',
                                        legend='pattern', comparison=True)

            elif comparison == 'tok':
                new_conf = conf.copy()
                new_conf['tok'] = 'sent_pair'
                sent_res, _ = get_results(new_conf, use_cases)
                sent_res = {uc: sent_res[uc][cat] for uc in sent_res}

                new_conf['tok'] = 'attr_pair'
                attr_res, _ = get_results(new_conf, use_cases)
                attr_res = {uc: attr_res[uc][cat] for uc in attr_res}

                plot_res = {}
                for uc in sent_res:
                    uc_sent_res = sent_res[uc]
                    uc_attr_res = attr_res[uc]
                    plot_res[use_case_map[uc]] = {'sent': uc_sent_res, 'attr': uc_attr_res}

                plot_pattern_freq_stats(plot_res, 'tot', 'all', target_pattern='all', target_pattern_metric='freq',
                                        legend='pattern', comparison=True)

            elif comparison == 'tune_tok':

                template_file_name = '{}_{}_{}_{}_{}_{}_{}_{}'.format(conf['data_type'], extractor_name, tester_name,
                                                                      conf['permute'],  conf['size'], agg_metric,
                                                                      analysis_type, comparison)

                new_conf = conf.copy()
                new_conf['fine_tune_method'] = None
                new_conf['tok'] = 'sent_pair'
                pretrain_sent_res, _ = get_results(new_conf, use_cases)
                pretrain_sent_res = {uc: pretrain_sent_res[uc][cat] for uc in pretrain_sent_res}

                new_conf['tok'] = 'attr_pair'
                pretrain_attr_res, _ = get_results(new_conf, use_cases)
                pretrain_attr_res = {uc: pretrain_attr_res[uc][cat] for uc in pretrain_attr_res}

                new_conf['fine_tune_method'] = 'simple'
                new_conf['tok'] = 'sent_pair'
                tuned_sent_res, _ = get_results(new_conf, use_cases)
                tuned_sent_res = {uc: tuned_sent_res[uc][cat] for uc in tuned_sent_res}

                new_conf['tok'] = 'attr_pair'
                tuned_attr_res, _ = get_results(new_conf, use_cases)
                tuned_attr_res = {uc: tuned_attr_res[uc][cat] for uc in tuned_attr_res}

                plot_res = {}
                for uc in pretrain_sent_res:
                    uc_pt_sent_res = pretrain_sent_res[uc]
                    uc_pt_attr_res = pretrain_attr_res[uc]
                    uc_ft_sent_res = tuned_sent_res[uc]
                    uc_ft_attr_res = tuned_attr_res[uc]

                    plot_res[use_case_map[uc]] = {'pt_sent': uc_pt_sent_res, 'pt_attr': uc_pt_attr_res, 'ft_sent': uc_ft_sent_res,
                                    'ft_attr': uc_ft_attr_res}

                save_path = os.path.join(RESULTS_DIR, f'PLOT_PATTERN_{template_file_name}.pdf')
                plot_pattern_freq_stats(plot_res, 'tot', 'all', target_pattern='all', target_pattern_metric='freq',
                                        legend='pattern', comparison=True, save_path=save_path)

        else:
            raise NotImplementedError()


    else:
        raise NotImplementedError()
