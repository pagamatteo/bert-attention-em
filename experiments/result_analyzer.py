import os
import pickle
import copy
import numpy as np
from pathlib import Path

from utils.result_collector import TestResultCollector
from attention.testers import GenericAttributeAttentionTest
from utils.plot import plot_results, plot_benchmark_results, plot_agg_results, plot_comparison
from experiments.confs import ConfCreator
from utils.general import get_testers


PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'analysis')


def get_results(conf: dict, use_cases: list):
    assert isinstance(conf, dict), "Wrong data type for parameter 'conf'."
    assert isinstance(use_cases, list), "Wrong data type for parameter 'use_cases'."
    assert len(use_cases) > 0, "Empty use case list."

    tester_params = {}
    tester_name = conf['tester']
    if tester_name == 'attr_tester':
        tester_param = {
            'permute': conf['permute'],
            'model_attention_grid': (12, 12),
        }
        tester_params[tester_name] = tester_param
    else:
        raise ValueError("Wrong tester name.")

    testers = get_testers(tester_params)

    assert len(testers) == 1
    tester = testers[0]

    results = {}
    for use_case in use_cases:

        out_path = os.path.join(RESULTS_DIR, use_case)

        template_file_name = '{}_{}_{}_{}_{}_{}_{}_AVG.pickle'.format(use_case, conf['data_type'], conf['extractor'],
                                                                      conf['tester'], conf['fine_tune_method'],
                                                                      conf['permute'], conf['tok'])

        res_file = os.path.join(out_path, template_file_name)
        with open(res_file, 'rb') as f:
            res = pickle.load(f)
        results[use_case] = res

    if len(results) > 0:
        if len(results) == 1:
            results = list(results.values())[0]

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

                for result_id in result_ids:  # aggregate the result ids
                    res = cat_res.get_result(result_id)
                    target_res[agg_fns[idx]] = agg_fn(res).reshape((-1, 1))

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
                for result_id in current_cat_res:
                    current_cat_res[result_id] = np.concatenate([current_cat_res[result_id],
                                                                 use_case_res[result_id]], axis=1)

        agg_results[cat] = current_cat_res

    return agg_results


def cmp_agg_results(res1, res2):
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
        for metric in cat_res2:
            out_cat_res[metric] -= cat_res2[metric]
        cmp_res[cat] = out_cat_res

    return cmp_res


def use_case_analysis(conf: dict, plot_params: list, categories: list):

    assert isinstance(conf, dict), "Wrong data type for parameter 'conf'."
    assert isinstance(plot_params, list), "Wrong data type for parameter 'plot_params'."
    assert len(plot_params) > 0, "Wrong value for parameter 'plot_params'."
    assert isinstance(categories, list), "Wrong data type for parameter 'categories'."
    assert len(categories) > 0, "Wrong value for parameter 'categories'."

    res, tester = get_results(conf, [conf['use_case']])

    plot_results(res, tester, target_cats=categories, plot_params=plot_params)


def use_case_comparison_analysis(confs: list, plot_params: list, categories: list):
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

            cmp_res = cmp_results(res1, res2)
            plot_comparison(res1, res2, cmp_res, tester, cmp_vals, target_cats=categories, plot_params=plot_params)


def benchmark_analysis(tuned_res_flag: bool, pretrain_res_flag: bool, exp_confs: list, categories: list,
                       agg_fns: list = None, target_agg_result_ids: list = None):
    assert isinstance(tuned_res_flag, bool)
    assert isinstance(pretrain_res_flag, bool)
    assert isinstance(exp_confs, list)
    assert len(exp_confs) > 0
    for conf in exp_confs:
        assert isinstance(conf, dict)
    assert isinstance(categories, list)
    assert len(categories) > 0
    for c in categories:
        assert isinstance(c, str)
    if agg_fns is not None:
        assert isinstance(agg_fns, list)
        assert len(agg_fns) > 0
        for f in agg_fns:
            assert isinstance(f, str)
    if target_agg_result_ids is not None:
        assert isinstance(target_agg_result_ids, list)
        assert len(target_agg_result_ids) > 0
        for r in target_agg_result_ids:
            assert isinstance(r, str)

    for exp_conf in exp_confs:

        print(exp_conf)
        results, pre_results, tester = get_benchmark_results(exp_conf, USE_CASES,
                                                             tuned_res_flag,
                                                             pretrain_res_flag)

        if agg_fns is not None:
            results = aggregate_results(results, agg_fns, target_agg_result_ids)
            pre_results = aggregate_results(pre_results, agg_fns, target_agg_result_ids)

            if tuned_res_flag:
                print()
                print("TUNED")
                plot_agg_results(results, target_cats=categories, xticks=USE_CASES, agg=True)

            if pretrain_res_flag:
                print()
                print("PRETRAIN")
                plot_agg_results(pre_results, target_cats=categories, xticks=USE_CASES,
                                 agg=True)

            if tuned_res_flag and pretrain_res_flag:
                cmp_res = cmp_agg_results(results, pre_results)
                plot_agg_results(cmp_res, target_cats=categories, xticks=USE_CASES,
                                 vmin=-0.5, vmax=0.5)

        else:

            if tuned_res_flag:
                print()
                print("TUNED")
                plot_benchmark_results(results, tester, USE_CASES, categories)

            if pretrain_res_flag:
                print()
                print("PRETRAIN")
                plot_benchmark_results(pre_results, tester, USE_CASES, categories)

            if tuned_res_flag and pretrain_res_flag:
                cmp_res = cmp_benchmark_results(results, pre_results)
                plot_benchmark_results(cmp_res, tester, USE_CASES, categories, vmin=-0.5,
                                       vmax=0.5)


if __name__ == '__main__':

    analysis_target = 'use_case'
    # analysis_target = 'benchmark'

    # analysis_type = 'simple'
    analysis_type = 'comparison'

    conf = {
        'use_case': "Structured_Beer",
        'data_type': 'train',
        'permute': False,
        'model_name': 'bert-base-uncased',
        'tok': 'sent_pair',
        'size': None,
        'fine_tune_method': 'simple',
        'extractor': 'attr_extractor',
        'tester': 'attr_tester',
    }

    # plot_params = ['match_attr_attn_loc', 'match_attr_attn_over_mean',
    #                'avg_attr_attn', 'attr_attn_3_last', 'attr_attn_last_1',
    #                'attr_attn_last_2', 'attr_attn_last_3',
    #                'avg_attr_attn_3_last', 'avg_attr_attn_last_1',
    #                'avg_attr_attn_last_2', 'avg_attr_attn_last_3']
    plot_params = ['match_attr_attn_loc', 'match_attr_attn_over_mean',
                   'avg_attr_attn', 'attr_attn_last_1',
                   'attr_attn_last_2', 'attr_attn_last_3']

    categories = ['all']

    conf_creator = ConfCreator()
    conf_creator.validate_conf(conf)

    if analysis_target == 'use_case':

        if analysis_type == 'simple':
            use_case_analysis(conf, plot_params, categories)

        elif analysis_type == 'comparison':
            comparison_params = ['tok']
            # comparison_params = ['fine_tune_method']
            confs = conf_creator.get_confs(conf, comparison_params)
            use_case_comparison_analysis(confs, plot_params, categories)

        else:
            raise NotImplementedError()

    elif analysis_target == 'benchmark':

        if analysis_type == 'simple':
            pass
        elif analysis_type == 'comparison':
            pass
        else:
            raise NotImplementedError()

    else:
        raise NotImplementedError()




    # if analysis == 'use_case':
    #
    #     use_case = "Structured_Fodors-Zagats"
    #     tuned_res_flag = True
    #     pretrain_res_flag = False
    #     plot_params = ['match_attr_attn_loc', 'match_attr_attn_over_mean',
    #                    'avg_attr_attn', 'attr_attn_3_last', 'attr_attn_last_1',
    #                    'attr_attn_last_2', 'attr_attn_last_3',
    #                    'avg_attr_attn_3_last', 'avg_attr_attn_last_1',
    #                    'avg_attr_attn_last_2', 'avg_attr_attn_last_3']
    #     tok = 'sent_pair'
    #     exp_confs = [
    #         {'tokenization': tok, 'permutation': False, 'data_type': 'train', 'analysis_subject': 'attr',
    #          'test_types': ['attr_attn']},
    #         # {'tokenization': tok, 'permutation': False, 'data_type': 'test', 'analysis_subject': 'attr',
    #         #  'test_types': ['attr_attn']},
    #         # {'tokenization': tok, 'permutation': True, 'data_type': 'train', 'analysis_subject': 'attr',
    #         #  'test_types': ['attr_attn']},
    #         # {'tokenization': tok, 'permutation': True, 'data_type': 'test', 'analysis_subject': 'attr',
    #         #  'test_types': ['attr_attn']}
    #     ]
    #     categories = ['all']
    #
    #     use_case_analysis(use_case, tuned_res_flag, pretrain_res_flag, plot_params, exp_confs, categories)
    #
    # elif analysis == 'benchmark':
    #
    #     tuned_res_flag = True
    #     pretrain_res_flag = False
    #     agg_fns = None
    #     target_agg_result_ids = None
    #     # aggregation
    #     # agg_fns = ['row_mean', 'row_std']
    #     # target_agg_result_ids = ['match_attr_attn_loc']
    #     tok = 'sent_pair'
    #     exp_confs = [
    #         {'tokenization': tok, 'permutation': False, 'data_type': 'train', 'analysis_subject': 'attr',
    #          'test_types': ['attr_attn']},
    #         # {'tokenization': tok, 'permutation': False, 'data_type': 'test', 'analysis_subject': 'attr',
    #         #  'test_types': ['attr_attn']},
    #         # {'tokenization': tok, 'permutation': True, 'data_type': 'train', 'analysis_subject': 'attr',
    #         #  'test_types': ['attr_attn']},
    #         # {'tokenization': tok, 'permutation': True, 'data_type': 'test', 'analysis_subject': 'attr',
    #         #  'test_types': ['attr_attn']}
    #     ]
    #
    #     categories = ['all']
    #
    #     benchmark_analysis(tuned_res_flag, pretrain_res_flag, exp_confs, categories, agg_fns, target_agg_result_ids)
