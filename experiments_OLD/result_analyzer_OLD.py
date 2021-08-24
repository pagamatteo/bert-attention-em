import os
import pickle
import copy
import numpy as np
from utils.result_collector import TestResultCollector
from core.attention.testers import GenericAttributeAttentionTest
from utils.plot import plot_results, plot_benchmark_results, plot_agg_results
from pathlib import Path


PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'analysis')


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


def get_benchmark_results(exp_conf: dict, use_cases: list, tuned_res_flag: bool, pretrain_res_flag: bool):
    assert isinstance(exp_conf, dict), "Wrong data type for parameter 'exp_conf'."
    params = ["tokenization", "permutation", "data_type", "analysis_subject", "test_types"]
    assert all([p in exp_conf for p in params]), "Wrong value for parameter 'exp_conf'."
    assert isinstance(use_cases, list), "Wrong data type for parameter 'use_cases'."
    assert len(use_cases) > 0, "Empty use case list."
    assert isinstance(tuned_res_flag, bool), "Wrong data type for parameter 'tuned_res_flag'."
    assert isinstance(pretrain_res_flag, bool), "Wrong data type for parameter 'pretrain_res_flag'."

    tokenization = exp_conf['tokenization']
    permutation = exp_conf['permutation']
    data_type = exp_conf['data_type']
    analysis_subject = exp_conf['analysis_subject']
    test_types = exp_conf['test_types']

    testers = []
    for test_type in test_types:

        if 'attr_attn' in test_types:

            tester = GenericAttributeAttentionTest(permute=permutation)
            testers.append(tester)

        else:
            print(f"Test type {test_type} not found.")

    assert len(testers) == 1
    tester = testers[0]

    results = {}
    pre_results = {}
    for use_case in use_cases:

        new_use_case = use_case.replace("/", "_")

        if tuned_res_flag:
            template = '{}_{}_{}_{}_{}_AVG'.format(new_use_case, data_type,
                                                   analysis_subject, permutation,
                                                   tokenization)
            res_file = os.path.join(RESULTS_DIR, new_use_case,
                                    '{}.pickle'.format(template))
            with open(res_file, 'rb') as f:
                res = pickle.load(f)
            results[use_case] = res

        if pretrain_res_flag:
            pre_template = '{}_PRETRAIN_{}_{}_{}_{}_AVG'.format(new_use_case,
                                                                data_type,
                                                                analysis_subject,
                                                                permutation,
                                                                tokenization)
            pre_res_file = os.path.join(RESULTS_DIR, new_use_case,
                                        '{}.pickle'.format(pre_template))
            with open(pre_res_file, 'rb') as f:
                pre_res = pickle.load(f)
            pre_results[use_case] = pre_res

    if len(results) > 0:
        if len(results) == 1:
            results = list(results.values())[0]

    if len(pre_results) > 0:
        if len(pre_results) == 1:
            pre_results = list(pre_results.values())[0]

    return results, pre_results, tester


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


def use_case_analysis(use_case: str, tuned_res_flag: bool, pretrain_res_flag: bool, plot_params: list, exp_confs: list,
                      categories: list):
    assert isinstance(use_case, str)
    assert isinstance(tuned_res_flag, bool)
    assert isinstance(pretrain_res_flag, bool)
    assert isinstance(plot_params, list)
    assert len(plot_params) > 0
    for p in plot_params:
        assert isinstance(p, str)
    assert isinstance(exp_confs, list)
    assert len(exp_confs) > 0
    for conf in exp_confs:
        assert isinstance(conf, dict)
    assert isinstance(categories, list)
    assert len(categories) > 0
    for c in categories:
        assert isinstance(c, str)

    for exp_conf in exp_confs:

        print(exp_conf)
        res, pre_res, tester = get_benchmark_results(exp_conf, [use_case],
                                                     tuned_res_flag,
                                                     pretrain_res_flag)

        if tuned_res_flag:
            print()
            print("TUNED")
            plot_results(res, tester, categories, plot_params=plot_params)

        if pretrain_res_flag:
            print()
            print("PRETRAIN")
            plot_results(pre_res, tester, categories, plot_params=plot_params)

        if tuned_res_flag and pretrain_res_flag:
            cmp_res = cmp_results(res, pre_res)
            plot_results(cmp_res, tester, categories, plot_params=plot_params)


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


USE_CASES = [
    "Structured_Fodors-Zagats",
    "Structured_DBLP-GoogleScholar",
    "Structured_DBLP-ACM", "Structured_Amazon-Google",
    "Structured_Walmart-Amazon", "Structured_Beer",
    "Structured_iTunes-Amazon",
    "Textual_Abt-Buy",
    "Dirty_iTunes-Amazon",
    "Dirty_DBLP-ACM",
    "Dirty_DBLP-GoogleScholar",
    "Dirty_Walmart-Amazon"
]

if __name__ == '__main__':

    analysis = 'use_case'
    # analysis = 'benchmark'

    if analysis == 'use_case':

        use_case = "Structured_Fodors-Zagats"
        tuned_res_flag = True
        pretrain_res_flag = False
        plot_params = ['match_attr_attn_loc', 'match_attr_attn_over_mean',
                       'avg_attr_attn', 'attr_attn_3_last', 'attr_attn_last_1',
                       'attr_attn_last_2', 'attr_attn_last_3',
                       'avg_attr_attn_3_last', 'avg_attr_attn_last_1',
                       'avg_attr_attn_last_2', 'avg_attr_attn_last_3']
        tok = 'sent_pair'
        exp_confs = [
            {'tokenization': tok, 'permutation': False, 'data_type': 'train', 'analysis_subject': 'attr',
             'test_types': ['attr_attn']},
            # {'tokenization': tok, 'permutation': False, 'data_type': 'test', 'analysis_subject': 'attr',
            #  'test_types': ['attr_attn']},
            # {'tokenization': tok, 'permutation': True, 'data_type': 'train', 'analysis_subject': 'attr',
            #  'test_types': ['attr_attn']},
            # {'tokenization': tok, 'permutation': True, 'data_type': 'test', 'analysis_subject': 'attr',
            #  'test_types': ['attr_attn']}
        ]
        categories = ['all']

        use_case_analysis(use_case, tuned_res_flag, pretrain_res_flag, plot_params, exp_confs, categories)

    elif analysis == 'benchmark':

        tuned_res_flag = True
        pretrain_res_flag = False
        agg_fns = None
        target_agg_result_ids = None
        # aggregation
        # agg_fns = ['row_mean', 'row_std']
        # target_agg_result_ids = ['match_attr_attn_loc']
        tok = 'sent_pair'
        exp_confs = [
            {'tokenization': tok, 'permutation': False, 'data_type': 'train', 'analysis_subject': 'attr',
             'test_types': ['attr_attn']},
            # {'tokenization': tok, 'permutation': False, 'data_type': 'test', 'analysis_subject': 'attr',
            #  'test_types': ['attr_attn']},
            # {'tokenization': tok, 'permutation': True, 'data_type': 'train', 'analysis_subject': 'attr',
            #  'test_types': ['attr_attn']},
            # {'tokenization': tok, 'permutation': True, 'data_type': 'test', 'analysis_subject': 'attr',
            #  'test_types': ['attr_attn']}
        ]

        categories = ['all']

        benchmark_analysis(tuned_res_flag, pretrain_res_flag, exp_confs, categories, agg_fns, target_agg_result_ids)
