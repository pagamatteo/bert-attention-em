import os
import pickle
from transformers import AutoModelForSequenceClassification
import pandas as pd
import copy

from utils.data_collector import DataCollector
from utils.result_collector import TestResultCollector
from models.em_dataset import EMDataset
from utils.data_selection import Sampler
from attention.extractors import AttributeAttentionExtractor
from attention.testers import GenericAttributeAttentionTest
from attention.analyzers import AttentionMapAnalyzer
from utils.plot import plot_layers_heads_attention

PROJECT_DIR = os.path.abspath('..')
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'analysis')


def init_test(exp_conf: dict, use_case_data_dir: str, model_name: str, model, label_col: str, left_prefix: str, right_prefix: str, max_len: int):

    assert isinstance(exp_conf, dict)
    params = ['permutation', 'data_type', 'analysis_subject', 'test_types', 'samples_seeds', 'tokenization']
    assert all([p in exp_conf for p in params])
    assert isinstance(use_case_data_dir, str)
    assert isinstance(model_name, str)
    assert isinstance(label_col, str)
    assert isinstance(left_prefix, str)
    assert isinstance(right_prefix, str)
    assert isinstance(max_len, int)

    permutation = exp_conf['permutation']
    data_type = exp_conf['data_type']
    analysis_subject = exp_conf['analysis_subject']
    test_types = exp_conf['test_types']
    samples_seeds = exp_conf['samples_seeds']
    tokenization = exp_conf['tokenization']

    # STEP 1: data selection
    if data_type == 'train':
        dataset_path = os.path.join(use_case_data_dir, "train.csv")
    elif data_type == 'test':
        dataset_path = os.path.join(use_case_data_dir, "test.csv")
    else:
        dataset_path = os.path.join(use_case_data_dir, "valid.csv")

    data = pd.read_csv(dataset_path)
    dataset = EMDataset(data, model_name, tokenization=tokenization, label_col=label_col, left_prefix=left_prefix,
                        right_prefix=right_prefix, max_len=max_len)

    sampler = Sampler(dataset, permute=permutation)
    sample = sampler.get_balanced_data(seeds=samples_seeds)

    # STEP 2: attention extractor selection
    attn_extractor = None
    if analysis_subject == 'attr':
        attn_extractor = AttributeAttentionExtractor(sample, model)

    assert attn_extractor is not None

    # STEP 3: test selection
    testers = []
    for test_type in test_types:

        if 'attr_attn' in test_types:

            generic_attr_tester = GenericAttributeAttentionTest(permute=permutation)
            testers.append(generic_attr_tester)

        else:
            print(f"Test type {test_type} not found.")

    assert len(testers) > 0

    analyzer = AttentionMapAnalyzer(attn_extractor, testers)

    return attn_extractor, {k: v for (k, v) in zip(test_types, testers)}, analyzer


def run_experiments(use_case: str, model_path: str, tok: str, save: bool, pretrain: bool, model_name: str = 'bert-base-uncased', label_col: str = 'label',
         left_prefix: str = 'left_', right_prefix: str = 'right_', max_len: int = 128):

    # STEP 0: download the data
    data_collector = DataCollector()
    use_case_data_dir = data_collector.get_data(use_case)

    model = None
    if pretrain:
        pass
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

    exp_confs = [
        {'id': 0, 'tokenization': tok, 'permutation': False, 'data_type': 'train', 'analysis_subject': 'attr',
         'test_types': ['attr_attn'], 'samples_seeds': [42, 42]},
        {'id': 1, 'tokenization': tok, 'permutation': False, 'data_type': 'train', 'analysis_subject': 'attr',
         'test_types': ['attr_attn'], 'samples_seeds': [42, 24]},
        {'id': 2, 'tokenization': tok, 'permutation': False, 'data_type': 'train', 'analysis_subject': 'attr',
         'test_types': ['attr_attn'], 'samples_seeds': [42, 12]},
        {'id': 0, 'tokenization': tok, 'permutation': False, 'data_type': 'test', 'analysis_subject': 'attr',
         'test_types': ['attr_attn'], 'samples_seeds': [42, 42]},
        {'id': 1, 'tokenization': tok, 'permutation': False, 'data_type': 'test', 'analysis_subject': 'attr',
         'test_types': ['attr_attn'], 'samples_seeds': [42, 24]},
        {'id': 2, 'tokenization': tok, 'permutation': False, 'data_type': 'test', 'analysis_subject': 'attr',
         'test_types': ['attr_attn'], 'samples_seeds': [42, 12]},
        {'id': 0, 'tokenization': tok, 'permutation': True, 'data_type': 'train', 'analysis_subject': 'attr',
         'test_types': ['attr_attn'], 'samples_seeds': [42, 42]},
        {'id': 1, 'tokenization': tok, 'permutation': True, 'data_type': 'train', 'analysis_subject': 'attr',
         'test_types': ['attr_attn'], 'samples_seeds': [42, 24]},
        {'id': 2, 'tokenization': tok, 'permutation': True, 'data_type': 'train', 'analysis_subject': 'attr',
         'test_types': ['attr_attn'], 'samples_seeds': [42, 12]},
        {'id': 0, 'tokenization': tok, 'permutation': True, 'data_type': 'test', 'analysis_subject': 'attr',
         'test_types': ['attr_attn'], 'samples_seeds': [42, 42]},
        {'id': 1, 'tokenization': tok, 'permutation': True, 'data_type': 'test', 'analysis_subject': 'attr',
         'test_types': ['attr_attn'], 'samples_seeds': [42, 24]},
        {'id': 2, 'tokenization': tok, 'permutation': True, 'data_type': 'test', 'analysis_subject': 'attr',
         'test_types': ['attr_attn'], 'samples_seeds': [42, 12]}
    ]

    all_results = {}
    all_results_counts = {}
    for exp_conf in exp_confs:

        print(exp_conf)

        _, _, analyzer = init_test(exp_conf, use_case_data_dir, model_name, model, label_col, left_prefix, right_prefix, max_len)

        if pretrain:
            template_file_name = '{}_PRETRAIN_{}_{}_{}_{}_{}'.format(use_case, exp_conf['data_type'], exp_conf['analysis_subject'], exp_conf['permutation'],
                                                                     exp_conf['tokenization'], exp_conf['exp_id'])
        else:
            template_file_name = '{}_{}_{}_{}_{}_{}'.format(use_case, exp_conf['data_type'], exp_conf['analysis_subject'], exp_conf['permutation'],
                                                            exp_conf['tokenization'], exp_conf['exp_id'])

        res_out_file_name = os.path.join(RESULTS_DIR, '{}.pickle'.format(template_file_name))

        # STEP 4: run the tests
        testers_res = analyzer.analyze_all()

        # append the current run results to the total results
        res_key = '_'.join(template_file_name.split('_')[:-1])
        if res_key not in all_results:
            res_copy = {}
            res = testers_res[0]
            all_res_cat_counts = {}
            for cat, cat_res in res.items():
                res_copy[cat] = copy.deepcopy(cat_res)
                if cat_res is not None:
                    all_res_cat_counts[cat] = 1
                else:
                    all_res_cat_counts[cat] = 0
            all_results[res_key] = res_copy
            all_results_counts[res_key] = all_res_cat_counts
        else:
            res = testers_res[0]
            for cat, cat_res in res.items():
                if cat_res is not None:
                    if all_results[res_key][cat] is not None:
                        all_results[res_key][cat].add_collector(cat_res)
                    else:
                        all_results[res_key][cat] = copy.deepcopy(cat_res)
                    all_results_counts[res_key][cat] += 1

        # STEP 5: save results into file
        if save:
            with open(res_out_file_name, 'wb') as f:
                pickle.dump(testers_res, f)

            # # save some stats
            # size = len(sample)

            # y_true, y_pred = analyzer.get_labels_and_preds()
            # f1 = f1_score(y_true, y_pred)
            # print("F1 Match class: {}".format(f1))

            # discarded_rows = attn_extractor.get_num_invalid_attr_attn_maps()
            # print("Num. discarded rows: {}".format(discarded_rows))

            # df = pd.DataFrame([{'size': size, 'f1': f1, 'skip': discarded_rows, 'data_type': data_type}])
            # df.to_csv(os.path.join(drive_results_out_dir, "stats_{}.csv".format(template_file_name)), index=False)

    # average results
    avg_results = {}
    for res_key in all_results:

        all_res = all_results[res_key]

        avg_res = {}
        for cat, all_cat_res in all_res.items():

            if all_cat_res is not None:
                assert all_results_counts[res_key][cat] > 0
                all_cat_res.transform_all(lambda x: x / all_results_counts[res_key][cat])
                avg_res[cat] = copy.deepcopy(all_cat_res)

        avg_results[res_key] = avg_res

        if save:
            out_avg_file = os.path.join(RESULTS_DIR, '{}_AVG.pickle'.format(res_key))
            with open(out_avg_file, 'wb') as f:
                pickle.dump(avg_res, f)


def run_inspection(use_case: str, model_path: str, conf: dict, save: bool, pretrain: bool, model_name: str = 'bert-base-uncased', label_col: str = 'label',
         left_prefix: str = 'left_', right_prefix: str = 'right_', max_len: int = 128):

    # STEP 0: download the data
    data_collector = DataCollector()
    use_case_data_dir = data_collector.get_data(use_case)

    model = None
    if pretrain:
        pass
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

    attn_extractor, testers, analyzer = init_test(conf, use_case_data_dir, model_name, model, label_col, left_prefix, right_prefix, max_len)

    inspect_row_idx = 0

    _, _, inspect_row_attns = attn_extractor.extract(inspect_row_idx)
    inspect_row_attns = inspect_row_attns['attns']

    inspect_row_results, label, pred, category, text_units = analyzer.analyze(inspect_row_idx)
    print("LABEL: {}".format(label))
    print("PRED: {}".format(pred))
    print("TEXT UNITS: {}".format(text_units))

    params_to_inspect = {
        'attr_attn': ['match_attr_attn_loc'],
    }
    for idx, test_type in enumerate(testers):

        print(test_type)
        test_params_to_inspect = params_to_inspect[test_type]
        inspect_row_test_results = inspect_row_results[idx]

        testers[test_type][idx].plot(inspect_row_test_results)

        if isinstance(inspect_row_test_results, dict):
            for key in inspect_row_test_results:
                print(key)
                test_collector = inspect_row_test_results[key]
                res = test_collector.get_results()
                for param in test_params_to_inspect:
                    mask = res[param] > 0
                    print(param)
                    plot_layers_heads_attention(inspect_row_attns, mask=mask)

        elif isinstance(inspect_row_test_results, TestResultCollector):
            res = inspect_row_test_results.get_results()
            for param in test_params_to_inspect:
                print(param)
                mask = res[param] > 0
                plot_layers_heads_attention(inspect_row_attns, mask=mask)


if __name__ == '__main__':

    test = 'all'
    # test = 'inspection'

    use_case = "Structured_Fodors-Zagats"
    # use_case = "Structured_DBLP-GoogleScholar"
    # use_case = "Structured_DBLP-ACM"
    # use_case = "Structured_Amazon-Google"
    # use_case = "Structured_Walmart-Amazon"
    # use_case = "Structured_Beer"
    # use_case = "Structured_iTunes-Amazon"
    # use_case = "Textual_Abt-Buy"
    # use_case = "Dirty_iTunes-Amazon"
    # use_case = "Dirty_DBLP-ACM"
    # use_case = "Dirty_DBLP-GoogleScholar"
    # use_case = "Dirty_Walmart-Amazon"

    model_path = os.path.join(MODELS_DIR, 'simple', f"{use_case}_tuned")
    tok = 'sent_pair'
    save = True
    pretrain = False

    if test == 'all':

        run_experiments(use_case, model_path, tok, save, pretrain)

    elif test == 'inspection':

        conf = {'id': 0, 'tokenization': tok, 'permutation': False, 'data_type': 'train', 'analysis_subject': 'attr',
                'test_types': ['attr_attn'], 'samples_seeds': [42, 42]}

        run_inspection(use_case, model_path, conf, save, pretrain)
