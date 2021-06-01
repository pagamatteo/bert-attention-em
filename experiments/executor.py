import os
import pickle
import copy
import itertools
from pathlib import Path

from utils.general import get_pipeline


PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'analysis')


def run(confs: list, num_attempts: int, save: bool):
    assert isinstance(confs, list), "Wrong data type for parameter 'confs'."
    assert len(confs) > 0, "Empty configuration list."
    assert isinstance(num_attempts, int), "Wrong data type for parameter 'num_attempts'."
    assert num_attempts > 0, "Wrong value for parameter 'num_attempts'."
    assert isinstance(save, bool), "Wrong data type for parameter 'save'."

    confs_by_use_case = {}
    for conf in confs:
        use_case = conf['use_case']
        if use_case not in confs_by_use_case:
            confs_by_use_case[use_case] = [conf]
        else:
            confs_by_use_case[use_case].append(conf)

    for use_case in confs_by_use_case:
        print(use_case)
        use_case_confs = confs_by_use_case[use_case]

        out_path = os.path.join(RESULTS_DIR, use_case)
        Path(out_path).mkdir(parents=True, exist_ok=True)

        all_results = {}
        all_results_counts = {}
        for idx, conf in enumerate(use_case_confs):

            print(conf)
            _, _, analyzers = get_pipeline(conf)
            analyzer = analyzers[0]

            template_file_name = '{}_{}_{}_{}_{}_{}_{}_{}'.format(use_case, conf['data_type'], conf['extractor'],
                                                                  conf['tester'], conf['fine_tune_method'],
                                                                  conf['permute'], conf['tok'], idx % num_attempts)

            res_out_file_name = os.path.join(out_path, '{}.pickle'.format(template_file_name))

            # run the tests
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

            # save the results into file
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

        # average the results
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
                out_avg_file = os.path.join(out_path, '{}_AVG.pickle'.format(res_key))
                with open(out_avg_file, 'wb') as f:
                    pickle.dump(avg_res, f)


if __name__ == '__main__':

    fixed_params = {
        'label_col': 'label',
        'left_prefix': 'left_',
        'right_prefix': 'right_',
        'max_len': 128,
        'verbose': False,
    }

    variable_params = {
        'use_case': ["Structured_Fodors-Zagats"],
        'data_type': ['train', 'test'],
        'permute': [False, True],
        'model_name': ['bert-base-uncased'],
        'tok': ['sent_pair'],  # 'sent_pair', 'attr', 'attr_pair'
        'size': [None],
        'target_class': ['both'],  # 'both', 0, 1
        'fine_tune_method': [None],  # None, 'simple', 'advanced'
        'extractor': ['attr_extractor'],
        'tester': ['attr_tester'],
        'seeds': [[42, 42], [42, 24], [42, 12]]
    }

    confs_vals = list(itertools.product(*variable_params.values()))
    confs = [{key: val for (key, val) in zip(list(variable_params), vals)} for vals in confs_vals]
    for conf in confs:
        conf.update(fixed_params)

    save = True
    num_attempts = len(variable_params['seeds'])

    run(confs, num_attempts, save)
