import os
from pathlib import Path

from utils.result_collector import TestResultCollector
from utils.plot import plot_layers_heads_attention
from utils.general import get_pipeline


PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'analysis')


def run_inspection(conf: dict, inspect_row_idx: int, save: bool):
    assert isinstance(conf, dict), "Wrong data type for parameter 'conf'."
    assert isinstance(inspect_row_idx, int), "Wrong data type for parameter 'inspect_row_idx'."
    assert isinstance(save, bool), "Wrong data type for parameter 'save'."

    use_case = conf['use_case']
    out_path = os.path.join(RESULTS_DIR, use_case, "inspection")
    Path(out_path).mkdir(parents=True, exist_ok=True)

    template_file_name = '{}_{}_{}_{}_{}_{}_{}'.format(use_case, conf['data_type'], conf['extractor'],
                                                       conf['tester'], conf['fine_tune_method'],
                                                       conf['permute'], conf['tok'])

    extractors, testers, analyzers = get_pipeline(conf)
    attn_extractor = extractors[0]
    analyzer = analyzers[0]

    _, _, inspect_row_attns = attn_extractor.extract(inspect_row_idx)
    inspect_row_attns = inspect_row_attns['attns']

    inspect_row_results, label, pred, category, text_units = analyzer.analyze(inspect_row_idx)
    print("LABEL: {}".format(label))
    print("PRED: {}".format(pred))

    template_file_name += "_{}_{}".format(label, pred)

    params_to_inspect = {
        'attr_tester': ['match_attr_attn_loc'],
    }
    original_testers = [conf['tester']]
    for idx, tester in enumerate(testers):

        tester_name = original_testers[idx]
        test_params_to_inspect = params_to_inspect[tester_name]
        inspect_row_test_results = inspect_row_results[idx]

        if save:
            out_dir = out_path
            out_file_name_prefix = template_file_name
        else:
            out_dir = None
            out_file_name_prefix = None
        tester.plot(inspect_row_test_results, out_dir=out_dir, out_file_name_prefix=out_file_name_prefix)

        if isinstance(inspect_row_test_results, dict):
            for key in inspect_row_test_results:
                print(key)
                test_collector = inspect_row_test_results[key]
                res = test_collector.get_results()
                for param in test_params_to_inspect:
                    mask = res[param] > 0
                    print(param)

                    if save:
                        out_file_name = os.path.join(out_path, '{}_{}'.format(template_file_name, "lxh_attns.pdf"))
                    else:
                        out_file_name = None
                    plot_layers_heads_attention(inspect_row_attns, mask=mask, out_file_name=out_file_name)

        elif isinstance(inspect_row_test_results, TestResultCollector):
            res = inspect_row_test_results.get_results()
            for param in test_params_to_inspect:
                print(param)
                mask = res[param] > 0

                if save:
                    out_file_name = os.path.join(out_path, '{}_{}'.format(template_file_name, "lxh_attns.pdf"))
                else:
                    out_file_name = None
                plot_layers_heads_attention(inspect_row_attns, mask=mask, out_file_name=out_file_name)


if __name__ == '__main__':
    conf = {
        'use_case': "Structured_Fodors-Zagats",
        'data_type': 'train',  # 'train', 'test', 'valid'
        'permute': False,
        'model_name': 'bert-base-uncased',
        'tok': 'attr_pair',  # 'sent_pair', 'attr', 'attr_pair'
        'label_col': 'label',
        'left_prefix': 'left_',
        'right_prefix': 'right_',
        'max_len': 128,
        'verbose': False,
        'size': None,
        'target_class': 1,  # 'both', 0, 1
        'fine_tune_method': 'advanced',  # None, 'simple', 'advanced'
        'extractor': 'attr_extractor',
        'tester': 'attr_tester',
        'seeds': [42, 42]
    }

    save = True
    inspect_row_idx = 0

    run_inspection(conf, inspect_row_idx, save)
