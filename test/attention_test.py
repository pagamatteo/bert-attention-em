from utils.general import get_dataset, get_model, get_sample, get_extractors
import os
from pathlib import Path
from multiprocessing import Process
import pickle
from attention.extractors import AttributeAttentionExtractor, WordAttentionExtractor, AttentionExtractor
from attention.analyzers import AttrToClsAttentionAnalyzer, EntityToEntityAttentionAnalyzer
from utils.test_utils import ConfCreator
import pandas as pd


PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'attention')


def run_attn_extractor(conf: dict, sampler_conf: dict, fine_tune: str, attn_params: dict, models_dir: str,
                       res_dir: str):

    assert isinstance(attn_params, dict)
    assert 'attn_extractor' in attn_params
    assert 'attn_extr_params' in attn_params
    assert attn_params['attn_extractor'] in ['attr_extractor', 'word_extractor', 'token_extractor']
    assert isinstance(attn_params['attn_extr_params'], dict)

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

    extractor_name = attn_params['attn_extractor']
    extractor_params = {
        'dataset': sample,
        'model': model,
    }
    extractor_params.update(attn_params['attn_extr_params'])
    attn_extractors = get_extractors({extractor_name: extractor_params})
    attn_extractor = attn_extractors[0]

    results = attn_extractor.extract_all()

    out_fname = f"{use_case}_{tok}_{sampler_conf['size']}_{fine_tune}_{extractor_name}"
    out_file = os.path.join(res_dir, use_case, out_fname)
    out_dir_path = out_file.split(os.sep)
    out_dir = os.sep.join(out_dir_path[:-1])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f'{out_file}.pkl', 'wb') as f:
        pickle.dump(results, f)


def load_saved_attn_data(use_case, conf, sampler_conf, fine_tune, attn_params, res_dir):
    tok = conf['tok']
    size = sampler_conf['size']
    extractor_name = attn_params['attn_extractor']
    out_fname = f"{use_case}_{tok}_{size}_{fine_tune}_{extractor_name}"
    data_path = os.path.join(res_dir, use_case, out_fname)
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


if __name__ == '__main__':
    use_cases = ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"]
    # use_cases = ["Structured_Fodors-Zagats"]

    # [BEGIN] INPUT PARAMS ---------------------------------------------------------------------------------------------
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

    attn_params = {
        'attn_extractor': 'token_extractor',     # 'attr_extractor', 'word_extractor', 'token_extractor'
        'attn_extr_params': {'special_tokens': True},
    }

    if attn_params['attn_extractor'] == 'word_extractor' and conf['tok'] == 'attr_pair':
        raise NotImplementedError()

    # experiment = 'compute_attn', 'attr_to_cls', 'attr_to_cls_entropy', 'word_to_cls', 'entity_to_entity'
    experiment = 'entity_to_entity'
    # [END] INPUT PARAMS -----------------------------------------------------------------------------------------------

    if experiment == 'compute_attn':
        # # no multi process
        # for use_case in use_cases:
        #     print(use_case)
        #
        #     uc_conf = conf.copy()
        #     uc_conf['use_case'] = use_case
        #     run_attn_extractor(uc_conf, sampler_conf, fine_tune, attn_params, MODELS_DIR, RESULTS_DIR)

        processes = []
        for use_case in use_cases:
            uc_conf = conf.copy()
            uc_conf['use_case'] = use_case
            p = Process(target=run_attn_extractor,
                        args=(uc_conf, sampler_conf, fine_tune, attn_params, MODELS_DIR, RESULTS_DIR,))
            processes.append(p)

        for p in processes:
            p.start()

        for p in processes:
            p.join()

    elif experiment in ['attr_to_cls', 'attr_to_cls_entropy']:

        assert attn_params['attn_extractor'] == 'attr_extractor'

        out_fname = f"{conf['tok']}_{sampler_conf['size']}_{fine_tune}_{attn_params['attn_extractor']}_{experiment}.pdf"
        out_file = os.path.join(RESULTS_DIR, out_fname)

        uc_grouped_attn = {}
        for use_case in use_cases:
            uc_attn = load_saved_attn_data(use_case, conf, sampler_conf, fine_tune, attn_params, RESULTS_DIR)

            target_categories = ['all', 'all_pred_pos', 'all_pred_neg']
            if experiment == 'attr_to_cls':
                grouped_attn_res = AttrToClsAttentionAnalyzer.group_or_aggregate(uc_attn,
                                                                                 target_categories=target_categories)
            else:
                grouped_attn_res = AttrToClsAttentionAnalyzer.group_or_aggregate(uc_attn,
                                                                                 target_categories=target_categories,
                                                                                 agg_metric='mean')
            uc_grouped_attn[use_case] = grouped_attn_res

        if experiment == 'attr_to_cls':
            AttrToClsAttentionAnalyzer.plot_multi_attr_to_cls_attn(uc_grouped_attn, save_path=out_file)
        else:
            entropy_res = AttrToClsAttentionAnalyzer.analyze_multi_results(uc_grouped_attn, analysis_type='entropy')
            entropy_res = entropy_res.rename(index=ConfCreator().use_case_map)
            AttrToClsAttentionAnalyzer.plot_attr_to_cls_attn_entropy(entropy_res, save_path=out_file)

    elif experiment == 'word_to_cls':
        assert attn_params['attn_extractor'] == 'word_extractor'

        out_fname = f"{conf['tok']}_{sampler_conf['size']}_{fine_tune}_{attn_params['attn_extractor']}_{experiment}.pdf"
        out_file = os.path.join(RESULTS_DIR, out_fname)

        uc_grouped_attn = {}
        for use_case in use_cases:
            uc_attn = load_saved_attn_data(use_case, conf, sampler_conf, fine_tune, attn_params, RESULTS_DIR)

    elif experiment == 'entity_to_entity':

        assert attn_params['attn_extr_params']['special_tokens'] is True
        tok = conf['tok']
        text_unit = attn_params['attn_extractor'].split('_')[0]
        out_fname = f"{conf['tok']}_{sampler_conf['size']}_{fine_tune}_{attn_params['attn_extractor']}_{experiment}.pdf"
        out_file = os.path.join(RESULTS_DIR, out_fname)

        uc_e2e_res = {}
        for use_case in use_cases:
            uc_attn = load_saved_attn_data(use_case, conf, sampler_conf, fine_tune, attn_params, RESULTS_DIR)
            analyzer = EntityToEntityAttentionAnalyzer(uc_attn, text_unit=text_unit, tokenization=tok)
            # same_e2e_res = analyzer.analyze(analysis_type='same_entity', ignore_special=True,
            #                                 target_categories=['all_pred_pos', 'all_pred_neg'])
            cross_e2e_res = analyzer.analyze(analysis_type='cross_entity', ignore_special=True,
                                             target_categories=['all_pred_pos', 'all_pred_neg'])
            #
            # merged_e2e_res = {}
            # cat_map = {'all': 'all', 'all_pred_pos': 'match', 'all_pred_neg': 'non_match'}
            # for cat in same_e2e_res:
            #     merged_e2e_res[f'same_{cat_map[cat]}'] = same_e2e_res[cat]
            # for cat in cross_e2e_res:
            #     merged_e2e_res[f'cross_{cat_map[cat]}'] = cross_e2e_res[cat]

            uc_e2e_res[use_case] = cross_e2e_res

        EntityToEntityAttentionAnalyzer.plot_multi_entity_to_entity_attn(uc_e2e_res, save_path=out_file)

    else:
        raise NotImplementedError()
