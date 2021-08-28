from utils.general import get_dataset, get_model, get_sample, get_extractors
import os
from pathlib import Path
from multiprocessing import Process
import pickle
from core.attention.extractors import AttributeAttentionExtractor, WordAttentionExtractor, AttentionExtractor
from core.attention.analyzers import AttrToClsAttentionAnalyzer, EntityToEntityAttentionAnalyzer
from utils.test_utils import ConfCreator


PROJECT_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'attention')


def get_attn_extractor(conf: dict, sampler_conf: dict, fine_tune: str, attn_params: dict, models_dir: str):

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

    return attn_extractor


def run_attn_extractor(conf: dict, sampler_conf: dict, fine_tune: str, attn_params: dict, models_dir: str,
                       res_dir: str):

    use_case = conf['use_case']
    tok = conf['tok']
    extractor_name = attn_params['attn_extractor']

    attn_extractor = get_attn_extractor(conf, sampler_conf, fine_tune, attn_params, models_dir)
    results = attn_extractor.extract_all()

    params = '_'.join([f'{x[0]}={x[1]}' for x in attn_params['attn_extr_params'].items()])
    out_fname = f"ATTN_{use_case}_{tok}_{sampler_conf['size']}_{fine_tune}_{extractor_name}_{params}"
    out_file = os.path.join(res_dir, use_case, out_fname)
    out_dir_path = out_file.split(os.sep)
    out_dir = os.sep.join(out_dir_path[:-1])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f'{out_file}.pkl', 'wb') as f:
        pickle.dump(results, f)

    return results


def load_saved_attn_data(use_case, conf, sampler_conf, fine_tune, attn_params, res_dir):
    tok = conf['tok']
    size = sampler_conf['size']
    extractor_name = attn_params['attn_extractor']
    params = '_'.join([f'{x[0]}={x[1]}' for x in attn_params['attn_extr_params'].items()])
    out_fname = f"ATTN_{use_case}_{tok}_{size}_{fine_tune}_{extractor_name}_{params}"
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


def compute_entity_to_entity_attention(use_cases, conf, sampler_conf, fine_tune, attn_params, res_dir,
                                       analysis_type='cross_entity', ignore_special=True, target_categories=None,
                                       extract_attention=False, precomputed=False, save=False):
    text_unit = attn_params['attn_extractor'].split('_')[0]
    tok = conf['tok']
    assert analysis_type in ['cross_entity', 'same_entity']

    uc_e2e_res = {}
    for use_case in use_cases:
        print(use_case)
        uc_e2e_res_path = f"E2E_{use_case}_{tok}_{sampler_conf['size']}_{fine_tune}_{attn_params['attn_extractor']}.pkl"
        uc_e2e_res_path = os.path.join(res_dir, use_case, uc_e2e_res_path)

        force_computation = not precomputed
        if precomputed:
            try:
                e2e_res = pickle.load(open(uc_e2e_res_path, "rb"))
            except Exception:
                print(f"No precomputed result found in {uc_e2e_res_path}.")
                force_computation = True

        if not precomputed or force_computation is True:
            if not extract_attention:
                uc_attn = load_saved_attn_data(use_case, conf, sampler_conf, fine_tune, attn_params, res_dir)
            else:
                uc_conf = conf.copy()
                uc_conf['use_case'] = use_case
                uc_attn = get_attn_extractor(uc_conf, sampler_conf, fine_tune, attn_params, MODELS_DIR)

            analyzer = EntityToEntityAttentionAnalyzer(uc_attn, text_unit=text_unit, tokenization=tok,
                                                       analysis_type=analysis_type, ignore_special=ignore_special,
                                                       target_categories=target_categories)
            e2e_res = analyzer.analyze_all()

            # free some memory
            uc_attn = None
            analyzer = None

            # merged_e2e_res = {}
            # cat_map = {'all': 'all', 'all_pred_pos': 'match', 'all_pred_neg': 'non_match'}
            # for cat in same_e2e_res:
            #     merged_e2e_res[f'same_{cat_map[cat]}'] = same_e2e_res[cat]
            # for cat in cross_e2e_res:
            #     merged_e2e_res[f'cross_{cat_map[cat]}'] = cross_e2e_res[cat]

        uc_e2e_res[use_case] = e2e_res

        if save:
            Path(os.path.join(res_dir, use_case)).mkdir(parents=True, exist_ok=True)
            with open(uc_e2e_res_path, 'wb') as f:
                pickle.dump(e2e_res, f)

    return uc_e2e_res


def compute_attn_to_cls(use_cases, conf, sampler_conf, fine_tune, attn_params, res_dir, attention_type,
                        target_categories):
    uc_grouped_attn = {}
    for use_case in use_cases:
        uc_attn = load_saved_attn_data(use_case, conf, sampler_conf, fine_tune, attn_params, res_dir)

        if attention_type == 'attr_to_cls':
            grouped_attn_res = AttrToClsAttentionAnalyzer.group_or_aggregate(uc_attn,
                                                                             target_categories=target_categories)
        else:
            grouped_attn_res = AttrToClsAttentionAnalyzer.group_or_aggregate(uc_attn,
                                                                             target_categories=target_categories,
                                                                             agg_metric='mean')
        uc_grouped_attn[use_case] = grouped_attn_res

    return uc_grouped_attn


if __name__ == '__main__':
    use_cases = ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"]
    use_cases = ["Structured_Fodors-Zagats", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon",
                 "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy",
                 "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_Walmart-Amazon"]
    # use_cases = ["Structured_DBLP-GoogleScholar",
    #              #"Structured_DBLP-ACM",
    #              "Dirty_DBLP-GoogleScholar",]
    #              #"Dirty_DBLP-ACM"]
    # use_cases = ["Structured_DBLP-ACM", "Structured_Amazon-Google",
    #             "Dirty_DBLP-ACM"]
    # use_cases = ["Structured_Amazon-Google"]
    # use_cases = ["Structured_Amazon-Google", "Structured_Beer", "Textual_Abt-Buy", "Dirty_Walmart-Amazon"]

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
        'size': None,
        'target_class': 'both',  # 'both', 0, 1
        'seeds': [42, 42],  # [42 -> class 0, 42 -> class 1]
    }

    fine_tune = 'simple'  # None, 'simple', 'advanced'

    attn_params = {
        'attn_extractor': 'word_extractor',     # 'attr_extractor', 'word_extractor', 'token_extractor'
        'attn_extr_params': {'special_tokens': True, 'agg_metric': 'mean'},
    }

    # if attn_params['attn_extractor'] == 'word_extractor' and conf['tok'] == 'attr_pair':
    #     raise NotImplementedError()

    # experiment = 'compute_attn', 'attr_to_cls', 'attr_to_cls_entropy', 'word_to_cls', 'entity_to_entity'
    experiment = 'compute_attn'
    sub_experiment = 'comparison'   # 'simple', 'comparison'
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
        assert attn_params['attn_extr_params']["special_tokens"] is True

        # target_categories = ['all', 'all_pred_pos', 'all_pred_neg']
        target_categories = ['all_pos', 'all_neg']
        extractor_name = attn_params['attn_extractor']
        extr_params = attn_params["attn_extr_params"]["agg_metric"]
        is_entropy = '_'
        if experiment == 'attr_to_cls_entropy':
            is_entropy = '_ENTROPY_'

        if sub_experiment == 'simple':

            attn2cls_res = compute_attn_to_cls(use_cases, conf, sampler_conf, fine_tune, attn_params, RESULTS_DIR,
                                               experiment, target_categories)

            out_fname = f"PLOT_ATT2CLS{is_entropy}{conf['tok']}_{sampler_conf['size']}_{fine_tune}_{extractor_name}_{extr_params}.pdf"

        elif sub_experiment == 'comparison':
            print("Target categories set to 'all'")
            target_categories = ['all']
            comparison = 'tok'  # 'tune', 'tok'

            if comparison == 'tune':
                new_fine_tune = None
                print("Load pre-trained data")
                pretrain_att2cls_res = compute_attn_to_cls(use_cases, conf, sampler_conf, new_fine_tune, attn_params,
                                                           RESULTS_DIR, experiment, target_categories)

                new_fine_tune = 'simple'
                print("Load fine-tuned data")
                finetune_att2cls_res = compute_attn_to_cls(use_cases, conf, sampler_conf, new_fine_tune, attn_params,
                                                           RESULTS_DIR, experiment, target_categories)

                attn2cls_res = {}
                for k in pretrain_att2cls_res:
                    attn2cls_res[k] = {'pre_train': pretrain_att2cls_res[k]['all']}
                for k in finetune_att2cls_res:
                    attn2cls_res[k]['fine_tune'] = finetune_att2cls_res[k]['all']

                out_fname = f"PLOT_ATT2CLS{is_entropy}CMP_{comparison}_{conf['tok']}_{sampler_conf['size']}_{extractor_name}_{extr_params}.pdf"

            elif comparison == 'tok':
                new_conf = conf.copy()
                new_conf['tok'] = 'sent_pair'
                print("Load sent_pair data")
                sentpair_att2cls_res = compute_attn_to_cls(use_cases, new_conf, sampler_conf, fine_tune, attn_params,
                                                           RESULTS_DIR, experiment, target_categories)

                new_conf['tok'] = 'attr_pair'
                print("Load attr_pair data")
                attrpair_att2cls_res = compute_attn_to_cls(use_cases, new_conf, sampler_conf, fine_tune, attn_params,
                                                           RESULTS_DIR, experiment, target_categories)

                attn2cls_res = {}
                for k in sentpair_att2cls_res:
                    attn2cls_res[k] = {'sent_pair': sentpair_att2cls_res[k]['all']}
                for k in attrpair_att2cls_res:
                    attn2cls_res[k]['attr_pair'] = attrpair_att2cls_res[k]['all']
                out_fname = f"PLOT_ATT2CLS{is_entropy}CMP_{comparison}_{sampler_conf['size']}_{fine_tune}_{extractor_name}_{extr_params}.pdf"

            else:
                raise ValueError("Wrong comparison variable.")

        else:
            raise ValueError("Wrong sub experiment.")

        out_file = os.path.join(RESULTS_DIR, out_fname)

        if experiment == 'attr_to_cls':
            AttrToClsAttentionAnalyzer.plot_multi_attr_to_cls_attn(attn2cls_res, save_path=out_file)
        else:
            entropy_res = AttrToClsAttentionAnalyzer.analyze_multi_results(attn2cls_res, analysis_type='entropy')
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
        # target_categories = ['all_pred_pos', 'all_pred_neg']
        target_categories = ['all_pos', 'all_neg']

        if sub_experiment == 'simple':
            e2e_res = compute_entity_to_entity_attention(use_cases, conf, sampler_conf, fine_tune, attn_params,
                                                         RESULTS_DIR, analysis_type='cross_entity', ignore_special=True,
                                                         target_categories=target_categories)

            out_fname = f"PLOT_E2E_{conf['tok']}_{sampler_conf['size']}_{fine_tune}_{attn_params['attn_extractor']}.pdf"

        elif sub_experiment == 'comparison':

            # print("Target categories set to 'all'")
            # target_categories = ['all']
            comparison = 'tok'     # 'tune', 'tok', 'tune_tok'

            if comparison == 'tune':
                new_fine_tune = None
                print("Load pre-trained data")
                pretrain_e2e_res = compute_entity_to_entity_attention(use_cases, conf, sampler_conf, new_fine_tune,
                                                                      attn_params, RESULTS_DIR,
                                                                      analysis_type='cross_entity', ignore_special=True,
                                                                      target_categories=['all'])

                new_fine_tune = 'simple'
                print("Load fine-tuned data")
                finetune_e2e_res = compute_entity_to_entity_attention(use_cases, conf, sampler_conf, new_fine_tune,
                                                                      attn_params, RESULTS_DIR,
                                                                      analysis_type='cross_entity', ignore_special=True,
                                                                      target_categories=['all_pos', 'all_neg'])

                e2e_res = {}
                for k in pretrain_e2e_res:
                    e2e_res[k] = {'pre_train': pretrain_e2e_res[k]['all']}
                for k in finetune_e2e_res:
                    e2e_res[k]['finetune_match'] = finetune_e2e_res[k]['all_pos']
                    e2e_res[k]['finetune_non_match'] = finetune_e2e_res[k]['all_neg']

                out_fname = f"PLOT_E2E_CMP_{comparison}_{conf['tok']}_{sampler_conf['size']}_{attn_params['attn_extractor']}.pdf"

            elif comparison == 'tok':
                new_conf = conf.copy()
                new_conf['tok'] = 'sent_pair'
                print("Load sent_pair data")
                sentpair_e2e_res = compute_entity_to_entity_attention(use_cases, new_conf, sampler_conf, fine_tune,
                                                                      attn_params, RESULTS_DIR,
                                                                      analysis_type='cross_entity', ignore_special=True,
                                                                      target_categories=['all_pred_pos', 'all_pred_neg'],
                                                                      precomputed=True)

                new_conf['tok'] = 'attr_pair'
                print("Load attr_pair data")
                attrpair_e2e_res = compute_entity_to_entity_attention(use_cases, new_conf, sampler_conf, fine_tune,
                                                                      attn_params, RESULTS_DIR,
                                                                      analysis_type='cross_entity', ignore_special=True,
                                                                      target_categories=['all_pred_pos', 'all_pred_neg'],
                                                                      precomputed=True)

                e2e_res = {}
                for k in sentpair_e2e_res:
                    e2e_res[k] = {'sentpair_match': sentpair_e2e_res[k]['all_pred_pos']}
                    e2e_res[k]['sentpair_non_match'] = sentpair_e2e_res[k]['all_pred_neg']
                for k in attrpair_e2e_res:
                    e2e_res[k]['attrpair_match'] = attrpair_e2e_res[k]['all_pred_pos']
                    e2e_res[k]['attrpair_non_match'] = attrpair_e2e_res[k]['all_pred_neg']
                out_fname = f"PLOT_E2E_CMP_{comparison}_{sampler_conf['size']}_{fine_tune}_{attn_params['attn_extractor']}.pdf"

            elif comparison == 'tune_tok':
                new_conf = conf.copy()
                new_conf['tok'] = 'sent_pair'
                new_fine_tune = None
                print("Load pretrain sent_pair data")
                pretrain_sent_e2e_res = compute_entity_to_entity_attention(use_cases, new_conf, sampler_conf,
                                                                           new_fine_tune, attn_params, RESULTS_DIR,
                                                                           analysis_type='cross_entity',
                                                                           ignore_special=True,
                                                                           target_categories=['all', 'all_pos',
                                                                                              'all_neg'],
                                                                           precomputed=True)
                                                                           #extract_attention=True, save=True)
                                                                           #target_categories=['all'])

                new_conf = conf.copy()
                new_conf['tok'] = 'attr_pair'
                print("Load pretrain attr_pair data")
                pretrain_attr_e2e_res = compute_entity_to_entity_attention(use_cases, new_conf, sampler_conf,
                                                                           new_fine_tune, attn_params, RESULTS_DIR,
                                                                           analysis_type='cross_entity',
                                                                           ignore_special=True,
                                                                           target_categories=['all', 'all_pos',
                                                                                              'all_neg'],
                                                                           precomputed=True)
                                                                           #extract_attention=True, save=True)
                                                                           #target_categories=['all'])

                new_fine_tune = 'simple'
                new_conf['tok'] = 'sent_pair'
                print("Load finetuned sent_pair data")
                finetune_sent_e2e_res = compute_entity_to_entity_attention(use_cases, new_conf, sampler_conf,
                                                                           new_fine_tune, attn_params, RESULTS_DIR,
                                                                           analysis_type='cross_entity',
                                                                           ignore_special=True,
                                                                           target_categories=['all', 'all_pos',
                                                                                              'all_neg'],
                                                                           precomputed=True)
                                                                           #extract_attention=True, save=True)
                                                                           #target_categories=['all'])

                new_conf['tok'] = 'attr_pair'
                print("Load finetuned attr_pair data")
                finetune_attr_e2e_res = compute_entity_to_entity_attention(use_cases, new_conf, sampler_conf,
                                                                           new_fine_tune, attn_params, RESULTS_DIR,
                                                                           analysis_type='cross_entity',
                                                                           ignore_special=True,
                                                                           target_categories=['all', 'all_pos',
                                                                                              'all_neg'],
                                                                           precomputed=True)
                                                                           #extract_attention=True, save=True)
                                                                           #target_categories=['all'])

                e2e_res = {}
                for k in pretrain_sent_e2e_res:
                    e2e_res[k] = {'pretrain_sent_pair': pretrain_sent_e2e_res[k]['all']}
                for k in pretrain_attr_e2e_res:
                    e2e_res[k]['pretrain_attr_pair'] = pretrain_attr_e2e_res[k]['all']
                for k in finetune_sent_e2e_res:
                    e2e_res[k]['finetune_sent_pair'] = finetune_sent_e2e_res[k]['all']
                for k in finetune_attr_e2e_res:
                    e2e_res[k]['finetune_attr_pair'] = finetune_attr_e2e_res[k]['all']
                out_fname = f"PLOT_E2E_CMP_{comparison}_{sampler_conf['size']}_{attn_params['attn_extractor']}.pdf"

            else:
                raise ValueError("Wrong comparison variable.")

        else:
            raise ValueError("Wrong sub experiment.")

        out_file = os.path.join(RESULTS_DIR, out_fname)
        EntityToEntityAttentionAnalyzer.plot_multi_entity_to_entity_attn(e2e_res, save_path=out_file)

    else:
        raise NotImplementedError()
