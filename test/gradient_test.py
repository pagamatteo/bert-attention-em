import os
from transformers import AutoTokenizer
from utils.general import get_dataset, get_model, get_sample
from pathlib import Path
from explanation.gradient.plot_gradients import plot_grads, plot_batch_grads
from explanation.gradient.extractors import EntityGradientExtractor, AggregateAttributeGradient
import logging
from multiprocessing import Process


PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
RESULT_DIR = os.path.join(PROJECT_DIR, 'results', 'gradient_analysis')


def run_gradient_test(conf, use_case, fine_tune, sampler_conf, text_unit, special_tokens, RESULT_DIR):
    use_case_conf = conf.copy()
    use_case_conf['use_case'] = use_case
    dataset = get_dataset(use_case_conf)
    tok = use_case_conf['tok']
    model_name = use_case_conf['model_name']

    if fine_tune is not None:
        model_path = os.path.join(MODELS_DIR, fine_tune, f"{use_case}_{tok}_tuned")
    else:
        model_path = None
    model = get_model(model_name, fine_tune, model_path)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)

    complete_sampler_conf = sampler_conf.copy()
    complete_sampler_conf['permute'] = use_case_conf['permute']
    sample = get_sample(dataset, complete_sampler_conf)

    # [END] MODEL AND DATA LOADING

    entity_grad_extr = EntityGradientExtractor(
        model,
        tokenizer,
        text_unit,
        special_tokens=special_tokens,
    )
    out_fname = f"{use_case}_{use_case_conf['tok']}_{sampler_conf['size']}_{fine_tune}_{text_unit}_{special_tokens}"
    out_dir = os.path.join(RESULT_DIR, use_case, out_fname)
    grads_data = entity_grad_extr.extract(sample, sample.max_len, out_path=out_dir)

    # if text_unit == 'attrs' and agg:
    #     aggregator = AggregateAttributeGradient(grads_data, target_categories=target_categories)
    #     agg_grads = aggregator.aggregate(agg)
    #     for cat in agg_grads:
    #         if agg_grads[cat] is None:
    #             logging.info(f"No {cat} data.")
    #         else:
    #             plot_grads(agg_grads[cat], 'all', title=f'{agg} {cat} gradients')
    #
    # else:
    #     plot_batch_grads(grads_data, 'all')


if __name__ == '__main__':

    # [BEGIN] PARAMS

    use_cases = ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"]
    # use_cases = ["Structured_Fodors-Zagats", "Structured_Beer"]

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
        'size': 2,
        'target_class': 'both',  # 'both', 0, 1
        'seeds': [42, 42],  # [42 -> class 0, 42 -> class 1]
    }

    fine_tune = 'simple'  # None, 'simple', 'advanced'

    text_unit = 'words'
    special_tokens = False
    agg = 'mean'
    target_categories = ['all', 'all_pos', 'all_neg', 'all_pred_pos', 'all_pred_neg', 'tp', 'tn', 'fp', 'fn']

    # [END] PARAMS

    # [BEGIN] MODEL AND DATA LOADING

    processes = [Process(target=run_gradient_test,
                         args=(conf, use_case, fine_tune, sampler_conf, text_unit, special_tokens, RESULT_DIR,)) for
                 use_case in use_cases]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
