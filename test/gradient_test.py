import os
from transformers import AutoTokenizer
from utils.general import get_dataset, get_model, get_sample
from pathlib import Path
from explanation.gradient.plot_gradients import plot_grads, plot_batch_grads
from explanation.gradient.extractors import EntityGradientExtractor, AggregateAttributeGradient
import logging


PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')


if __name__ == '__main__':

    # [BEGIN] PARAMS

    conf = {
        'use_case': "Structured_Beer",
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
        'size': 10,
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

    dataset = get_dataset(conf)
    use_case = conf['use_case']
    tok = conf['tok']
    model_name = conf['model_name']

    if fine_tune is not None:
        model_path = os.path.join(MODELS_DIR, fine_tune, f"{use_case}_{tok}_tuned")
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
        text_unit,
        special_tokens=special_tokens,
    )
    grads_data = entity_grad_extr.extract(sample, sample.max_len)

    if text_unit == 'attrs' and agg:
        aggregator = AggregateAttributeGradient(grads_data, target_categories=target_categories)
        agg_grads = aggregator.aggregate(agg)
        for cat in agg_grads:
            if agg_grads[cat] is None:
                logging.info(f"No {cat} data.")
            else:
                plot_grads(agg_grads[cat], 'all', title=f'{agg} {cat} gradients')

    else:
        plot_batch_grads(grads_data, 'all')
