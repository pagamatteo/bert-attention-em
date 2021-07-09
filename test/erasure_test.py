import os
from transformers import AutoTokenizer
from utils.general import get_dataset, get_model, get_sample
from pathlib import Path
from explanation.erasure.extractors import DeltaPredictionExtractor
from utils.nlp import get_synonyms_from_sent_pair, get_random_words_from_sent_pair, get_common_words_from_sent_pair

PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')

if __name__ == '__main__':

    # [BEGIN] PARAMS

    conf = {
        'use_case': "Structured_Fodors-Zagats",
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

    word_selector_fn1 = lambda x, y: get_synonyms_from_sent_pair(x, y, topk=1)
    word_selector_fn2 = lambda x, y, z: get_random_words_from_sent_pair(x, y, num_words=1, exclude_synonyms=True,
                                                                        seed=z)
    word_selector_fn3 = lambda x, y, z: get_common_words_from_sent_pair(x, y, num_words=1, seed=z)
    word_selector_fns = [word_selector_fn1, word_selector_fn2, word_selector_fn3]
    delta_pred_extractor = DeltaPredictionExtractor(model, tokenizer, word_selector_fns, ['jsd', 'tvd'],
                                                    only_left_word=True, only_right_word=True)
    delta_score_res = delta_pred_extractor.extract(sample)
    print(delta_score_res)
