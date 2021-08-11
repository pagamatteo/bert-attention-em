import os
from transformers import AutoTokenizer
from pathlib import Path
import time
from multiprocessing import Process
import pickle

from utils.general import get_dataset, get_model, get_sample
from utils.nlp import get_synonyms_from_sent_pair, get_random_words_from_sent_pair, get_common_words_from_sent_pair, \
    get_synonyms_or_common_words_from_sent_pair, simple_tokenization_and_clean
from explanation.erasure.extractors import DeltaPredictionExtractor
from explanation.gradient.extractors import EntityGradientExtractor
from explanation.gradient.utils import get_words_sorted_by_gradient, get_pair_words_sorted_by_gradient
from explanation.erasure.plot_erasure_results import plot_delta_scores, plot_erasure_method_hits
from explanation.erasure.analyzers import ErasureMethodHitsAnalysis
from test.gradient_test import load_saved_grads_data


PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
RESULT_DIR = os.path.join(PROJECT_DIR, 'results', 'erasure_analysis')


def run_erasure_test(conf, sampler_conf, fine_tune, models_dir, res_dir, experiment_name):

    synonym_fn = lambda x, y: get_synonyms_from_sent_pair(x, y, num_words=1)
    random_fn = lambda x, y, z: get_random_words_from_sent_pair(x, y, num_words=1, exclude_synonyms=True, seed=z)
    random_top3_fn = lambda x, y, z: get_random_words_from_sent_pair(x, y, num_words=3, exclude_synonyms=False, seed=z)
    common_fn = lambda x, y, z: get_common_words_from_sent_pair(x, y, num_words=1, seed=z)
    common_synonym_fn = lambda x, y, z: get_synonyms_or_common_words_from_sent_pair(x, y, num_words=1, seed=z)
    gradient_fn = None  # this function will be defined later

    experiment_params = {
        'delta_score_synonyms_vs_random_vs_common': {
            'word_selector_fns': {'synonym': synonym_fn, 'random': random_fn, 'common': common_fn},
            'text_unit': 'sent',
            'single_words': False,
            'text_clean_fn': lambda x: simple_tokenization_and_clean(x),
            'only_left_word': True,
            'only_right_word': True,
        },
        'delta_score_random_vs_common': {
            'word_selector_fns': {'random': random_fn, 'common': common_fn},
            'text_unit': 'sent',
            'single_words': False,
            'text_clean_fn': lambda x: simple_tokenization_and_clean(x),
            'only_left_word': True,
            'only_right_word': True,
        },
        'delta_score_common-synonyms_vs_random': {
            'word_selector_fns': {'common_synonym': common_synonym_fn, 'random': random_fn},
            'text_unit': 'sent',
            'single_words': False,
            'text_clean_fn': lambda x: simple_tokenization_and_clean(x),
            'only_left_word': True,
            'only_right_word': True,
        },
        'delta_score_random_words_in_attr': {
            'word_selector_fns': {'random': random_top3_fn},
            'text_unit': 'attr',
            'single_words': True,
            'text_clean_fn': None,
            'only_left_word': False,
            'only_right_word': False,
        },
        'delta_score_gradient': {
            'word_selector_fns': {'gradient': gradient_fn},
            'text_unit': 'sent',
            'single_words': True,
            'text_clean_fn': None,
            'only_left_word': False,
            'only_right_word': False,
        },
        'delta_score_syn_vs_rand_vs_grad': {
            'word_selector_fns': {'common_synonym': common_synonym_fn, 'random': random_fn, 'gradient': gradient_fn},
            'text_unit': 'sent',
            'single_words': False,
            'text_clean_fn': lambda x: simple_tokenization_and_clean(x),
            'only_left_word': False,
            'only_right_word': False,
        }
    }

    experiment_conf = experiment_params[experiment_name]
    # experiment_conf = experiment_params['delta_score_random_words_in_attr']
    # experiment_conf = experiment_params['delta_score_gradient']

    dataset = get_dataset(conf)
    use_case = conf['use_case']
    tok = conf['tok']
    model_name = conf['model_name']

    if fine_tune is not None:
        model_path = os.path.join(models_dir, fine_tune, f"{use_case}_{tok}_tuned")
    else:
        model_path = None
    model = get_model(model_name, fine_tune, model_path)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)

    complete_sampler_conf = sampler_conf.copy()
    complete_sampler_conf['permute'] = conf['permute']
    sample = get_sample(dataset, complete_sampler_conf)

    if 'gradient' in experiment_conf['word_selector_fns']:
        grads = load_saved_grads_data(use_case, conf, sampler_conf, fine_tune,
                                      {'text_unit': 'words', 'special_tokens': True})
        if experiment_name == 'delta_score_syn_vs_rand_vs_grad':
            gradient_fn = lambda x, y, z: get_pair_words_sorted_by_gradient(x, y, z, grads, grad_agg_metric='max',
                                                                            num_words=1)
        elif experiment_name == 'delta_score_gradient':
            gradient_fn = lambda x, y, z: get_words_sorted_by_gradient(x, y, z, grads)
        else:
            raise ValueError("No grad in experiment.")
        experiment_conf['word_selector_fns']['gradient'] = gradient_fn

    delta_pred_extractor = DeltaPredictionExtractor(model, tokenizer, experiment_conf['word_selector_fns'],
                                                    ['jsd', 'tvd'], experiment_conf['text_unit'],
                                                    single_words=experiment_conf['single_words'],
                                                    text_clean_fn=experiment_conf['text_clean_fn'],
                                                    only_left_word=experiment_conf['only_left_word'],
                                                    only_right_word=experiment_conf['only_right_word'])

    # save delta scores
    out_fname = f"{use_case}_{tok}_{sampler_conf['size']}_{fine_tune}_{experiment_name}"
    out_file = os.path.join(res_dir, use_case, out_fname)
    delta_pred_extractor.extract(sample, out_file=out_file)


def load_saved_delta_scores(conf, sampler_conf, fine_tune, experiment_name, res_dir):
    use_case = conf['use_case']
    tok = conf['tok']
    out_fname = f"{use_case}_{tok}_{sampler_conf['size']}_{fine_tune}_{experiment_name}"
    out_file = os.path.join(res_dir, use_case, out_fname)
    delta_scores = pickle.load(open(f"{out_file}.pkl", "rb"))
    return delta_scores


if __name__ == "__main__":

    conf = {
        'data_type': 'train',  # 'train', 'test', 'valid'
        'model_name': 'bert-base-uncased',
        'tok': 'attr_pair',  # 'sent_pair', 'attr', 'attr_pair'
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

    use_cases = ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"]
    # use_cases = ["Structured_Fodors-Zagats"]

    task = 'compute_delta'  # 'compute_delta', 'plot_delta', 'hit_analysis'
    # experiment_name = 'delta_score_common-synonyms_vs_random'
    # experiment_name = 'delta_score_syn_vs_rand_vs_grad'
    experiment_name = 'delta_score_random_words_in_attr'

    start_time = time.time()

    if task == 'compute_delta':
        # no multi process
        # for use_case in use_cases:
        #     print(use_case)
        #
        #     # if not use_case == 'Dirty_DBLP-ACM':
        #     #     continue
        #
        #     uc_conf = conf.copy()
        #     uc_conf['use_case'] = use_case
        #     run_erasure_test(uc_conf, sampler_conf, fine_tune, MODELS_DIR, RESULT_DIR, experiment_name)

        processes = []
        for use_case in use_cases:
            uc_conf = conf.copy()
            uc_conf['use_case'] = use_case
            p = Process(target=run_erasure_test,
                        args=(uc_conf, sampler_conf, fine_tune, MODELS_DIR, RESULT_DIR, experiment_name,))
            processes.append(p)

        for p in processes:
            p.start()

        for p in processes:
            p.join()

    elif task == 'plot_delta':
        delta_scores = {}
        for use_case in use_cases:
            uc_conf = conf.copy()
            uc_conf['use_case'] = use_case
            uc_delta_scores = load_saved_delta_scores(uc_conf, sampler_conf, fine_tune, experiment_name, RESULT_DIR)
            delta_scores[use_case] = uc_delta_scores

        out_plot_name = os.path.join(RESULT_DIR,
                                     f"delta_{conf['tok']}_{sampler_conf['size']}_{fine_tune}_{experiment_name}.pdf")
        plot_delta_scores(delta_scores, plot_target_data_type='all', plot_target_delta_metric='jsd',
                          out_plot_name=out_plot_name)

    elif task == 'hit_analysis':

        hit_stats = {}
        for use_case in use_cases:
            uc_conf = conf.copy()
            uc_conf['use_case'] = use_case
            uc_delta_scores = load_saved_delta_scores(uc_conf, sampler_conf, fine_tune, experiment_name, RESULT_DIR)
            hit_stats[use_case] = ErasureMethodHitsAnalysis(uc_delta_scores).get_scores()

        # data_categories = ['all', 'all_pos', 'all_pred_pos', 'all_neg', 'all_pred_neg']
        data_categories = ['all']
        plot_erasure_method_hits(hit_stats, data_categories=data_categories)

    else:
        raise ValueError("Task not found.")

    end_time = time.time()
    print(end_time - start_time)
    print(':)')

    # TODO: for the gradient experiment get gradient and delta scores and compute the tau rank correlation
