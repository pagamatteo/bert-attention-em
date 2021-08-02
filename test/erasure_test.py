import os
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer
from utils.general import get_dataset, get_model, get_sample
from pathlib import Path
from explanation.erasure.extractors import DeltaPredictionExtractor, AggregateDeltaPredictionScores
from utils.nlp import get_synonyms_from_sent_pair, get_random_words_from_sent_pair, get_common_words_from_sent_pair, \
    get_synonyms_or_common_words_from_sent_pair
from utils.nlp import simple_tokenization_and_clean
from explanation.gradient.extractors import EntityGradientExtractor
from explanation.gradient.utils import get_words_sorted_by_gradient

PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')

if __name__ == '__main__':

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
        'size': 4,
        'target_class': 'both',  # 'both', 0, 1
        'seeds': [42, 42],  # [42 -> class 0, 42 -> class 1]
    }

    fine_tune = 'simple'  # None, 'simple', 'advanced'

    synonym_fn = lambda x, y: get_synonyms_from_sent_pair(x, y, num_words=1)
    random_fn = lambda x, y, z: get_random_words_from_sent_pair(x, y, num_words=1, exclude_synonyms=True, seed=z)
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
            'word_selector_fns': {'random': random_fn},
            'text_unit': 'attr',
            'single_words': False,
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
        }
    }

    # experiment_conf = experiment_params['delta_score_common-synonyms_vs_random']
    # experiment_conf = experiment_params['delta_score_random_words_in_attr']
    experiment_conf = experiment_params['delta_score_gradient']

    # use_cases = ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
    #              "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
    #              "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
    #              "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"]
    use_cases = ["Structured_Fodors-Zagats"]

    plot_target_data_type = 'all'
    plot_target_delta_metric = 'jsd'

    if len(use_cases) == 1:
        fig, ax = plt.subplots(figsize=(20, 15))
        axes = [ax]
    else:
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15), sharey=True)
        axes = axes.flat

    for idx, uc in enumerate(use_cases):

        conf['use_case'] = uc
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

        entity_grad_extr = EntityGradientExtractor(model, tokenizer, 'words', special_tokens=False, show_progress=False)
        if 'gradient' in experiment_conf['word_selector_fns']:
            gradient_fn = lambda x, y, z: get_words_sorted_by_gradient(x, y, z, entity_grad_extr)
            experiment_conf['word_selector_fns']['gradient'] = gradient_fn

        delta_pred_extractor = DeltaPredictionExtractor(model, tokenizer, experiment_conf['word_selector_fns'],
                                                        ['jsd', 'tvd'], experiment_conf['text_unit'],
                                                        single_words=experiment_conf['single_words'],
                                                        text_clean_fn=experiment_conf['text_clean_fn'],
                                                        only_left_word=experiment_conf['only_left_word'],
                                                        only_right_word=experiment_conf['only_right_word'])
        delta_score_res = delta_pred_extractor.extract(sample)
        exit(1)

        # TODO: for the gradient experiment get gradient and delta scores and compute the tau rank correlation

        # FIXME: now pair, left and right contain a list of dict
        delta_score_aggregator = AggregateDeltaPredictionScores(delta_score_res)
        delta_grouped_data = delta_score_aggregator.get_grouped_data()

        # agg_delta_scores = delta_score_aggregator.aggregate(agg)
        # plot_data = {method: [agg_delta_scores[method][plot_target_delta_metric][plot_target_data_type]['mean']] for
        #              method in agg_delta_scores}
        # plot_data_table = pd.DataFrame(plot_data)
        # plot_data_table.plot.bar()
        # plt.title(uc)
        # plt.show()

        ax = axes[idx]
        group_plot_data = {}
        for method in delta_grouped_data:
            delta_scores = delta_grouped_data[method]
            for text_unit in delta_scores:
                text_unit_scores = delta_scores[text_unit]
                for pair_or_single in text_unit_scores:
                    pair_or_single_scores = text_unit_scores[pair_or_single]
                    key = f'{method}&{text_unit}&{pair_or_single}'
                    group_plot_data[key] = pair_or_single_scores[plot_target_delta_metric][plot_target_data_type]
        if experiment_conf['text_unit'] == 'attr' and len(group_plot_data) == len(sample.columns):
            aligned_group_plot_data = {}
            for col in sample.columns:
                aligned_key = None
                for key in group_plot_data:
                    if col in key.split("&"):
                        aligned_key = key
                        break
                assert aligned_key is not None
                aligned_group_plot_data[col] = group_plot_data[aligned_key]
            group_plot_data = aligned_group_plot_data.copy()
        group_plot_data_tab = pd.DataFrame(group_plot_data)
        group_plot_data_tab.boxplot(ax=ax)
        ax.set_title(uc)

    plt.show()

    print(":)")
