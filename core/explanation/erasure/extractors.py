import torch
import pandas as pd
from torch.nn.functional import softmax
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
import numpy as np
import pathlib
import pickle
import os

from core.data_models.em_dataset import EMDataset
from utils.bert_utils import tokenize_entity_pair
from utils.result_collector import BinaryClassificationResultsAggregator


class DeltaPredictionExtractor(object):
    def __init__(self, model, tokenizer, word_selector_fns: dict, delta_metrics: list, text_unit: str,
                 single_words: bool = False, text_clean_fn=None, only_left_word: bool = False,
                 only_right_word: bool = False):

        assert isinstance(word_selector_fns, dict), "Wrong data type for parameter 'word_selector_fns'."
        assert len(word_selector_fns) > 0, "Empty word_selector_fns."
        assert isinstance(delta_metrics, list), "Wrong data type for parameter 'delta_metrics'."
        assert len(delta_metrics) > 0, "Empty delta_metrics."
        assert all([isinstance(m, str) for m in delta_metrics]), "Wrong data format for parameter 'delta_metrics'."
        available_metrics = {'jsd': lambda x, y: jensenshannon(x, y), 'tvd': lambda x, y: (sum(abs(x - y)) / 2).item()}
        assert all([m in available_metrics for m in delta_metrics]), f"Delta metric not in {list(available_metrics)}."
        available_text_units = ['sent', 'attr']
        assert isinstance(text_unit, str), "Wrong data type for parameter 'text_unit'."
        assert text_unit in available_text_units, f"Text unit {text_unit} not found in {available_text_units}."
        assert isinstance(single_words, bool), "Wrong data type for parameter 'single_words'."
        assert isinstance(only_left_word, bool), "Wrong data type for parameter 'only_left_word'."
        assert isinstance(only_right_word, bool), "Wrong data type for parameter 'only_right_word'."

        self.model = model
        self.tokenizer = tokenizer
        self.word_selector_fns = word_selector_fns
        self.delta_metrics = delta_metrics
        self.delta_fns = [available_metrics[dm] for dm in delta_metrics]
        self.text_unit = text_unit
        self.single_words = single_words
        self.text_clean_fn = text_clean_fn
        self.only_left_word = only_left_word
        self.only_right_word = only_right_word

    @staticmethod
    def check_extracted_scores(results: dict):
        assert isinstance(results, dict), "Wrong data type for parameter 'results'."
        assert len(results) > 0, "Empty data."
        error_msg = "Wrong data format for parameter 'results'."
        first_level_params = ['label', 'pred', 'probs', 'target_words', 'idx', 'pair']
        second_level_params = ['jsd', 'tvd', 'diff', 'flip']
        for word_fn_name, word_fn_res in results.items():
            assert isinstance(word_fn_res, list), error_msg
            assert len(word_fn_res) > 0, error_msg
            for row_res in word_fn_res:
                assert isinstance(row_res, dict), error_msg
                for text_unit_key, text_unit_res in row_res.items():
                    assert isinstance(text_unit_res, dict), error_msg
                    assert all([p in text_unit_res for p in first_level_params]), error_msg
                    assert isinstance(text_unit_res['pair'], list), error_msg
                    assert len(text_unit_res['pair']) > 0, error_msg
                    for text_unit_res_by_metric in text_unit_res['pair']:
                        assert isinstance(text_unit_res_by_metric, dict), error_msg
                        assert all([p in text_unit_res_by_metric for p in second_level_params]), error_msg
                    if 'left' in text_unit_res and text_unit_res['left'] is not None:
                        assert isinstance(text_unit_res['left'], list), error_msg
                        assert len(text_unit_res['left']) > 0, error_msg
                        for text_unit_res_by_metric in text_unit_res['left']:
                            assert isinstance(text_unit_res_by_metric, dict), error_msg
                            assert all([p in text_unit_res_by_metric for p in second_level_params]), error_msg
                    if 'right' in text_unit_res and text_unit_res['right'] is not None:
                        assert isinstance(text_unit_res['right'], list), error_msg
                        assert len(text_unit_res['right']) > 0, error_msg
                        for text_unit_res_by_metric in text_unit_res['right']:
                            assert isinstance(text_unit_res_by_metric, dict), error_msg
                            assert all([p in text_unit_res_by_metric for p in second_level_params]), error_msg

    def extract(self, data: EMDataset, out_file: str = None):
        # assert isinstance(data, EMDataset), "Wrong data type for parameter 'data'."
        if out_file is not None:
            assert isinstance(out_file, str), "Wrong data type for parameter 'out_file'."

        def _get_pred(model, features):

            new_features = {}
            for key in ['input_ids', 'attention_mask', 'token_type_ids']:
                val = features[key]
                if val.ndim == 1:
                    val = val.unsqueeze(0)
                new_features[key] = val

            with torch.no_grad():
                outputs = model(input_ids=new_features['input_ids'], attention_mask=new_features['attention_mask'],
                                token_type_ids=new_features['token_type_ids'])
            logits = outputs['logits']
            probs = softmax(logits, dim=-1).squeeze(0)
            pred = torch.argmax(logits, dim=1).squeeze(0).item()

            return probs, pred

        def _get_delta_scores(metric_names, metric_fns, score1, score2, pred1, pred2, label):
            out = {}
            for dm_idx in range(len(metric_names)):
                delta_metric = metric_names[dm_idx]
                delta_fn = metric_fns[dm_idx]
                score = delta_fn(score1, score2)
                # FIXME: sometimes the jsd generates nan. I think that this happen when the scores are very close, so
                # FIXME: I replace the nan value with 0
                if np.isnan(score):
                    out[delta_metric] = 0
                else:
                    out[delta_metric] = score
            out['diff'] = (score2[label] - score1[label]).item()
            out['flip'] = pred1 != pred2

            return out

        def _get_entity_words(entity, text_unit, text_clean_fn):

            if text_clean_fn is None:
                text_clean_fn = lambda x: str(x).split()

            if text_unit == 'sent':
                sent = ' '.join([str(val) for val in entity if not pd.isnull(val)])
                words = text_clean_fn(sent)
            else:
                words = {}
                for attr, val in entity.iteritems():
                    if not pd.isnull(val):
                        attr_sent = str(val)
                        attr_words = text_clean_fn(attr_sent)
                        if len(attr_words) > 0:
                            words[attr] = attr_words

            return words

        def _get_entity_pair_words(entity1, entity2, text_unit, text_clean_fn):

            words1 = _get_entity_words(entity1, text_unit, text_clean_fn)
            words2 = _get_entity_words(entity2, text_unit, text_clean_fn)

            if text_unit == 'attr':
                common_attrs = set(words1).intersection(set(words2))
                old_words1 = words1.copy()
                old_words2 = words2.copy()
                words1 = {}
                words2 = {}
                for attr in common_attrs:
                    words1[attr] = old_words1[attr]
                    words2[attr] = old_words2[attr]

            return words1, words2

        def _get_word_pairs_to_remove(left_params, right_params, features, word_sel_fn_name, word_selector_fn, row_idx):

            # select the pairs of words to remove
            if word_sel_fn_name in ['synonym']:
                word_pairs_to_remove = word_selector_fn(left_params['words'], right_params['words'])

            elif word_sel_fn_name in ['random', 'common', 'common_synonym']:
                word_pairs_to_remove = word_selector_fn(left_params['words'], right_params['words'], row_idx)

            elif word_sel_fn_name in ['gradient']:
                # word_pairs_to_remove = word_selector_fn(left_params['entity'], right_params['entity'], features)
                word_pairs_to_remove = word_selector_fn(left_params['entity'], right_params['entity'], row_idx)

            else:
                raise ValueError("Too many arguments expected by the word_selector_fn.")

            if word_pairs_to_remove is not None:  # no word pair to remove
                assert isinstance(word_pairs_to_remove, list), "word_pairs_to_remove is not a dict."
                assert len(word_pairs_to_remove) > 0, "Empty word_pairs_to_remove."
                error_msg = "word_pair_to_remove has a wrong format."
                assert all([isinstance(wptr, dict) for wptr in word_pairs_to_remove]), error_msg
                assert all([p in wptr for wptr in word_pairs_to_remove for p in ['left', 'right']]), error_msg
                assert all([isinstance(wptr[p], str) for wptr in word_pairs_to_remove for p in
                            ['left', 'right'] if wptr[p] is not None]), error_msg

            return word_pairs_to_remove

        def _remove_word_from_entity(entity, word, unk_token):
            found = False
            removed_entity = entity.copy()
            for attr, val in entity.iteritems():
                words = str(val).split()
                if word in words:
                    words.remove(word)
                    if len(words) == 0:
                        new_val = unk_token
                    else:
                        new_val = ' '.join(words)
                    removed_entity[attr] = new_val
                    found = True
                    break
            assert found

            return removed_entity

        def _remove_word_from_entity_by_attr(entity, word, attr, unk_token):
            removed_entity = entity.copy()
            removed_attr = str(removed_entity[attr]).split()
            removed_attr.remove(word)
            removed_attr = ' '.join(removed_attr)
            if removed_attr == '':
                removed_attr = unk_token
            removed_entity[attr] = removed_attr

            return removed_entity

        # save one result for each word selector function
        out_data = {key: [] for key in self.word_selector_fns}

        idx = 0
        # loop over the data
        for left_entity, right_entity, features in tqdm(data):
            original_label = features['labels'].item()

            # get the words contained in the current pair of sentences
            # if the text_unit param is set to 'attr' then a list of words for each attribute is returned
            left_words, right_words = _get_entity_pair_words(left_entity, right_entity, self.text_unit,
                                                             self.text_clean_fn)

            left_params = {'entity': left_entity, 'words': left_words}
            right_params = {'entity': right_entity, 'words': right_words}

            # loop over all the word selector functions and try to retrieve some pairs of words to remove
            # in the following only the records from which it is possible to extract some words for all the word
            # selector functions will be considered
            word_pairs_by_sel_fn = {}
            for word_sel_fn_name, word_selector_fn in self.word_selector_fns.items():

                # select the words to remove
                word_pairs_to_remove_map = {}
                if self.text_unit == 'sent':
                    # select some words to remove for each sentence
                    word_pairs_to_remove = _get_word_pairs_to_remove(left_params, right_params, features,
                                                                     word_sel_fn_name, word_selector_fn, idx)
                    if word_pairs_to_remove is not None:
                        word_pairs_to_remove_map['sent'] = word_pairs_to_remove
                else:  # attr
                    # select some words to remove for each attribute
                    for attr in left_words:
                        if attr not in right_words:
                            attr_word_pairs_to_remove = None
                        else:
                            attr_left_params = left_params.copy()
                            attr_left_params['words'] = left_params['words'][attr]
                            attr_right_params = right_params.copy()
                            attr_right_params['words'] = right_params['words'][attr]
                            attr_word_pairs_to_remove = _get_word_pairs_to_remove(attr_left_params, attr_right_params,
                                                                                  features, word_sel_fn_name,
                                                                                  word_selector_fn, idx)
                        if attr_word_pairs_to_remove is not None:
                            word_pairs_to_remove_map[attr] = attr_word_pairs_to_remove

                if len(word_pairs_to_remove_map) > 0:
                    # if the tokens are extracted at attribute level, check that one pair of tokens is extracted for
                    # each attribute
                    if all([k in list(left_entity.index) for k in word_pairs_to_remove_map]):
                        assert sorted(list(left_entity.index)) == sorted(list(word_pairs_to_remove_map))
                    word_pairs_by_sel_fn[word_sel_fn_name] = word_pairs_to_remove_map

            # skip the current row if none of the word selector functions have extracted some words to remove
            if len(word_pairs_by_sel_fn) != len(self.word_selector_fns):
                idx += 1
                continue

            # get the model prediction for the original pair of sentences
            original_probs, original_pred = _get_pred(self.model, features)

            # remove the selected words from the pair of sentences
            for word_sel_fn_name, word_pairs_by_key in word_pairs_by_sel_fn.items():

                delta_res = {}
                # remove the words by sentence or by attribute
                for key in word_pairs_by_key:
                    word_pairs = word_pairs_by_key[key]
                    removed_left_entity = left_entity.copy()
                    removed_right_entity = right_entity.copy()
                    removed_word_pairs = []

                    # remove some words at sentence level
                    if key == 'sent':
                        # if the single_words flag is enabled then remove from the sentences one word at a time,
                        # otherwise remove jointly all the words included in the word_pairs variable
                        for word_pair in word_pairs:
                            if word_pair['left'] is not None:
                                removed_left_entity = _remove_word_from_entity(removed_left_entity, word_pair['left'],
                                                                               self.tokenizer.unk_token)
                            if word_pair['right'] is not None:
                                removed_right_entity = _remove_word_from_entity(removed_right_entity,
                                                                                word_pair['right'],
                                                                                self.tokenizer.unk_token)
                            if self.single_words:
                                removed_word_pairs.append((removed_left_entity.copy(), removed_right_entity.copy()))
                                removed_left_entity = left_entity.copy()
                                removed_right_entity = right_entity.copy()
                        if not self.single_words:
                            removed_word_pairs.append((removed_left_entity.copy(), removed_right_entity.copy()))

                    else:  # remove some words at attribute level
                        # if the single_words flag is enabled then remove from the attributes one word at a time,
                        # otherwise remove jointly all the words included in the word_pairs variable
                        for word_pair in word_pairs:
                            if word_pair['left'] is not None:
                                removed_left_entity = _remove_word_from_entity_by_attr(removed_left_entity,
                                                                                       word_pair['left'], key,
                                                                                       self.tokenizer.unk_token)

                            if word_pair['right'] is not None:
                                removed_right_entity = _remove_word_from_entity_by_attr(removed_right_entity,
                                                                                        word_pair['right'], key,
                                                                                        self.tokenizer.unk_token)
                            if self.single_words:
                                removed_word_pairs.append((removed_left_entity.copy(), removed_right_entity.copy()))
                                removed_left_entity = left_entity.copy()
                                removed_right_entity = right_entity.copy()
                        if not self.single_words:
                            removed_word_pairs.append((removed_left_entity.copy(), removed_right_entity.copy()))

                    delta_res_params = {'label': original_label, 'pred': original_pred, 'probs': original_probs,
                                        'target_words': word_pairs, 'idx': idx, 'pair': None, 'left': None,
                                        'right': None}

                    # loop over the new entity pairs where some words have been removed
                    for k, removed_word_pair in enumerate(removed_word_pairs):
                        removed_left_entity = removed_word_pair[0]
                        removed_right_entity = removed_word_pair[1]

                        # get the model prediction for a pair of entities where the selected words have been removed
                        pair_removed_features = tokenize_entity_pair(removed_left_entity, removed_right_entity,
                                                                     self.tokenizer, data.tokenization, data.max_len)
                        pair_removed_probs, pair_removed_pred = _get_pred(self.model, pair_removed_features)

                        # get delta score and save
                        pair_delta_scores = _get_delta_scores(self.delta_metrics, self.delta_fns, original_probs,
                                                              pair_removed_probs, original_pred, pair_removed_pred,
                                                              original_label)
                        if delta_res_params['pair'] is None:
                            delta_res_params['pair'] = [pair_delta_scores]
                        else:
                            delta_res_params['pair'].append(pair_delta_scores)

                        # if a word has been removed from both entities and the only_left_word or only_right_word flags
                        # are enabled, then measure the delta score even in the case of removing a single word
                        if word_pairs[k]['left'] is not None and word_pairs[k]['right'] is not None:
                            # get the model prediction for a pair of entities where only the left word has been removed
                            if self.only_left_word:
                                left_removed_features = tokenize_entity_pair(removed_left_entity, right_entity,
                                                                             self.tokenizer, data.tokenization,
                                                                             data.max_len)
                                left_removed_probs, left_removed_pred = _get_pred(self.model, left_removed_features)

                                # get delta score and save
                                left_delta_scores = _get_delta_scores(self.delta_metrics, self.delta_fns,
                                                                      original_probs, left_removed_probs, original_pred,
                                                                      left_removed_pred, original_label)
                                if delta_res_params['left'] is None:
                                    delta_res_params['left'] = [left_delta_scores]
                                else:
                                    delta_res_params['left'].append(left_delta_scores)

                            # get the model prediction for a pair of entities where only the right word has been removed
                            if self.only_right_word:
                                right_removed_features = tokenize_entity_pair(left_entity, removed_right_entity,
                                                                              self.tokenizer, data.tokenization,
                                                                              data.max_len)
                                right_removed_probs, right_removed_pred = _get_pred(self.model, right_removed_features)

                                # get delta score and save
                                right_delta_scores = _get_delta_scores(self.delta_metrics, self.delta_fns,
                                                                       original_probs, right_removed_probs,
                                                                       original_pred, right_removed_pred,
                                                                       original_label)
                                if delta_res_params['right'] is None:
                                    delta_res_params['right'] = [right_delta_scores]
                                else:
                                    delta_res_params['right'].append(right_delta_scores)

                        delta_res[key] = delta_res_params

                out_data[word_sel_fn_name].append(delta_res)
            idx += 1

        if out_file:
            out_dir_path = out_file.split(os.sep)
            out_dir = os.sep.join(out_dir_path[:-1])
            pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
            with open(f'{out_file}.pkl', 'wb') as f:
                pickle.dump(out_data, f)

        return out_data


class AggregateDeltaPredictionScores(object):

    def __init__(self, delta_pred_scores: dict, target_categories: list = None):
        DeltaPredictionExtractor.check_extracted_scores(delta_pred_scores)

        self.delta_pred_scores = delta_pred_scores
        self.agg_metrics = ['mean']
        self.target_categories = target_categories
        self.delta_score_names = ['jsd', 'tvd', 'diff', 'flip']

        self.flatten_delta_scores = self._get_flatten_data(delta_pred_scores)

        self.aggregators = {}
        for delta_score_method in self.flatten_delta_scores:
            delta_score = self.flatten_delta_scores[delta_score_method]
            aggregators_by_method = {}
            for text_unit in delta_score:
                text_unit_delta_score = delta_score[text_unit]
                for pair_or_single in text_unit_delta_score:
                    pair_or_single_scores = text_unit_delta_score[pair_or_single]
                    for target_param in self.delta_score_names:
                        aggregator = BinaryClassificationResultsAggregator(target_param,
                                                                           target_categories=target_categories)
                        aggregator.add_batch_data(pair_or_single_scores)
                        if text_unit in aggregators_by_method:
                            if pair_or_single in aggregators_by_method[text_unit]:
                                aggregators_by_method[text_unit][pair_or_single][target_param] = aggregator
                            else:
                                aggregators_by_method[text_unit][pair_or_single] = {target_param: aggregator}
                        else:
                            aggregators_by_method[text_unit] = {pair_or_single: {target_param: aggregator}}
            self.aggregators[delta_score_method] = aggregators_by_method

    @staticmethod
    def _get_flatten_data(data):
        flatten_delta_scores = {}
        # loop over the functions that select the words to remove
        for word_sel_method_name, word_sel_method_res in data.items():
            results = {}
            # loop over the records that generated the results
            for record in word_sel_method_res:
                # loop over the text unit (i.e., sent or attribute)
                for text_unit in record:
                    text_unit_res = record[text_unit]
                    # loop over the methods used to remove the words (i.e., pair, left, right)
                    params = {p: text_unit_res[p] for p in text_unit_res if p not in ['pair', 'left', 'right']}
                    for word_rem_method in ['pair', 'left', 'right']:
                        if word_rem_method in text_unit_res and text_unit_res[word_rem_method] is not None:
                            res = params.copy()
                            avg_scores = pd.DataFrame(text_unit_res[word_rem_method])
                            res.update(dict(avg_scores.mean()))

                            if text_unit not in results:
                                results[text_unit] = {word_rem_method: [res]}
                            else:
                                if word_rem_method not in results[text_unit]:
                                    results[text_unit][word_rem_method] = [res]
                                else:
                                    results[text_unit][word_rem_method].append(res)

            flatten_delta_scores[word_sel_method_name] = results
        return flatten_delta_scores

    def get_grouped_data(self):
        out_data = {}
        for delta_score_method in self.aggregators:
            delta_scores = self.aggregators[delta_score_method]
            grouped_data_by_method = {}
            for text_unit in delta_scores:
                text_unit_delta_scores = delta_scores[text_unit]
                for pair_or_single in text_unit_delta_scores:
                    pair_or_single_scores = text_unit_delta_scores[pair_or_single]
                    for target_param in pair_or_single_scores:
                        aggregator = pair_or_single_scores[target_param]
                        grouped_data = aggregator.get_results()
                        if text_unit in grouped_data_by_method:
                            if pair_or_single in grouped_data_by_method[text_unit]:
                                grouped_data_by_method[text_unit][pair_or_single][target_param] = grouped_data
                            else:
                                grouped_data_by_method[text_unit][pair_or_single] = {target_param: grouped_data}
                        else:
                            grouped_data_by_method[text_unit] = {pair_or_single: {target_param: grouped_data}}
            out_data[delta_score_method] = grouped_data_by_method
        return out_data

    def aggregate(self, metric: str):
        assert isinstance(metric, str), "Wrong data type for parameter 'metric'."
        assert metric in self.agg_metrics, f"Wrong metric: {metric} not in {self.agg_metrics}."

        def get_agg_scores(aggregator, metric):
            if metric == 'mean':
                agg_data = aggregator.aggregate(metric)
            else:
                raise NotImplementedError()

            return agg_data

        out_data = {}
        for delta_score_method in self.aggregators:
            delta_scores = self.aggregators[delta_score_method]
            delta_out_data = {}
            for target_param in delta_scores:
                aggregator = delta_scores[target_param]
                agg_data = get_agg_scores(aggregator, metric)
                delta_out_data[target_param] = agg_data

            out_data[delta_score_method] = delta_out_data

        return out_data
