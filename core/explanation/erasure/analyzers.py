from core.explanation.erasure.extractors import AggregateDeltaPredictionScores
from utils.result_collector import BinaryClassificationResultsAggregator
import itertools
import numpy as np


class ErasureMethodHitsAnalysis(object):
    def __init__(self, delta_scores):
        delta_score_aggregator = AggregateDeltaPredictionScores(delta_scores)
        self.delta_scores = delta_score_aggregator.get_grouped_data()

    def get_scores(self, remove_from_entity: str = 'pair', delta_metric: str = 'jsd', data_categories: list = None):
        assert isinstance(remove_from_entity, str), "Wrong data type for parameter 'remove_from_entity'."
        assert remove_from_entity in ['pair', 'left', 'right']
        assert isinstance(delta_metric, str), "Wrong data type for parameter 'delta_metric'."
        assert delta_metric in ['jsd', 'tvd', 'diff']
        if data_categories is not None:
            assert isinstance(data_categories, list), "Wrong data type for parameter 'data_types'."
            assert all([dc in BinaryClassificationResultsAggregator.categories for dc in data_categories])

        delta_scores_by_cat = {}
        for method in self.delta_scores:
            method_scores = self.delta_scores[method]
            for text_unit in method_scores:
                text_unit_scores = method_scores[text_unit][remove_from_entity][delta_metric]
                for data_cat in text_unit_scores:

                    if data_categories is not None:
                        if data_cat not in data_categories:
                            continue

                    if text_unit_scores[data_cat] is None:
                        continue

                    if data_cat not in delta_scores_by_cat:
                        delta_scores_by_cat[data_cat] = {method: text_unit_scores[data_cat]}
                    else:
                        delta_scores_by_cat[data_cat][method] = text_unit_scores[data_cat]

        hits_by_cat = {}
        for cat in delta_scores_by_cat:
            cat_delta_scores = delta_scores_by_cat[cat]
            cmp_meth_pairs = list(itertools.combinations(list(cat_delta_scores), 2))
            cmp_hits = {}
            for cmp_meth_pair in cmp_meth_pairs:
                cmp_meth1 = cmp_meth_pair[0]
                cmp_meth2 = cmp_meth_pair[1]
                scores1 = np.array(cat_delta_scores[cmp_meth1])
                scores2 = np.array(cat_delta_scores[cmp_meth2])
                cmp_key = f'{cmp_meth1}_{cmp_meth2}'

                cmp_hits[cmp_key] = {
                    cmp_meth1: int(((scores1 >= scores2).sum() / len(scores1)) * 100),
                    cmp_meth2: int(((scores2 > scores1).sum() / len(scores1)) * 100),
                }
            hits_by_cat[cat] = cmp_hits

        first_cat = list(hits_by_cat.keys())[0]
        method_pairs = len(hits_by_cat[first_cat])
        hit_stats_by_method_pair = [{} for _ in range(method_pairs)]
        for cat in hits_by_cat:
            for i, method_pair in enumerate(hits_by_cat[cat]):
                hit_stats_by_method_pair[i][cat] = hits_by_cat[cat][method_pair]

        return hit_stats_by_method_pair
