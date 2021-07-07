import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from explanation.gradient.extractors import EntityGradientExtractor


def plot_grads(grads_data: list, target_entity: str):
    EntityGradientExtractor.check_extracted_grad(grads_data)
    entities = ['all', 'left', 'right']
    assert isinstance(target_entity, str), "Wrong data type for parameter 'target_entity'."
    assert target_entity in entities, f"Wrong target entity: {target_entity} not in {entities}"

    for idx, grad_data in enumerate(grads_data):

        if grad_data is None:
            logging.info(f"No gradients for item {idx}.")
            continue

        plt.subplots(figsize=(20, 10))

        x = grad_data['grad'][target_entity]

        # check for duplicated labels
        label_counts = pd.Series(x).value_counts()
        not_unique_labels = label_counts[label_counts > 1]
        if len(not_unique_labels) > 0:
            new_x = x.copy()
            for nul in list(not_unique_labels.index):
                nul_idxs = np.where(np.array(x) == nul)[0]
                for i, nul_idx in enumerate(nul_idxs, 1):
                    new_x[nul_idx] = f'{nul}_{i}'           # concatenating the duplicated label with an incremental id
            x = new_x.copy()

        y = grad_data['grad'][f'{target_entity}_grad']
        label = grad_data['label']
        prob = grad_data['prob']
        pred = grad_data['pred']

        plt.bar(x, y)
        plt.xticks(rotation=90)
        plt.title(f"Gradients for item#{idx} - label: {label} - pred: {pred} - prob: {prob}")
        plt.show()
