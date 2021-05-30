import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd


class EMDataset(Dataset):

    def __init__(self, data: pd.DataFrame, model_name: str,
                 tokenization: str = 'sent_pair',
                 label_col: str = 'label', left_prefix: str = 'left_',
                 right_prefix: str = 'right_', max_len: int = 256,
                 verbose: bool = False, categories: list = None,
                 permute: bool = False, seed: int = 42):

        assert isinstance(tokenization, str)
        assert tokenization in ['sent_pair', 'attr', 'attr_pair']

        self.data = data
        self.model_name = model_name
        self.tokenization = tokenization
        self.label_col = label_col
        self.left_prefix = left_prefix
        self.right_prefix = right_prefix
        self.max_len = max_len
        assert (self.max_len % 2) == 0
        self.verbose = verbose
        self.categories = categories
        self.permute = permute
        self.seed = seed

        if label_col not in self.data.columns:
            raise ValueError("Label column not found.")

        # remove labels from feature table
        self.labels = self.data[self.label_col]
        self.X = self.data.drop([self.label_col], axis=1)

        # remove entity identifiers
        ids = ['{}id'.format(self.left_prefix), '{}id'.format(self.right_prefix)]
        for single_id in ids:
            if single_id in self.X.columns:
                self.X = self.X.drop([single_id], axis=1)

        # extract left and right features
        self.left_cols = []
        self.right_cols = []
        remove_cols = []
        for col in self.X.columns:
            if col.startswith(left_prefix):
                self.left_cols.append(col)
            elif col.startswith(right_prefix):
                self.right_cols.append(col)
            else:
                remove_cols.append(col)

        if len(remove_cols) > 0:
            print("Warning: the following columns will be removed from the data: {}".format(remove_cols))
            self.X = self.X.drop(remove_cols, axis=1)

        # check that the dataset contains the same number of left and right features
        assert len(self.left_cols) == len(self.right_cols)

        # check that the left and right feature names are equal
        c1 = [c.replace(self.left_prefix, "") for c in self.left_cols]
        c2 = [c.replace(self.right_prefix, "") for c in self.right_cols]
        assert c1 == c2

        self.complete_data = self.X.copy()
        self.complete_data[self.label_col] = self.labels
        self.columns = c1
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

    def get_complete_data(self):
        return self.complete_data

    def get_columns(self):
        return self.columns

    def get_params(self):
        params = {'model_name': self.model_name, 'label_col': self.label_col, 'left_prefix': self.left_prefix,
                  'right_prefix': self.right_prefix, 'max_len': self.max_len, 'tokenization': self.tokenization}

        return params

    def __len__(self):
        return len(self.data)

    def convert_to_features(self, entity1, entity2):

        if self.tokenization == 'sent_pair':

            sent1 = ' '.join([str(val) for val in entity1.to_list()])  # if val != unk_token])
            sent2 = ' '.join([str(val) for val in entity2.to_list()])  # if val != unk_token])

            # Tokenize the text pairs
            features = self.tokenizer(sent1, sent2, padding='max_length',
                                      truncation=True, return_tensors="pt",
                                      max_length=self.max_len,
                                      add_special_tokens=True,
                                      pad_to_max_length=True,
                                      return_attention_mask=True)

        elif self.tokenization == 'attr':
            sent = ""
            for attr_val in entity1.to_list():
                sent += "{} [SEP] ".format(str(attr_val))
            for attr_val in entity2.to_list():
                sent += "{} [SEP] ".format(str(attr_val))
            sent = sent[:-7]  # remove last ' [SEP] '
            features = self.tokenizer(sent, padding='max_length',
                                      truncation=True, return_tensors="pt",
                                      max_length=self.max_len,
                                      add_special_tokens=True,
                                      pad_to_max_length=True,
                                      return_attention_mask=True)

        elif self.tokenization == 'attr_pair':
            sent1 = ""
            for attr_val in entity1.to_list():
                sent1 += "{} [SEP] ".format(str(attr_val))
            sent1 = sent1[:-7]  # remove last ' [SEP] '

            sent2 = ""
            for attr_val in entity2.to_list():
                sent2 += "{} [SEP] ".format(str(attr_val))
            sent2 = sent2[:-7]  # remove last ' [SEP] '

            features = self.tokenizer(sent1, sent2, padding='max_length',
                                      truncation=True, return_tensors="pt",
                                      max_length=self.max_len,
                                      add_special_tokens=True,
                                      pad_to_max_length=True,
                                      return_attention_mask=True)

        flat_features = {}
        for feature in features:
            flat_features[feature] = features[feature].squeeze(0)

        return entity1, entity2, flat_features

    def __getitem__(self, idx):
        row = self.X.iloc[idx]
        label = self.labels.iloc[idx]
        if self.categories is not None:
            category = self.categories[idx]

        left_row = row[self.left_cols]
        left_row.index = self.columns

        right_row = row[self.right_cols]
        right_row.index = self.columns

        unk_token = self.tokenizer.unk_token
        left_row = left_row.fillna(unk_token)
        right_row = right_row.fillna(unk_token)

        if self.permute:
            np.random.seed(self.seed + idx)
            # perm = np.random.permutation(len(self.columns))
            perm = list(reversed(range(len(self.columns))))
            perm_cols = [self.columns[ix] for ix in perm]
            left_row.index = perm_cols
            left_row = left_row.reindex(index=self.columns)
            for attr, val in left_row.copy().iteritems():
                permuted_val = ' '.join(np.random.permutation(str(val).split()))
                left_row[attr] = permuted_val

        left_row, right_row, tokenized_row = self.convert_to_features(left_row, right_row)
        tokenized_row['labels'] = torch.tensor(label, dtype=torch.long)
        if self.categories is not None:
            tokenized_row['category'] = category

        if not self.verbose:
            return tokenized_row

        return left_row, right_row, tokenized_row


if __name__ == '__main__':
    from utils.data_collector import DataCollector
    import os

    uc = "Structured_Fodors-Zagats"

    data_type = 'train'
    # data_type = 'test'
    # data_type = 'valid'

    model_name = 'bert-base-uncased'
    tok = 'sent_pair'
    # tok = 'attr'
    # tok = 'attr_pair'
    label_col = 'label'
    left_prefix = 'left_'
    right_prefix = 'right_'
    max_len = 128
    verbose = False
    permute = False

    # download the data
    data_collector = DataCollector()
    use_case_data_dir = data_collector.get_data(uc)

    # data selection
    if data_type == 'train':
        dataset_path = os.path.join(use_case_data_dir, "train.csv")
    elif data_type == 'test':
        dataset_path = os.path.join(use_case_data_dir, "test.csv")
    else:
        dataset_path = os.path.join(use_case_data_dir, "valid.csv")

    data = pd.read_csv(dataset_path)
    dataset = EMDataset(data, model_name, tokenization=tok, label_col=label_col, left_prefix=left_prefix,
                        right_prefix=right_prefix, max_len=max_len, verbose=verbose, permute=permute)

    row = dataset[0]
    left_entity = None
    right_entity = None
    features = None

    if verbose:
        left_entity = row[0]
        right_entity = row[1]
        features = row[2]
    else:
        features = row

    assert features is not None

    if left_entity is not None:
        print(left_entity)

    if right_entity is not None:
        print(right_entity)

    row_text = dataset.tokenizer.convert_ids_to_tokens(features['input_ids'])
    row_label = features['labels']
    print(row_text)
    print(row_label)
    print("Num. sentences: {}".format(len(features['token_type_ids'].unique())))
