import os
import pprint
from collections import OrderedDict
import pandas as pd
from utils.general import get_use_case


def get_df(use_case, data_type):
    use_case_data_dir = get_use_case(use_case)

    if data_type == 'train':
        dataset_path = os.path.join(use_case_data_dir, "train.csv")
    elif data_type == 'test':
        dataset_path = os.path.join(use_case_data_dir, "test.csv")
    else:
        dataset_path = os.path.join(use_case_data_dir, "valid.csv")

    data = pd.read_csv(dataset_path)

    return data


def get_use_case_avg_attr_len(df):
    # ignore non relevant attribute
    drop_attrs = []
    for attr in df.columns:
        if 'id' in attr or 'label' in attr:
            drop_attrs.append(attr)
    df.drop(drop_attrs, axis=1, inplace=True)
    print(df.columns)
    return df.applymap(lambda x: 0 if pd.isnull(x) else len(str(x))).mean(axis=0).map(lambda x: round(x, 2)).to_dict(
        OrderedDict)


def get_benchmark_avg_attr_len(use_cases, data_type):
    dfs = [get_df(use_case=uc, data_type=data_type) for uc in use_cases]

    avg_attr_len = {}
    for uc_idx in range(len(use_cases)):
        uc = use_cases[uc_idx]
        print(uc)
        df = dfs[uc_idx]
        avg_attr_len[uc] = get_use_case_avg_attr_len(df)

    return avg_attr_len


if __name__ == '__main__':

    use_cases = ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                 "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                 "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                 "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"]

    avg_attr_len = get_benchmark_avg_attr_len(use_cases, data_type='train')
    print("\n\n\n")
    print("----" * 10)
    for uc in avg_attr_len:
        print(uc)
        pprint.pprint(avg_attr_len[uc])
        print()
