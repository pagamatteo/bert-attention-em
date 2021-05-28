from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_metric
import numpy as np
import os
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score

from models.em_dataset import EMDataset

PROJECT_DIR = os.path.abspath('..')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'models', 'simple')


def train(model_name: str, num_epochs: int, train_dataset: EMDataset, val_dataset: EMDataset,
          out_model_path: str = None):

    assert isinstance(model_name, str), "Wrong data type for parameter 'model_name'."
    assert isinstance(num_epochs, int), "Wrong data type for parameter 'num_epochs'."
    assert isinstance(train_dataset, EMDataset), "Wrong data type for parameter 'train_dataset'."
    assert isinstance(val_dataset, EMDataset), "Wrong data type for parameter 'val_dataset'."
    if out_model_path is not None:
        assert isinstance(out_model_path, str), "Wrong data type for parameter 'out_model_path'."

    training_args = TrainingArguments(
        output_dir=os.path.join(RESULTS_DIR, 'results'),  # output directory
        num_train_epochs=num_epochs,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=os.path.join(RESULTS_DIR, 'logs'),  # directory for storing logs
        logging_steps=10,
        evaluation_strategy="epoch",
        seed=42,
    )

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                  num_labels=2,
                                                                  output_attentions=True)

    metric = load_metric("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = logits[0]
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        # model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        model_init=model_init,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    if out_model_path is not None:
        trainer.save_model(out_model_path)


def evaluate(model_path: str, eval_dataset: EMDataset):

    assert isinstance(model_path, str), "Wrong data type for parameter 'model_path'."
    assert isinstance(eval_dataset, EMDataset), "Wrong data type for parameter 'eval_dataset'."
    assert os.path.exists(model_path), f"No model found at {model_path}."

    print("Starting the inference task...")

    tuned_model = AutoModelForSequenceClassification.from_pretrained(model_path)

    tuned_model.to('cpu')
    tuned_model.eval()

    preds = []
    labels = []
    for features in tqdm(eval_dataset):
        input_ids = features['input_ids'].unsqueeze(0)
        token_type_ids = features['token_type_ids'].unsqueeze(0)
        attention_mask = features['attention_mask'].unsqueeze(0)
        label = features['labels'].tolist()
        labels.append(label)

        outputs = tuned_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logits = outputs['logits']
        # attns = outputs['attentions']
        pred = torch.argmax(logits, axis=1).tolist()
        preds.append(pred)

    print("F1: {}".format(f1_score(labels, preds)))


if __name__ == '__main__':
    from utils.data_collector import DataCollector
    import os
    import pandas as pd

    fit = False

    uc = "Structured_Fodors-Zagats"

    label_col = 'label'
    left_prefix = 'left_'
    right_prefix = 'right_'
    verbose = False
    permute = False

    # download the data
    data_collector = DataCollector()
    use_case_data_dir = data_collector.get_data(uc)

    model_name = 'bert-base-uncased'
    max_len = 128

    train_dataset_path = os.path.join(use_case_data_dir, "train.csv")
    train_data = pd.read_csv(train_dataset_path)
    train_dataset = EMDataset(train_data, model_name, tokenization='sent_pair', label_col=label_col,
                              left_prefix=left_prefix, right_prefix=right_prefix, max_len=max_len, verbose=verbose,
                              permute=permute)

    val_dataset_path = os.path.join(use_case_data_dir, "valid.csv")
    val_data = pd.read_csv(val_dataset_path)
    val_dataset = EMDataset(val_data, model_name, tokenization='sent_pair', label_col=label_col,
                            left_prefix=left_prefix, right_prefix=right_prefix, max_len=max_len, verbose=verbose,
                            permute=permute)

    test_dataset_path = os.path.join(use_case_data_dir, "test.csv")
    test_data = pd.read_csv(test_dataset_path)
    test_dataset = EMDataset(test_data, model_name, tokenization='sent_pair', label_col=label_col,
                             left_prefix=left_prefix, right_prefix=right_prefix, max_len=max_len, verbose=verbose,
                             permute=permute)

    out_model_path = os.path.join(RESULTS_DIR, f"{uc}_tuned")

    if fit:
        num_epochs = 10
        train(model_name, num_epochs, train_dataset, val_dataset, out_model_path=out_model_path)
    else:
        evaluate(out_model_path, test_dataset)
