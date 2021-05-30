import pytorch_lightning as pl
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn
from sklearn.metrics import f1_score


class MatcherTransformer(pl.LightningModule):
    def __init__(self, model_name: str, num_labels: int = 2,
                 learning_rate: float = 2e-5, max_epochs: int = 10,
                 adam_epsilon: float = 1e-8, warmup_steps: int = 0,
                 weight_decay: float = 0.0, train_batch_size: int = 32,
                 eval_batch_size: int = 32):

        super().__init__()

        if not isinstance(model_name, str):
            raise TypeError("Wrong model name type.")

        # save hyper parameters in the hparams attribute of the model
        self.save_hyperparameters()

        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True,
                                               output_attentions=True)
        self.dropout = nn.Dropout(self.model.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):

        # input shapes: (batch_size, channel, seq_len)

        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.squeeze(1)

        if token_type_ids is not None:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask
            )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.hparams.num_labels), labels.view(-1))

        # get hidden states
        hidden_states = outputs[2]

        # get attention maps
        attention = outputs[-1]

        return {'loss': loss, 'logits': logits, 'hidden_states': hidden_states, 'attentions': attention}

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        train_loss = outputs['loss']
        logits = outputs['logits']

        preds = torch.argmax(logits, axis=1)
        labels = batch["labels"]

        return {'loss': train_loss, "preds": preds, "labels": labels}

    def training_epoch_end(self, outputs):

        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.log('train_loss', loss, prog_bar=True)
        f1_scores = f1_score(labels, preds, average=None)
        f1_neg = f1_scores[0]
        f1_pos = f1_scores[1]
        self.log('train_f1_neg', f1_neg)
        self.log('train_f1_pos', f1_pos, prog_bar=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss = outputs['loss']
        logits = outputs['logits']

        preds = torch.argmax(logits, axis=1)
        labels = batch["labels"]

        return {'loss': val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):

        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.log('val_loss', loss, prog_bar=True)
        f1_scores = f1_score(labels, preds, average=None)
        f1_neg = f1_scores[0]
        f1_pos = f1_scores[1]
        self.log('val_f1_neg', f1_neg)
        self.log('val_f1_pos', f1_pos, prog_bar=True)

    def setup(self, stage):
        if stage == 'fit':
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()

            # Calculate total steps
            self.total_steps = (
                    (len(train_loader.dataset) // self.hparams.train_batch_size)
                    * float(self.hparams.max_epochs)
            )

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]


def classify(test_loader, model):
    model.eval()

    preds = torch.empty(0)
    labels = torch.empty(0)
    num_batches = len(test_loader)
    ix = 1
    for test_batch in test_loader:
        print("{}/{}".format(ix, num_batches))
        input_ids = test_batch['input_ids']
        attention_mask = test_batch['attention_mask']
        token_type_ids = test_batch['token_type_ids']
        batch_labels = test_batch['labels']

        with torch.no_grad():
            _, logits, hidden_states, _ = model(input_ids=input_ids, attention_mask=attention_mask,
                                                token_type_ids=token_type_ids)
        batch_preds = torch.argmax(logits, axis=1)
        preds = torch.cat((preds, batch_preds))
        labels = torch.cat((labels, batch_labels))

        ix += 1

    average_f1 = f1_score(labels, preds)
    f1_class_scores = f1_score(labels, preds, average=None)
    neg_f1 = f1_class_scores[0]
    pos_f1 = f1_class_scores[1]

    print("Average F1: {}".format(average_f1))
    print("F1 Neg class: {}".format(neg_f1))
    print("F1 Pos class: {}".format(pos_f1))


if __name__ == '__main__':
    # use_case = "Structured/Fodors-Zagats" # v
    # use_case = "Structured/DBLP-GoogleScholar"
    # use_case = "Structured/DBLP-ACM" # v
    # use_case = "Structured/Amazon-Google" # v
    # use_case = "Structured/Walmart-Amazon" # v
    # use_case = "Structured/Beer" # v
    # use_case = "Structured/iTunes-Amazon" # v
    # use_case = "Textual/Abt-Buy" # v
    # use_case = "Dirty/iTunes-Amazon" # v
    # use_case = "Dirty/DBLP-ACM" # v
    # use_case = "Dirty/DBLP-GoogleScholar"
    use_case = "Dirty/Walmart-Amazon"  # v

    fine_tuning = False
    model_out_name = "MatcherTransformer_{}.zip".format(use_case.replace("/", "_"))
    evaluate = True

    # get data
    data_collector = DataCollector()
    use_case_data_dir = data_collector.get_data(use_case)
    train_path = os.path.join(use_case_data_dir, "train.csv")
    valid_path = os.path.join(use_case_data_dir, "valid.csv")
    test_path = os.path.join(use_case_data_dir, "test.csv")

    model_name = 'bert-base-uncased'
    dm = EMDataModule(train_path, valid_path, test_path, model_name, max_len=128)
    dm.setup()

    if fine_tuning:

        print("Starting fine tuning the model...")
        pl.seed_everything(42)

        N_EPOCHS = 10

        # fine-tuning the transformer
        model = MatcherTransformer(model_name, max_epochs=N_EPOCHS)
        trainer = pl.Trainer(deterministic=True, gpus=1, progress_bar_refresh_rate=30, max_epochs=N_EPOCHS)
        trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

        # # check results
        # %load_ext tensorboard
        # %tensorboard --logdir ./lightning_logs

        # save the model
        trainer.save_checkpoint(os.path.join(drive_models_out_dir, model_out_name))

    else:

        print("Loading pre-trained model...")
        model = MatcherTransformer.load_from_checkpoint(
            checkpoint_path=os.path.join(drive_models_out_dir, model_out_name))

    if evaluate:
        classify(dm.test_dataloader(), model)
