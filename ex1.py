import sys
import wandb
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, \
    DataCollatorWithPadding
from transformers.data.data_collator import default_data_collator
from evaluate import load
from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments
from typing import Optional
from transformers.integrations import WandbCallback

WANDB_API_KEY = '902f9495b8a8e6de924410fab6956889951f9196'


class CustomWandbCallback(WandbCallback):
    def __init__(self, wandb_curr_run):
        super().__init__()
        self.my_run = wandb_curr_run

    def on_save(self, args, state, control, **kwargs):
        super().on_save(args, state, control, **kwargs)
        if state.best_model_checkpoint:
            artifact = wandb.Artifact("model", type="model")
            artifact.add_dir(state.best_model_checkpoint)
            self.my_run.log_artifact(artifact)


def get_datasets(data_arguments, tokenizer):
    def tokenize_function(examples):
        # Don't apply padding during tokenization - we'll let the DataCollator handle that
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            padding=False  # dynamic padding
        )

    raw_datasets = load_dataset("glue", "mrpc")
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    train_dataset = tokenized_datasets["train"]
    if data_arguments.max_train_samples != -1:
        assert data_arguments.max_train_samples >= 0, f"max_train_samples should be positive, got{data_arguments.max_train_samples}"
        train_dataset = train_dataset.select(range(data_arguments.max_train_samples))

    val_dataset = tokenized_datasets["validation"]
    if data_arguments.max_eval_samples != -1:
        assert data_arguments.max_eval_samples >= 0, f"max_eval_samples should be positive, got{data_arguments.max_eval_samples}"
        val_dataset = val_dataset.select(range(data_arguments.max_eval_samples))

    test_dataset = tokenized_datasets["test"]
    if data_arguments.max_predict_samples != -1:
        assert data_arguments.max_predict_samples >= 0, f"max_predict_samples should be positive, got{data_arguments.max_predict_samples}"
        test_dataset = test_dataset.select(range(data_arguments.max_predict_samples))

    return train_dataset, val_dataset, test_dataset


class NopadDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = default_data_collator(features)
        return batch


def load_model_and_tokenizer(model_args):
    config = AutoConfig.from_pretrained(model_args.model_path, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_path, config=config)
    return model, tokenizer


def define_metrics():
    metric = load("accuracy")

    def compute_metrics(preds):
        logits, labels = preds
        predictions = logits.argmax(axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    return compute_metrics


@dataclass
class ModelArguments:
    model_path: Optional[str] = field(
        default="bert-base-uncased",
        metadata={"help": "Pretrained model path or name"}
    )


@dataclass
class DataArguments:
    max_train_samples: int = field(default=-1)
    max_eval_samples: int = field(default=-1)
    max_predict_samples: int = field(default=-1)


@dataclass
class CustomArguments:
    do_train: bool = field(
        default=False,
        metadata={"help": "Whether to run training."}
    )
    do_predict: bool = field(
        default=False,
        metadata={"help": "Whether to run prediction on the test set."}
    )
    lr: float = field(default=5e-5, metadata={"help": "Learning rate"})
    batch_size: int = field(default=16, metadata={"help": "Training batch size"})
    num_train_epochs: int = field(default=3, metadata={"help": "Number of epochs to run train"})
    log: bool = field(default=False)


def load_arguments():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomArguments))
    model_args, data_args, custom_args = parser.parse_args_into_dataclasses()
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=custom_args.num_train_epochs,
        per_device_train_batch_size=custom_args.batch_size,
        per_device_eval_batch_size=1,
        learning_rate=custom_args.lr,
        do_train=custom_args.do_train,
        do_predict=custom_args.do_predict,
        report_to="wandb",
        run_name=f"lr_{custom_args.lr}_batch_{custom_args.batch_size}_epochs_{custom_args.num_train_epochs}",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,

    )
    return model_args, data_args, training_args, custom_args


def init_wandb(model_args, training_args):
    wandb.login(key=WANDB_API_KEY)
    wandb_run = wandb.init(
        project="anlp_ex1",
        name=training_args.run_name,
        config={
            "model_name": model_args.model_path,
            "dataset": "MRPC",
            "lr": training_args.learning_rate,
            "epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size,
        }
    )
    if model_args.model_path != "bert-base-uncased":
        artifact = wandb_run.use_artifact(model_args.model_path, type='model')
        model_args.model_path = artifact.download()
    return wandb_run


def log_res(epochs, lr, batch_size, eval_acc):
    file_path = 'res.txt'
    s = f"epoch_num: {epochs}, lr: {lr}, batch_size: {batch_size}, eval_acc:{eval_acc}"
    try:
        # Check if file exists
        try:
            with open(file_path, 'r') as f:
                # File exists, do nothing here
                pass
        except FileNotFoundError:
            # File doesn't exist, create it (empty file)
            with open(file_path, 'w') as f:
                pass

        # Now add the line to the file (appending)
        with open(file_path, 'a') as f:
            # Add a newline before writing if the file is not empty
            if f.tell() > 0:
                f.write('\n')
            f.write(s)

        return True

    except Exception as e:
        print(f"Error adding line to file: {e}")
        return False


def log_predictions(test_dataset, predicted_classes):
    lst = [f"{test_dataset[i]['sentence1']}###{test_dataset[i]['sentence2']}###{predicted_classes[i]}"
           for i in range(len(test_dataset))]
    s = "\n".join(lst)
    with open('predictions.txt', "w", encoding="utf-8") as f:
        f.write(s)


def argmax(logits):
    return [max(enumerate(logit), key=lambda x: x[1])[0] for logit in logits]


if __name__ == "__main__":
    model_args, data_args, training_args, custom_args = load_arguments()
    if not custom_args.do_train and not custom_args.do_predict:
        sys.exit(0)
    if custom_args.do_train:
        model_args.model_path = "bert-base-uncased"

    wandb_run = init_wandb(model_args, training_args)


    model, tokenizer = load_model_and_tokenizer(model_args)
    train_dataset, val_dataset, test_dataset = get_datasets(data_args, tokenizer)
    compute_metrics = define_metrics()

    train_data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=train_data_collator,
        callbacks=[CustomWandbCallback(wandb_run)]

    )

    if custom_args.do_train:
        if custom_args.log:
            print("start training")

        model.train()
        train_result = trainer.train()

        if custom_args.log:
            print("train metrics:")
            print(train_result)

        model.eval()

        if custom_args.log:
            print("start evaluating")

        eval_results = trainer.evaluate(eval_dataset=val_dataset)

        if custom_args.log:
            print("eval metrics:")
            print(eval_results)

        log_res(training_args.num_train_epochs, training_args.learning_rate,
                training_args.per_device_train_batch_size, eval_results['eval_accuracy'])

    if custom_args.do_predict:
        print("start predicting")

        model.eval()
        predicted_classes = []
        trainer.data_collator = NopadDataCollator(tokenizer)
        test_results = trainer.predict(test_dataset=test_dataset)
        if custom_args.log:
            print("test metrics:")
            print(test_results)
        # Log predictions
        predicted_classes = test_results.predictions.argmax(axis=-1)
        log_predictions(test_dataset, predicted_classes)
        wandb.finish()
