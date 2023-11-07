import argparse
import sys
from os.path import dirname, realpath
import os

sys.path.append(dirname(dirname(realpath(__file__))))
from src.lightning import MLP, CNN, Resnet, RiskModel
from src.dataset import PathMnist, NLST
from lightning.pytorch.cli import LightningArgumentParser
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import optuna
from optuna.integration import PyTorchLightningPruningCallback

NAME_TO_MODEL_CLASS = {
    "mlp": MLP,
    "cnn": CNN,
    "resnet": Resnet,
    "risk_model": RiskModel
}

NAME_TO_DATASET_CLASS = {
    "pathmnist": PathMnist,
    "nlst": NLST
}


def add_main_args(parser: LightningArgumentParser) -> LightningArgumentParser:

    parser.add_argument(
        "--model_name",
        default="resnet",
        help="Name of model to use. Options include: mlp, cnn, resnet",
    )

    parser.add_argument(
        "--dataset_name",
        default="pathmnist",
        help="Name of dataset to use. Options: pathmnist, nlst"
    )

    parser.add_argument(
        "--project_name",
        default="CPH 200A Project 2",
        help="Name of project for wandb"
    )

    parser.add_argument(
        "--monitor_key",
        default="val_acc",
        help="Name of metric to use for checkpointing. (e.g. val_loss, val_acc)"
    )

    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Path to checkpoint to load from. If None, init from scratch."
    )

    parser.add_argument(
        "--train",
        default=True,
        action="store_true",
        help="Whether to train the model."
    )

    parser.add_argument(
        "--depth",
        default=5,
        help="define the depth of MLP."
    )

    parser.add_argument(
        "--hidden_dim",
        default=1024,
        help="define the hidden dimension of MLP."
    )

    return parser

def parse_args() -> argparse.Namespace:
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(pl.Trainer, nested_key="trainer")
    for model_name, model_class in NAME_TO_MODEL_CLASS.items():
        parser.add_lightning_class_args(model_class, nested_key=model_name)
    for dataset_name, data_class in NAME_TO_DATASET_CLASS.items():
        parser.add_lightning_class_args(data_class, nested_key=dataset_name)
    parser = add_main_args(parser)
    args = parser.parse_args()
    return args

class OptunaPruning(PyTorchLightningPruningCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def objective(trial):
    exp_name = "trial_{}".format(trial.number)

    logger = pl.loggers.WandbLogger(project='Resnet', entity="cancer-busters", name=exp_name)

    trainer = pl.Trainer(
        logger=logger,
        accelerator='auto',
        precision='bf16-mixed',
        callbacks=[OptunaPruning(trial, monitor="val_acc")],
        max_epochs=20
        )
    
    model_args = dict()
    model_args['init_lr'] = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
    model_args['optimizer'] = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])
    model_args['dropout_p'] = trial.suggest_float('dropout_p', 0, 0.5)
    model_args['seed'] = trial.suggest_int('seed', 0, 3407)
    model_args['n_fc'] = trial.suggest_int('n_fc', 1, 5)

    model_args['pretrained'] = True
    model_args['trial'] = trial
    pl.seed_everything(model_args['seed'])

    model = Resnet(**model_args)

    batch_size = trial.suggest_int('batch_size', 64, 4096)
    datamodule = PathMnist(use_data_augmentation=True, batch_size=batch_size, num_workers=8)
    trainer.fit(model, datamodule)

    return trainer.callback_metrics["val_acc"].item()


def main(args: argparse.Namespace):
    print(args)
    print("Loading data ..")

    print("Preparing lighning data module (encapsulates dataset init and data loaders)")
    print(args[args.dataset_name])

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=1000, timeout=None)

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial

    print("  Best values: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == '__main__':
    __spec__ = None
    args = parse_args()
    main(args)
