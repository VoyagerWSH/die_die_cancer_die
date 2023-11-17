import argparse
import sys
from os.path import dirname, realpath

sys.path.append(dirname(dirname(realpath(__file__))))
from src.lightning import MLP, CNN, Resnet, CNN_3D, RiskModel
from src.dataset import PathMnist, NLST
from lightning.pytorch.cli import LightningArgumentParser
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

NAME_TO_MODEL_CLASS = {
    "mlp": MLP,
    "cnn": CNN,
    "resnet": Resnet,
    "cnn_3d": CNN_3D,
    "risk_model": RiskModel
}

NAME_TO_DATASET_CLASS = {
    "pathmnist": PathMnist,
    "nlst": NLST
}


def add_main_args(parser: LightningArgumentParser) -> LightningArgumentParser:

    parser.add_argument(
        "--model_name",
        default="cnn_3d",
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
        default="val_auc",
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


def main(args: argparse.Namespace):
    print(args)
    print("Loading data ..")

    print("Preparing lighning data module (encapsulates dataset init and data loaders)")
    """
        Most the data loading logic is pre-implemented in the LightningDataModule class for you.
        However, you may want to alter this code for special localization logic or to suit your risk
        model implementations
    """
    pl.seed_everything(1312)
    datamodule = NAME_TO_DATASET_CLASS[args.dataset_name](**vars(args[args.dataset_name]))

    print("Initializing model")

    if args.model_name == "mlp":
        args[args.model_name]['layers'] = [28*28*3] + int(args.depth)*[int(args.hidden_dim)] + [9]
        exp_name = "MLP_layers=" + str(args.depth) + "_hidden_dim=" + str(args.hidden_dim)
        args[args.model_name]['init_lr'] = 1e-5
        args[args.model_name]['optimizer'] = "AdamW"
    elif args.model_name == "cnn":
        args[args.model_name]['conv_layers'] = [3, 6, 12, 24]
        args[args.model_name]['pooling'] = "max"
        # args[args.model_name]['optimizer'] = "AdamW"
        args[args.model_name]['init_lr'] = 1e-5
        exp_name = "CNN_convLayers=" + str(len(args[args.model_name]['conv_layers'])) + "_LR=" + str(args[args.model_name]['init_lr']) + "_opti=" + args[args.model_name]['optimizer']
    elif args.model_name == "resnet":
        args[args.model_name]['init_lr'] = 0.0006027654386720487
        args[args.model_name]['optimizer'] = "AdamW"
        args[args.model_name]['pre_train'] = True
        exp_name = "Resnet_pretrain=" + str(args[args.model_name]['pre_train']) + "_convLayers=18_fc=2" + "_LR=" + str(args[args.model_name]['init_lr']) + "_opti=" + args[args.model_name]['optimizer']
    elif args.model_name == "cnn_3d":
        args[args.model_name]['init_lr'] = 1e-4
        args[args.model_name]['optimizer'] = "AdamW"
        args[args.model_name]['pooling'] = "max"
        args[args.model_name]['conv_layers'] = [1, 3, 6, 12, 24]
        exp_name = "3D CNN_convLayers=" + str(len(args[args.model_name]['conv_layers'])) + "_LR=" + str(args[args.model_name]['init_lr']) + "_opti=" + args[args.model_name]['optimizer']

    if args.checkpoint_path is None:
        model = NAME_TO_MODEL_CLASS[args.model_name](**vars(args[args.model_name]))
    else:
        model = NAME_TO_MODEL_CLASS[args.model_name].load_from_checkpoint(args.checkpoint_path)

    print("Initializing trainer")
    logger = pl.loggers.WandbLogger(project=args.project_name, entity="cancer-busters", name=exp_name)
    # logger = pl.loggers.WandbLogger(project=args.project_name, entity="cancer-busters", name=exp_name, mode="disabled")

    args.trainer.accelerator = 'auto' ## “cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “mps”, or “auto”
    args.trainer.logger = logger
    args.trainer.precision = "bf16-mixed" ## This mixed precision training is highly recommended
    args.trainer.max_epochs = 200
    args.trainer.num_nodes = 1 ## Number of GPU nodes for distributed training

    args.trainer.callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor=args.monitor_key,
            mode='min' if "loss" in args.monitor_key else "max",
            save_last=True,
            dirpath = "checkpoints/" + exp_name
        )]
        # EarlyStopping(
        #     monitor=args.monitor_key,
        #     mode='min' if "loss" in args.monitor_key else "max"
        # )]

    trainer = pl.Trainer(**vars(args.trainer))

    if args.train:
        print("Training model")
        trainer.fit(model, datamodule)

    print("Best model checkpoint path: ", trainer.checkpoint_callback.best_model_path)

    print("Evaluating model on validation set")
    trainer.validate(model, datamodule)

    print("Evaluating model on test set")
    trainer.test(model, datamodule)

    print("Done")


if __name__ == '__main__':
    __spec__ = None
    args = parse_args()
    main(args)
