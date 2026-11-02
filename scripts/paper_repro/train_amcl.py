"""Run from the AMCL repository root with its modules on PYTHONPATH."""
from pathlib import Path

import torch

from DataLoad import data_load
from losses import FocalDiceLoss, ResampleLoss, gain_class_freq_and_train_num
from models.TextCNN import TextCNN
from pep_main import get_config
from train import CosineScheduler, DataTrain
from train_test import TrainAndTest


def main():
    args = get_config()
    output = Path(args.model_path)
    if output.exists():
        raise FileExistsError(f"Choose a new -model_path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    train_loaders, _, _, _ = data_load(
        train_direction=args.train_direction, test_direction=args.test_direction,
        batch=args.batch_size, subtest=False, CV=False,
    )
    model = TextCNN(args.vocab_size, args.embedding_size, args.filter_num,
                    args.filter_size, args.output_size, args.dropout).cuda()
    freq, count = gain_class_freq_and_train_num()
    losses = [FocalDiceLoss(), ResampleLoss(class_freq=freq, train_num=count,
                                            reweight_func="rebalance")]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = CosineScheduler(args.epochs, base_lr=args.learning_rate)
    trainer = DataTrain(model, optimizer, losses, scheduler=scheduler,
                        loss_weights=args.loss_weights)
    trainer.train_step(train_loaders[0], epochs=args.epochs, model_name="AMCL")
    torch.save(model.state_dict(), output)
    TrainAndTest(args)


if __name__ == "__main__":
    main()
