from trainer.trainer import Trainer


def main():

    trainer = Trainer(config, model, train_set=train_set, eval_set=eval_set, test_set=test_set)