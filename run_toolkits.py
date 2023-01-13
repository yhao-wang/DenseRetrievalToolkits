import os
import sys
from DRT.trainer.trainer import Trainer
from DRT.arguments import DataArguments, ModelArguments, TrainingArguments
from transformers import HfArgumentParser, AutoTokenizer, AutoConfig
from DRT.dataset.data_collator import QPCollator
from DRT.dataset.abstract_dataset import ExactMatchDataset
from DRT.dataloader.exactmatch_dataloader import ExactMatch_dataloader
from DRT.trainer.sampler import RandomSampleNegatives
from DRT.model.biencoder import DRModel
from transformers import Trainer


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    num_labels = 1

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )

    model = DRModel.build(
        model_args=model_args,
        data_args=data_args,
        train_args=training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    
    train_dataset = ExactMatchDataset(data_args, tokenizer, cache_dir=data_args.data_cache_dir or model_args.cache_dir, set="train")
    rnd_sampler = RandomSampleNegatives(data_args)
    data_collator = QPCollator(
            data_args=data_args,
            tokenizer=tokenizer,
            sampler=rnd_sampler
        )
    batch_size = training_args.train_batch_size
    exact_dataloader = ExactMatch_dataloader(train_dataset, data_collator, batch_size=128)
    train_dataloader = exact_dataloader.get_train_dataloader()

    trainer = Trainer(training_args, model, train_dataloader, eval_loader=None, test_loader=None)
    trainer.train()
    trainer.save_model()


if __name__=='__main__':
    main()