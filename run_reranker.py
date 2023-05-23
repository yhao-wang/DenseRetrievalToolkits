import os
import sys
import datetime

from DRT.arguments import DataArguments, ModelArguments
from transformers import HfArgumentParser, AutoTokenizer, AutoConfig
from DRT.arguments import RRTrainingArguments as TrainingArguments
from DRT.dataset.reranker_dataset import RRDataset
from DRT.dataset.abstract_dataset import ExactMatchDataset
from DRT.dataloader.exactmatch_dataloader import ExactMatch_dataloader
from DRT.dataloader.reranker_dataloader import Reranker_dataloader
from DRT.trainer.sampler import RandomSampleNegatives
from DRT.model.reranker import RRModel
from DRT.trainer.trainer import RRTrainer


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
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    model = RRModel.build(
        model_args=model_args,
        tokenizer=tokenizer,
        train_args=training_args,
        cache_dir=model_args.cache_dir,
    )

    batch_size = [training_args.train_batch_size, training_args.eval_batch_size, training_args.test_batch_size]
    dataset = ExactMatchDataset(data_args, tokenizer, cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    rnd_sampler = RandomSampleNegatives(data_args)
    dataloader = ExactMatch_dataloader(data_args, dataset, tokenizer, rnd_sampler, batch_size=batch_size)
    train_dataloader = dataloader.get_rr_dataloader()
    eval_dataset = RRDataset(data_args, tokenizer, data_args.data_cache_dir)
    eval_dataloader = Reranker_dataloader(data_args, eval_dataset, tokenizer, batch_size=training_args.eval_batch_size).get_eval_dataloader()

    trainer = RRTrainer(training_args, model, train_loader=train_dataloader, eval_loader=eval_dataloader)
    # trainer.train()
    trainer.evaluate(eval_dataloader, 3)


if __name__ == '__main__':
    import torch.distributed as dist
    dist.init_process_group(backend='nccl', init_method=None, timeout=datetime.timedelta(seconds=180000),
                                         world_size=- 1, rank=- 1, store=None, group_name='', pg_options=None)
    main()
