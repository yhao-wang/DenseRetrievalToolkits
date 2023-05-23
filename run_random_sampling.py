# Run DPR with random negative sampling
import os
import sys
import datetime
from DRT.arguments import DataArguments, ModelArguments, TrainingArguments
from transformers import HfArgumentParser, AutoTokenizer, AutoConfig
from DRT.dataset.abstract_dataset import ExactMatchDataset, RelevancyDataset, EXACTMATCH_DATASET
from DRT.dataset.CorpusDataset import CorpusDataset
from DRT.dataloader.corpus_dataloader import Corpus_dataloader
from DRT.dataloader.exactmatch_dataloader import ExactMatch_dataloader
from DRT.dataloader.relevancy_dataloader import Relevancy_dataloader
from DRT.trainer.sampler import RandomSampleNegatives
from DRT.model.biencoder import DRModel
from DRT.trainer.trainer import Trainer
# from transformers.trainer import Trainer


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
        cache_dir=model_args.cache_dir
    )

    model = DRModel.build(
        model_args=model_args,
        data_args=data_args,
        train_args=training_args,
        cache_dir=model_args.cache_dir,
    )

    batch_size = [training_args.train_batch_size, training_args.eval_batch_size, training_args.test_batch_size]
    dataset = ExactMatchDataset(data_args, tokenizer, cache_dir=data_args.data_cache_dir or model_args.cache_dir) \
        if data_args.dataset in EXACTMATCH_DATASET else RelevancyDataset(data_args, tokenizer, cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    rnd_sampler = RandomSampleNegatives(data_args)
    corpus = CorpusDataset(data_args, tokenizer, data_args.data_cache_dir)
    dataloader = ExactMatch_dataloader(data_args, dataset, tokenizer, rnd_sampler, batch_size=batch_size) \
        if data_args.dataset in EXACTMATCH_DATASET else Relevancy_dataloader(data_args, dataset, tokenizer, rnd_sampler,
                                                                             batch_size=batch_size)
    train_dataloader, eval_dataloader, test_dataloader = dataloader.get_dataloader()
    corpus_dataloader = Corpus_dataloader(data_args, corpus, tokenizer, training_args.corpus_batch_size, num_labels).get_dataloder()
    trainer = Trainer(training_args, model, train_loader=train_dataloader, corpus_dataloader=corpus_dataloader,
                      eval_loader=eval_dataloader, test_loader=test_dataloader)
    trainer.train()


if __name__ == '__main__':
    import torch.distributed as dist
    dist.init_process_group(backend='nccl', init_method=None, timeout=datetime.timedelta(seconds=180000),
                                         world_size=- 1, rank=- 1, store=None, group_name='', pg_options=None)
    main()
