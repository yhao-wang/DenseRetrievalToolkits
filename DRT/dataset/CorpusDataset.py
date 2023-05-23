import os

from .abstract_dataset import AbstractDataset
from datasets import load_dataset
from .preprocess import DocPreProcessor


class CorpusDataset(AbstractDataset):
    def __init__(
            self,
            data_args,
            tokenizer,
            cache_dir,
    ):
        self.cache_dir = cache_dir
        self.dataset = load_dataset('json', data_files=os.path.join(cache_dir, 'wiki/corpus.json'), cache_dir=self.cache_dir)['train']
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num

    def load_dataset(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        self.dataset = self.dataset.map(
            DocPreProcessor(self.tokenizer, self.p_max_len),
            batched=False,
            num_proc=self.proc_num,
            remove_columns=self.dataset.column_names,
            desc="Running tokenizer on corpus",
        )
        return self.dataset


def get_corpus_dataset(dir="/home/wangyuhao/DRT_cache/wiki/para"):
    text = []
    with open(dir, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            i, t = line.split('\t')
            text.append(t)
        f.close()
    return CorpusDataset(text)
