from .abstract_dataset import AbstractDataset
from datasets import load_dataset
from .preprocess import RREVPreProcessor
import os


class RRDataset(AbstractDataset):
    def __init__(
            self,
            data_args,
            tokenizer,
            cache_dir,
    ):
        data_path = os.path.join(cache_dir, 'retrieve')
        self.cache_dir = os.path.join(cache_dir, 'reranker_eval')
        file_list = os.listdir(data_path)
        file_list = [os.path.join(data_path, file) for file in file_list]
        self.dataset = load_dataset('json', data_files=file_list, cache_dir=self.cache_dir)['train']
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.has_load_train = False

    def load_dataset(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        self.dataset = self.dataset.map(
            RREVPreProcessor(self.tokenizer, self.q_max_len, self.p_max_len),
            batched=False,
            num_proc=self.proc_num,
            remove_columns=self.dataset.column_names,
            desc="Running tokenizer on evaluation corpus",
        )
        return self.dataset
