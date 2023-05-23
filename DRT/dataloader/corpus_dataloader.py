from ..dataset.CorpusDataset import get_corpus_dataset
from ..dataset.data_collator import PPCollator
from torch.cuda import device_count
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import os


class Corpus_dataloader:
    def __init__(self, data_args, dataset, tokenizer, batch_size=8, num_workers=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_args = data_args
        self.tokenizer = tokenizer

    def _get_sampler(self, dataset, shuffle=False):
        if 'RANK' in os.environ and device_count() > 1:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        else:
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)
        return sampler

    def get_dataloder(self):
        self.dataset = self.dataset.load_dataset()
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=PPCollator(
                data_args=self.data_args,
                tokenizer=self.tokenizer,
            ),
            sampler=self._get_sampler(self.dataset, False)
        )
