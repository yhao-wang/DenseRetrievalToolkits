from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.cuda import device_count
import os
from ..dataset.data_collator import EncodeCollator, QPCollator, EVCollator, PPCollator, RRCollator


class ExactMatch_dataloader:
    def __init__(self, data_args, dataset, tokenizer, neg_sampler, batch_size=[1, 1, 1], num_workers=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.neg_sampler = neg_sampler

    def _get_sampler(self, dataset, shuffle=False):
        if 'RANK' in os.environ and device_count() > 1:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        else:
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)
        return sampler

    def get_dataset(self):
        self.train_dataset, self.eval_dataset, self.test_dataset = self.dataset.load_train()

    def get_bm25dataloader(self, dataset):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size[0],
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=QPCollator(
                data_args=self.data_args,
                tokenizer=self.tokenizer,
                sampler=self.neg_sampler
            ),
            sampler=self._get_sampler(dataset, True)
        )

    def get_passage(self, key):
        self.get_dataset()
        dataset = getattr(self, key)
        data = []
        for sample in dataset:
            for t in ['positives', 'negatives']:
                for p in sample[t]:
                    data.append(p)
        dataset = ListDataset(data)
        return DataLoader(
            dataset,
            batch_size=self.batch_size[1],
            num_workers=self.num_workers,
            collate_fn=PPCollator(
                data_args=self.data_args,
                tokenizer=self.tokenizer,
            ),
            sampler=self._get_sampler(dataset),
        )

    def get_dataloader(self):
        if not hasattr(self, 'train_dataset'):
            self.get_dataset()
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size[0],
            shuffle=False,
            num_workers=self.num_workers, 
            collate_fn=QPCollator(
                    data_args=self.data_args,
                    tokenizer=self.tokenizer,
                    sampler=self.neg_sampler
                ),
            sampler=self._get_sampler(self.train_dataset, True)
            )
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size[1],
            shuffle=False,
            num_workers=self.num_workers, 
            collate_fn=EVCollator(
                    data_args=self.data_args,
                    tokenizer=self.tokenizer,
                    sampler=self.neg_sampler
                ),
            sampler=self._get_sampler(self.eval_dataset)
            )
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size[2],
            shuffle=False,
            num_workers=self.num_workers, 
            collate_fn=EVCollator(
                    data_args=self.data_args,
                    tokenizer=self.tokenizer,
                    sampler=self.neg_sampler
                ),
            sampler=self._get_sampler(self.test_dataset)
            )

        return train_dataloader, eval_dataloader, test_dataloader

    def get_rr_dataloader(self):
        if not hasattr(self, 'train_dataset'):
            self.get_dataset()
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size[0],
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=RRCollator(
                    data_args=self.data_args,
                    tokenizer=self.tokenizer,
                    sampler=self.neg_sampler
                ),
            sampler=self._get_sampler(self.train_dataset, True)
            )

        return train_dataloader

    def get_query_dataloader(self):
        self.dataset = self.dataset.load_query_data()
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size[0],
            shuffle=False,
            num_workers=self.num_workers, 
            collate_fn=EncodeCollator(
                self.tokenizer,
                padding='max_length',
                q_max_len=self.data_args.q_max_len
            ),
            sampler=self._get_sampler(self.dataset)
        )

    def get_corpus_dataloader(self, batch_size):
        self.dataset = self.dataset.load_corpus_data()
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers, 
            collate_fn=PPCollator(
                tokenizer=self.tokenizer,
                data_args=self.data_args,
            ),
            sampler=self._get_sampler(self.dataset)
        )
