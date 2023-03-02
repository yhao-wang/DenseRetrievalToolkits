from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.cuda import device_count
from ..dataset.data_collator import EncodeCollator, QPCollator


class Relevancy_dataloader:
    def __init__(self, data_args, dataset, tokenizer, neg_sampler, batch_size=[256, 256, 256], num_workers=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.neg_sampler = neg_sampler

    def _get_sampler(self, dataset, shuffle=False):
        if device_count() > 1:
            datset_sampler = DistributedSampler(dataset, shuffle=shuffle)
        else:
            if shuffle:
                datset_sampler = RandomSampler(dataset)
            else:
                datset_sampler = SequentialSampler(dataset)
        return datset_sampler

    def get_dataloader(self):
        self.train_dataset, self.eval_dataset, self.test_dataset = self.dataset.load_train()
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
            collate_fn=QPCollator(
                    data_args=self.data_args,
                    tokenizer=self.tokenizer,
                    sampler=self.neg_sampler
                ),
            sampler=self._get_sampler(self.eval_dataset),
            )
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size[2],
            shuffle=False,
            num_workers=self.num_workers, 
            collate_fn=QPCollator(
                    data_args=self.data_args,
                    tokenizer=self.tokenizer,
                    sampler=self.neg_sampler
                ),
            sampler=self._get_sampler(self.test_dataset)
            )

        return train_dataloader, eval_dataloader, test_dataloader

    def get_query_dataloader(self):
        self.dataset = self.dataset.load_query_data()
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size, 
            shuffle=self.shuffle, 
            num_workers=self.num_workers, 
            collate_fn=EncodeCollator(
                self.tokenizer,
                padding='max_length',
                q_max_len=self.data_args.q_max_len
            ),
            sampler=self._get_sampler(self.dataset)
        )

    def get_corpus_dataloader(self):
        self.dataset = self.dataset.load_corpus_data()
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size, 
            shuffle=self.shuffle, 
            num_workers=self.num_workers, 
            collate_fn=EncodeCollator(
                self.tokenizer,
                padding='max_length',
                p_max_len=self.data_args.p_max_len
            ),
            sampler=self._get_sampler(self.dataset)
        )
