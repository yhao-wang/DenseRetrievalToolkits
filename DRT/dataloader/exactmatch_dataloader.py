from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.cuda import device_count
from ..dataset.data_collator import EncodeCollator, QPCollator


class ExactMatch_dataloader:
    def __init__(self, data_args, dataset, tokenizer, neg_sampler, batch_size=1, shuffle=False, num_workers=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.neg_sampler = neg_sampler

    def get_sampler(self):
        if device_count() > 1:
            self.sampler = DistributedSampler(self.dataset, shuffle=shuffle)
        else:
            if self.shuffle:
                self.sampler = RandomSampler(self.dataset)
            else:
                self.sampler = SequentialSampler(self.dataset)

    def get_train_dataloader(self):
        self.dataset = self.dataset.load_train()
        self.get_sampler()
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size, 
            shuffle=self.shuffle, 
            num_workers=self.num_workers, 
            collate_fn=QPCollator(
                    data_args=self.data_args,
                    tokenizer=self.tokenizer,
                    sampler=self.neg_sampler
                ),
            sampler=self.sampler
            )

    def get_query_dataloader(self):
        self.dataset = self.dataset.load_query_data()
        self.get_sampler()
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
            sampler=self.sampler
        )

    def get_corpus_dataloader(self):
        self.dataset = self.dataset.load_corpus_data()
        self.get_sampler()
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
            sampler=self.sampler
        )
