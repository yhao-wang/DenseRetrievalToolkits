from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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
        self.sampler = DistributedSampler(self.dataset, shuffle=shuffle)
        # self.data_collater = data_collater
    
    def get_train_dataloader(self):
        return DataLoader(
            self.dataset.load_train(), 
            batch_size=self.batch_size, 
            shuffle=self.shuffle, 
            num_workers=self.num_workers, 
            collate_fn=QPCollator(
                    data_args=self.data_args,
                    tokenizer=self.tokenizer,
                    sampler=self.neg_sampler
                )
            )

    def get_query_dataloader(self):
        return DataLoader(
            self.dataset.load_query_data(),
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
        return DataLoader(
            self.dataset.load_corpus_data(),
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
