from torch.utils.data import DataLoader

class ExactMatch_dataloader:
    def __init__(self, dataset, data_collater, batch_size=1, shuffle=False, num_workers=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.data_collater = data_collater
    
    def get_train_dataloader(self):
        return DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle, 
            num_workers=self.num_workers, 
            collate_fn=self.data_collater
            )

    def get_query_dataloader(self):
        return DataLoader(
            self.dataset.load_query_data(),
            batch_size=self.batch_size, 
            shuffle=self.shuffle, 
            num_workers=self.num_workers, 
            collate_fn=self.data_collater
            )

    def get_corpus_dataloader(self):
        return DataLoader(
            self.dataset.load_corpus_data(),
            batch_size=self.batch_size, 
            shuffle=self.shuffle, 
            num_workers=self.num_workers, 
            collate_fn=self.data_collater
        )
