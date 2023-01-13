import json
from datasets import load_dataset
from torch.utils.data import Dataset
import tqdm
from transformers import PreTrainedTokenizer

from .preprocess import RelevancyPreProcessor, ExactMatchPreProcessor, QueryPreProcessor, CorpusPreProcessor, TrainPreProcessor
from ..arguments import DataArguments

import logging
logger = logging.getLogger(__name__)


RELEVANCY_DATASET = ["MSMARCO"]
EXACTMATCH_DATASET = ["NQ", "WQ", "TQ", "Squad"]


class AbstractDataset(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            tokenizer: PreTrainedTokenizer,
            cache_dir: str,
    ):
        self.set = set
        self.cache_dir = cache_dir
        self.dataset = load_dataset(data_args.dataset_name, data_files=data_args.data_path, cache_dir=self.cache_dir)
        self.train_dataset = self.dataset["train"]
        self.valid_dataset = self.dataset["dev"]
        self.test_dataset = self.dataset["test"]        
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.neg_num = data_args.train_n_passages - 1
        self.separator = getattr(self.tokenizer, data_args.passage_field_separator, data_args.passage_field_separator)
        # self.docid2text = self.load_id_text()

    def __len__(self):
        return len(self.dataset)

    def load_query_data(self, shard_num=1, shard_idx=0):
        self.preprocessor = QueryPreProcessor
        self.test_dataset = self.test_dataset.shard(shard_num, shard_idx)
        query_data = self.test_dataset.map(
            self.preprocessor(self.tokenizer, self.q_max_len),
            batched=False,
            num_proc=self.proc_num,
            remove_columns=self.test_dataset.column_names,
            desc="Running tokenization",
        )
        return query_data

    def load_corpus_data(self, shard_num=1, shard_idx=0):
        self.corpus = load_dataset(self.data_args.corpus_name, data_files=self.data_args.corpus_path, cache_dir=self.cache_dir)["train"]
        self.preprocessor = CorpusPreProcessor
        self.corpus = self.corpus.shard(shard_num, shard_idx)
        corpus_data = self.corpus.map(
            self.preprocessor(self.tokenizer, self.p_max_len),
            batched=False,
            num_proc=self.proc_num,
            remove_columns=self.corpus.column_names,
            desc="Running tokenization",
            )
        return corpus_data

    def load_id_text(self):
        print("Mapping docid to text...(it may take a few minutes)")
        corpus_data = self.load_corpus_data()
        id_text = {}
        for c in tqdm(corpus_data):
            id_text[c["docid"]] = c["text"]
        # with open(self.data_args.corpus_path,encoding='utf8') as f:
        #     data = [json.loads(line) for line in f]
        #     for c in data:
        #         id_text[c["docid"]] = c["text"]
                
        return id_text

    # id2text and tokenize
    def __getitem__(self, index):
        return self.dataset[index]
        # sample = self.dataset[index]
        # query = sample["query"]
        # query_ids = self.tokenizer.encode(query,
        #                               add_special_tokens=False,
        #                               max_length=self.q_max_len,
        #                               truncation=True)
        # positive_passages = [self.docid2text[pid] if pid in self.docid2text else " " for pid in sample["pos_doc_ids"]]
        # negative_passages = [self.docid2text[nid] if nid in self.docid2text else " " for nid in sample["neg_doc_ids"]]
        # positive_passages_ids = []
        # for pos in positive_passages:
        #     positive_passages_ids.append(self.tokenizer.encode(pos,
        #                                            add_special_tokens=False,
        #                                            max_length=self.p_max_len,
        #                                            truncation=True))
        # negative_passages_ids = []
        # for neg in negative_passages:
        #     negative_passages_ids.append(self.tokenizer.encode(neg,
        #                                            add_special_tokens=False,
        #                                            max_length=self.p_max_len,
        #                                            truncation=True))
        
        # return {'query': query_ids, 'positives': positive_passages_ids, 'negatives': negative_passages_ids}

    def load_train(self, shard_num=1, shard_idx=0):
        self.train_dataset = self.train_dataset.shard(shard_num, shard_idx)
        self.preprocessor=TrainPreProcessor
        self.train_dataset = self.train_dataset.map(
            self.preprocessor(self.tokenizer, self.q_max_len, self.p_max_len, self.separator),
            batched=False,
            num_proc=self.proc_num,
            remove_columns=self.train_dataset.column_names,
            desc="Running tokenizer on train dataset",
        )
        return self.train_dataset

    # TODO 
    def collect_batch(self):
        pass


class RelevancyDataset(AbstractDataset):
    def __init__(
            self,
            data_args: DataArguments,
            tokenizer: PreTrainedTokenizer,
            cache_dir: str,
            set: str
    ):
        super().__init__(data_args, tokenizer, cache_dir, set)
    
    def process(self, shard_num=1, shard_idx=0):
        self.train_dataset = self.train_dataset.shard(shard_num, shard_idx)
        self.preprocessor = RelevancyPreProcessor
        answerset = self.train_dataset.map(
            self.preprocessor(self.tokenizer, self.q_max_len, self.p_max_len, self.separator),
            batched=False,
            num_proc=self.proc_num,
            remove_columns=self.train_dataset.column_names,
            desc="Running tokenizer on train dataset",
        )
        return answerset
    
    def load_query_data(self, shard_num=1, shard_idx=0):
        self.preprocessor = QueryPreProcessor
        self.test_dataset = self.test_dataset.shard(shard_num, shard_idx)
        query_data = self.test_dataset.map(
            self.preprocessor(self.tokenizer, self.q_max_len),
            batched=False,
            num_proc=self.proc_num,
            remove_columns=self.test_dataset.column_names,
            desc="Running tokenization",
        )
        return query_data
    
    def load_corpus_data(self, shard_num=1, shard_idx=0):
        self.corpus = load_dataset(self.data_args.corpus_name, data_files=self.data_args.corpus_path, cache_dir=self.cache_dir)["train"]
        self.preprocessor = CorpusPreProcessor
        self.corpus = self.corpus.shard(shard_num, shard_idx)
        corpus_data = self.corpus.map(
            self.preprocessor(self.tokenizer, self.p_max_len),
            batched=False,
            num_proc=self.proc_num,
            remove_columns=self.corpus.column_names,
            desc="Running tokenization",
            )
        return corpus_data


class ExactMatchDataset(AbstractDataset):
    def __init__(
            self,
            data_args: DataArguments,
            tokenizer: PreTrainedTokenizer,
            cache_dir: str,
    ):
        super().__init__(data_args, tokenizer, cache_dir)
    
    def process(self, shard_num=1, shard_idx=0):
        self.train_dataset = self.train_dataset.shard(shard_num, shard_idx)
        self.preprocessor = ExactMatchPreProcessor
        answerset = self.train_dataset.map(
            self.preprocessor(self.tokenizer, self.q_max_len, self.p_max_len, self.separator),
            batched=False,
            num_proc=self.proc_num,
            remove_columns=self.train_dataset.column_names,
            desc="Running tokenizer on train dataset",
        )
        return answerset

    def load_query_data(self, shard_num=1, shard_idx=0):
        self.preprocessor = QueryPreProcessor
        self.test_dataset = self.test_dataset.shard(shard_num, shard_idx)
        query_data = self.test_dataset.map(
            self.preprocessor(self.tokenizer, self.q_max_len),
            batched=False,
            num_proc=self.proc_num,
            remove_columns=self.test_dataset.column_names,
            desc="Running tokenization",
        )
        return query_data
    
    def load_corpus_data(self, shard_num=1, shard_idx=0):
        self.corpus = load_dataset(self.data_args.corpus_name, data_files=self.data_args.corpus_path, cache_dir=self.cache_dir)["train"]
        self.preprocessor = CorpusPreProcessor
        self.corpus = self.corpus.shard(shard_num, shard_idx)
        corpus_data = self.corpus.map(
            self.preprocessor(self.tokenizer, self.p_max_len),
            batched=False,
            num_proc=self.proc_num,
            remove_columns=self.corpus.column_names,
            desc="Running tokenization",
        )
        return corpus_data
