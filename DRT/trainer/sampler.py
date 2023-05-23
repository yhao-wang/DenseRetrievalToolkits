import random
from torch.utils.data import Dataset

from ..arguments import DataArguments
from ..evaluator.index import BM25Retriever
from tqdm import tqdm
import os
import json
from multiprocessing import Pool, cpu_count


class BM25Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class RandomSampleNegatives:
    def __init__(self, data_args: DataArguments):
        self.num_negative = data_args.train_n_passages - 1
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len

    def __call__(self, samples):
        queries = []
        documents = []
        for sample in samples:
            document = []
            queries.append(sample['query'])
            document.append(random.choices(sample['positives'])[0])
            all_negative_num = len(sample['negatives'])
            assert all_negative_num >= self.num_negative, \
                f"The num_negative {self.num_negative} should be no larger than all_negative_num {all_negative_num}."
            cand_doc = list(range(all_negative_num))
            random.shuffle(cand_doc)
            cand_doc = cand_doc[:self.num_negative]
            for idx in cand_doc:
                document.append(sample['negatives'][idx])
            documents.append(document)

        return queries, documents


class BM25Negatives:
    def __init__(self, data_args: DataArguments, vocab_size: int):
        self.cache_dir = data_args.data_cache_dir
        self.num_negative = data_args.train_n_passages - 1
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.retriever = BM25Retriever(self.num_negative, vocab_size)

    def load_passages(self, corpus):
        out_dir = os.path.join(self.cache_dir, "BM25data")
        data_name = "bm25negatives"
        data = []
        if os.path.exists(os.path.join(out_dir, data_name)):
            f = open(os.path.join(out_dir, data_name), 'r', encoding='utf-8')
            for line in f.readlines():
                data.append(json.loads(line))
            f.close()
        else:
            bp, ep = self.retriever.load_passages(corpus)

            def process_sample(o):
                sample, b, e = o
                document = []
                neg_docs = self.retriever.search(sample['query'], self.num_negative + len(sample['positives']))
                for doc in neg_docs:
                    if doc >= b and doc < e:
                        continue
                    document.append(self.retriever.passage[doc])
                    if len(document) == self.num_negative:
                        break
                sample['negatives'] = document
                return sample

            with Pool() as pool:
                data = pool.map(process_sample, zip(corpus, bp, ep))

            # for i, sample in tqdm(enumerate(corpus), 'Building Dataset'):
            #     document = []
            #     neg_docs = self.retriever.search(sample['query'], self.num_negative + len(sample['positives']))
            #     for doc in neg_docs:
            #         if doc >= bp[i] and doc < ep[i]:
            #             continue
            #         document.append(self.retriever.passage[doc])
            #         if len(document) == self.num_negative:
            #             break
            #     sample['negatives'] = document
            #     data.append(sample)


                self.save(data, out_dir, data_name)
        return ListDataset(data)

    def save(self, data, out_dir, data_name):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_file = os.path.join(out_dir, data_name)
        f = open(out_file, 'w', encoding='utf-8')
        for sample in data:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')
        f.close()

    def __call__(self, samples):
        queries = []
        documents = []
        for sample in samples:
            document = []
            queries.append(sample['query'])
            document.append(random.choices(sample['positives'])[0])
            neg_docs = self.retriever.search(sample['query'], self.num_negative + len(sample['positives']))
            iter_bar = tqdm(neg_docs, desc='BM25 Retrieving')
            for doc in iter_bar:
                if doc not in sample['positives']:
                    continue
                document.append(doc)
                if len(document) == self.num_negative:
                    break
            documents.append(document)
        return queries, documents
