import numpy as np
import faiss
import math
import torch
from typing import List
import logging
from tqdm import tqdm

from random import randint
import os
import torch.distributed as dist

logger = logging.getLogger(__name__)


class BaseFaissIPRetriever:
    def __init__(self, init_reps):
        if isinstance(init_reps, np.ndarray):
            index = faiss.IndexFlatIP(init_reps.shape[1])
        elif init_reps is None:
            index = None
        else:
            index = faiss.IndexFlatIP(init_reps)
        self.index = index
        self.docid = []

    def add(self, p_reps: np.ndarray):
        self.index.add(p_reps)


    def search(self, q_reps: np.ndarray, k: int = 1000):
        scores, indices = self.index.search(q_reps, k)
        return np.array([indice[x] for indice, x in zip(indices, np.argsort(-scores))])

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int, quiet: bool=False):
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        for start_idx in tqdm(range(0, num_query, batch_size), disable=quiet):
            nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        all_indices = np.concatenate(all_indices, axis=0)
        return all_indices


class FaissRetriever(BaseFaissIPRetriever):

    def __init__(self, init_reps: np.ndarray, factory_str: str):
        index = faiss.index_factory(init_reps.shape[1], factory_str)
        self.index = index
        self.index.verbose = True
        if not self.index.is_trained:
            self.index.train(init_reps)


class BM25Retriever(BaseFaissIPRetriever):
    def __init__(self, topK: int, vocab_size: int):
        self.topK = topK
        self.eps = 0.25
        self.k1 = 1.2  # Control term frequency saturation
        self.b = 0.75  # Control document length normalization
        self.idf = {}  # 记录查询词的 IDF
        self.docContainedWord = {}  # 包含单词 word 的文档集合
        self.vocab_size = vocab_size
        self.passage = []
        self.cnt = None

    def load_passages(self, corpus: List[List[int]]):
        # Set BM25 parameters
        # Initialize the corpus to calculate the various coefficients of BM25
        # Calculate how many documents each word appears in the corpus
        bp = []
        ep = []
        iter_bar = tqdm(corpus, desc="Loading corpus")
        for samples in iter_bar:
            bp.append(len(self.passage))
            for key in ['positives', 'negatives']:
                for p in samples[key]:
                    self.passage.append(p)
                if key == 'positives':
                    ep.append(len(self.passage))

        corpus_size = len(self.passage)

        del corpus
        self.cnt = [{}] * corpus_size

        iter_bar = tqdm(self.passage, desc="Calculating IDF 1st phrase")
        for i, doc in enumerate(iter_bar):
            for word in doc:
                if word not in self.docContainedWord:
                    self.docContainedWord[word] = set()
                self.docContainedWord[word].add(i)
                if word in self.cnt[i].keys():
                    self.cnt[i][word] += 1
                else:
                    self.cnt[i][word] = 1

        idf_sum = 0.  # collect idf sum to calculate an average idf for epsilon value
        negative_idfs = []
        iter_bar = tqdm(self.docContainedWord.items(), desc="Calculating IDF 2nd phrase")
        for word, doc_ids in iter_bar:
            doc_nums_contained_word = len(doc_ids)
            idf = math.log(corpus_size - doc_nums_contained_word +
                           0.5) - math.log(doc_nums_contained_word + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)

        average_idf = float(idf_sum) / len(self.idf)
        eps = self.eps * average_idf
        for word in negative_idfs:
            self.idf[word] = eps

        # Calculate the length and average length of each document in the corpus
        self.avg_doc_len = sum([len(doc) for doc in self.passage]) / corpus_size
        return bp, ep

    def search(self, q_reps: np.ndarray, k: int = 1000):
        score = {}
        for word in q_reps:
            for id in self.docContainedWord[word]:
                if id not in score.keys():
                    score[id] = 0
                score[id] += self.idf[word] * self.cnt[id][word] * (1 + self.k1) / \
                             (self.cnt[id][word] + 1 - self.b + self.b * len(self.passage[id]) / self.avg_doc_len)
        kl, vl = [], []
        for k, v in score.items():
           kl.append(k)
           vl.append(v)
        while len(kl) < k:
            x = randint(0, len(self.passage))
            if x not in k:
                kl.append(x)
                vl.append(0)

        s, i = torch.topk(torch.tensor(vl), k)
        return [kl[x] for x in i]

    def retrieve(self, query: torch.Tensor, documents: torch.Tensor):
        # Return the topK result index according to query and BM25 retrieval algorithm, return type is Tensor(List[int])

        # Store the score of each document
        scores = []
        for i, doc in enumerate(documents):
            # Initialize document score to 0
            doc_freqs = {}
            doc_len = len(doc)
            for word in doc:
                if word not in doc_freqs:
                    doc_freqs[word] = 0
                doc_freqs[word] += 1
            score = 0
            for word in query:
                if word in doc:
                    score += self.idf[word] * doc_freqs[word] * (self.k1 + 1) / (
                            doc_freqs[word] + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len))
            scores.append(score)

        scores_tensor = torch.tensor(scores)  # Convert score list to tensor type

        # Return topk highest scores and their corresponding indices
        topk_scores, topk_indices = torch.topk(scores_tensor, k=self.topK)

        return topk_indices