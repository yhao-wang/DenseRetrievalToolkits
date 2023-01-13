import random

from arguments import DataArguments


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

