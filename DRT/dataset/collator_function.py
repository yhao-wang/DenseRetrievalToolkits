import collections
import random
import transformers.trainer


class RandomSampleNegatives:
    def __init__(self, config):
        self.num_negative = config.num_negative

    def __call__(self, samples):
        queries = []
        documents = []
        for sample in samples:
            document = []
            queries.append(sample['query'])
            document.append(sample['positives'])
            all_negative_num = len(sample['negatives'])
            assert all_negative_num >= self.num_negative, \
                f"The num_negative {self.num_negative} should be no larger than all_negative_num {all_negative_num}."
            cand_doc = range(all_negative_num)
            random.shuffle(cand_doc)
            cand_doc = cand_doc[:self.num_negative]
            for idx in cand_doc:
                document.append(sample['negatives'][idx])
            documents.append(document)
        return {'query': queries}, {'passage': documents}


collect_fn_class = collections.defaultdict(
            lambda: RandomSampleNegatives, {
                'random_sample': RandomSampleNegatives,
            }
)


class AbstractCollate:

    def __init__(self, config):
        self.collect_fn = collect_fn_class[config['collect_fn']](config)

    def __call__(self, samples):
        return self.collect_fn(samples)
