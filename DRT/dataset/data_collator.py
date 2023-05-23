from dataclasses import dataclass
from typing import List
from transformers import DataCollatorWithPadding, DefaultDataCollator


def create_one_example(text_encoding: List[int], tokenizer, q_max_len=None, p_max_len=None):
    item = tokenizer.prepare_for_model(
        text_encoding,
        truncation='only_first',
        max_length=q_max_len if q_max_len else p_max_len,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return item


@dataclass
class EVCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    def __init__(self, data_args, sampler, tokenizer):
        super().__init__(tokenizer=tokenizer)
        self.data_args = data_args
        self.sampler = sampler
        self.max_q_len = data_args.q_max_len
        self.max_p_len = data_args.p_max_len

    def __call__(self, features):
        qq = []
        qid = []
        ans = []
        qt = []

        for sample in features:
            qq.append(sample['query'])
            ans.append(sample['answers'])
            qid.append(sample['query_id'])
            qt.append(sample['original'])

        enq = []
        for q in qq:
            enq.append(create_one_example(q, self.tokenizer, q_max_len=self.max_q_len))

        q_collated = self.tokenizer.pad(
            enq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )

        return qid, q_collated, ans, qt


@dataclass
class EVRRCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    def __init__(self, data_args, tokenizer):
        super().__init__(tokenizer=tokenizer)
        self.data_args = data_args
        self.max_q_len = data_args.q_max_len
        self.max_p_len = data_args.p_max_len

    def create_one_example(self, query_encoding: List[int], text_encoding: List[int]):
        item = self.tokenizer.prepare_for_model(
            query_encoding,
            text_encoding,
            truncation='only_first',
            max_length=self.max_q_len + self.max_p_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __call__(self, features):
        qq = []
        ans = []
        doc = []
        dd = []
        pair = []
        qid = []
        did = []

        for sample in features:
            qid.append(sample['query_id'])
            qq.append(sample['query'])
            did.append(sample['doc_id'])
            ans.append(sample['answers'])
            doc.append(sample['original'])
            dd.append(sample['document'])

        for i, (q, d) in enumerate(zip(qq, dd)):
            pair.append(self.create_one_example(q, d))

        pair_collated = self.tokenizer.pad(
            pair,
            padding='max_length',
            max_length=self.max_q_len + self.max_p_len,
            return_tensors="pt",
        )

        return qid, pair_collated, ans, doc, did


@dataclass
class QPCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    def __init__(self, data_args, sampler, tokenizer):
        super().__init__(tokenizer=tokenizer)
        self.data_args = data_args
        self.sampler = sampler
        self.max_q_len = data_args.q_max_len
        self.max_p_len = data_args.p_max_len

    def __call__(self, features):
        sampled_features = self.sampler(features)
        qq = sampled_features[0]
        dd = sampled_features[1]

        enq = []
        end = []
        for q in qq:
            enq.append(create_one_example(q, self.tokenizer, q_max_len=self.max_q_len))
        for s in dd:
            one_sample = []
            for d in s:
                one_sample.append(create_one_example(d, self.tokenizer, p_max_len=self.max_p_len))
            end.append(one_sample)

        if isinstance(end[0], list):
            end = sum(end, [])
        q_collated = self.tokenizer.pad(
            enq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            end,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )

        return q_collated, d_collated


@dataclass
class PPCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    def __init__(self, data_args, tokenizer):
        super().__init__(tokenizer=tokenizer)
        self.data_args = data_args
        self.max_q_len = data_args.q_max_len
        self.max_p_len = data_args.p_max_len

    def __call__(self, features):

        dd = []
        did = []

        for sample in features:
            dd.append(sample['text'])
            did.append(sample['id'])

        enp = []
        for d in dd:
            enp.append(create_one_example(d, self.tokenizer, p_max_len=self.max_p_len))

        p_collated = self.tokenizer.pad(
            enp,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )

        return did, p_collated


@dataclass
class EncodeCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer, padding, q_max_len=None, p_max_len=None):
        super().__init__(tokenizer=tokenizer, padding=padding)
        self.q_max_len = q_max_len
        self.p_max_len = p_max_len

    def __call__(self, features):
        text_ids = [x['query_id'] if 'query_id' in x else x['doc_id'] for x in features]
        text_features = [x['query'] if 'query' in x else x['text'] for x in features]
        encoded = []
        for text in text_features:
            encoded.append(create_one_example(text, self.tokenizer, q_max_len=self.q_max_len, p_max_len=self.p_max_len))
        collated_features = super().__call__(encoded)
        return text_ids, collated_features


@dataclass
class DRInferenceCollator(DefaultDataCollator):
    def __call__(self, features):
        text_ids = [x["doc_id"] for x in features]
        collated_features = super().__call__(features)
        return text_ids, collated_features


@dataclass
class RRCollator(DataCollatorWithPadding):
    def __init__(self, data_args, sampler, tokenizer):
        super().__init__(tokenizer=tokenizer)
        self.data_args = data_args
        self.sampler = sampler
        self.max_q_len = data_args.q_max_len
        self.max_p_len = data_args.p_max_len

    def create_one_example(self, query_encoding: List[int], text_encoding: List[int]):
        item = self.tokenizer.prepare_for_model(
            query_encoding,
            text_encoding,
            truncation='only_first',
            max_length=self.max_q_len + self.max_p_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __call__(self, features):
        sampled_features = self.sampler(features)
        qq = sampled_features[0]
        dd = sampled_features[1]

        pos_pair = []
        neg_pair = []

        for i, (q, ds) in enumerate(zip(qq, dd)):
            pos_pair.append(self.create_one_example(q, ds[0]))
            for d in ds[1:]:
                neg_pair.append(self.create_one_example(q, d))

        pos_collated = self.tokenizer.pad(
            pos_pair,
            padding='max_length',
            max_length=self.max_q_len + self.max_p_len,
            return_tensors="pt",
        )
        neg_collated = self.tokenizer.pad(
            neg_pair,
            padding='max_length',
            max_length=self.max_q_len + self.max_p_len,
            return_tensors="pt",
        )

        return pos_collated, neg_collated
