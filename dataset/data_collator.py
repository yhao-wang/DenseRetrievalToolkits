from dataclasses import dataclass
from typing import List
from transformers import DataCollatorWithPadding, DefaultDataCollator


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

    def create_one_example(self, text_encoding: List[int], is_query=False):
        item = self.tokenizer.prepare_for_model(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __call__(self, features):
        sampled_features = self.sampler(features)
        qq = sampled_features[0]
        dd = sampled_features[1]

        enq = []
        end = []
        for q in qq:
            enq.append(self.create_one_example(q, is_query=True))
        for s in dd:
            one_sample = []
            for d in s:
                one_sample.append(self.create_one_example(d, is_query=False))
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
class EncodeCollator(DataCollatorWithPadding):
    def __call__(self, features):
        text_ids = [x[0] for x in features]
        text_features = [x[1] for x in features]
        collated_features = super().__call__(text_features)
        return text_ids, collated_features

@dataclass
class DRInferenceCollator(DefaultDataCollator):
    def __call__(self, features):
        text_ids = [x["doc_id"] for x in features]
        collated_features = super().__call__(features)
        return text_ids, collated_features


@dataclass
class RRInferenceCollator(DefaultDataCollator):
    def __call__(self, features):
        query_ids = [x["query_id"] for x in features]
        doc_ids = [x["doc_id"] for x in features]
        collated_features = super().__call__(features)
        return query_ids, doc_ids, collated_features
