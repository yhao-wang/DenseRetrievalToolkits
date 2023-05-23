class TrainPreProcessor:
    def __init__(self, tokenizer, query_max_length=32, text_max_length=256, separator=' '):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, example):
        query = self.tokenizer.encode(example['query'],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        positives = []
        for pos in example['positive_passages']:
            text = pos['title'] + self.separator + pos['text'] if 'title' in pos else pos['text']
            positives.append(self.tokenizer.encode(text,
                                                   add_special_tokens=False,
                                                   max_length=self.text_max_length,
                                                   truncation=True))
        negatives = []
        for neg in example['negative_passages']:
            text = neg['title'] + self.separator + neg['text'] if 'title' in neg else neg['text']
            negatives.append(self.tokenizer.encode(text,
                                                   add_special_tokens=False,
                                                   max_length=self.text_max_length,
                                                   truncation=True))

        return {'query': query, 'positives': positives, 'negatives': negatives}


class EvalPreProcessor:
    def __init__(self, tokenizer, query_max_length=32, text_max_length=256, separator=' '):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, example):
        query = self.tokenizer.encode(example['query'],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        positives = []
        for pos in example['positive_passages']:
            positives.append(pos['docid'])

        return {'query': query, 'positives_ids': positives}


class DocPreProcessor:
    def __init__(self, tokenizer, text_max_length=256):
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length

    def __call__(self, example):
        id = example['id']
        text = self.tokenizer.encode(example['text'],
                                      add_special_tokens=False,
                                      max_length=self.text_max_length,
                                      truncation=True)
        return {'id': id, 'text': text, 'original': example['text']}


class RREVPreProcessor:
    def __init__(self, tokenizer, query_max_length=32, text_max_length=256):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length

    def __call__(self, example):
        query = self.tokenizer.encode(example['query'],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        document = self.tokenizer.encode(example['document'],
                                         add_special_tokens=False,
                                         max_length=self.text_max_length,
                                         truncation=True)
        return {'query_id': example['query_id'],
                'query': query,
                'doc_id': example['document'],
                'document': document,
                'original': example['document'],
                'answers': example['answers']}


class RelevancyPreProcessor:
    def __init__(self, tokenizer, query_max_length=32):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length

    def __call__(self, example):
        query_id = example['query_id']
        query = self.tokenizer.encode(example['query'],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        pos_doc_ids = example['pos_doc_ids']
        return {'query_id': query_id, 'query': query, 'pos_doc_ids': pos_doc_ids}


class ExactMatchPreProcessor:
    def __init__(self, tokenizer, query_max_length=32):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length

    def __call__(self, example):
        query_id = example['query_id']
        query = self.tokenizer.encode(example['query'],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        # answers = self.tokenizer.encode(example['answers'],
        #                               add_special_tokens=False,
        #                               max_length=self.query_max_length,
        #                               truncation=True)
        answers = example['answers']
        return {'query_id': query_id, 'query': query, 'answers': answers, 'original': example['query']}


class QueryPreProcessor:
    def __init__(self, tokenizer, query_max_length=32):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length

    def __call__(self, example):
        query_id = example['query_id']
        query = self.tokenizer.encode(example['query'],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        return {'query_id': query_id, 'query': query}


class CorpusPreProcessor:
    def __init__(self, tokenizer, text_max_length=256, separator=' '):
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, example):
        docid = example['docid']
        text = example['title'] + self.separator + example['text'] if 'title' in example else example['text']
        endtext = self.tokenizer.encode(text,
                                     add_special_tokens=False,
                                     max_length=self.text_max_length,
                                     truncation=True)
        # return {'id': id, 'text': text, 'original': example ['text']}
        print(text)
        return {'id': docid, 'text': endtext, 'original': text}
