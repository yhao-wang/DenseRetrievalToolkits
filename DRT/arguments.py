import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    target_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained reranker target model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    num_labels: Optional[int] = field(
        default=1, metadata={"help": "number of labels"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # modeling
    untie_encoder: bool = field(
        default=False,
        metadata={"help": "no weight sharing between qry passage encoders"}
    )
    feature: str = field(
        default="last_hidden_state",
        metadata={"help": "What feature to be extracted from the HF PLM"}
    )
    pooling: str = field(
        default="first",
        metadata={"help": "How to pool the features from the HF PLM"}
    )

    # out projection
    add_linear_head: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=768)

    # for Jax training
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one "
                    "of `[float32, float16, bfloat16]`. "
        },
    )

    encoder_only: bool = field(
        default=False,
        metadata={"help": "Whether to only use the encoder of T5"}
    )
    pos_token: Optional[str] = field(
        default=None,
        metadata={"help": "Token that indicates a relevant document"}
    )
    neg_token: Optional[str] = field(
        default=None,
        metadata={"help": "Token that indicates a irrelevant document"}
    )

    normalize: bool = field(
        default=False,
        metadata={"help": "Whether to normalize the embeddings"}
    )
    param_efficient_method: Optional[str] = field(
        default=None,
        metadata={"help": "Param efficient method used in model training"}
    )


@dataclass
class DataArguments:
    dataset: str = field(
        default=None, metadata={"help": "dataset name: nq, wq, t, squad, msmacro"}
    )
    data_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    dataset_name: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    corpus_name: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    corpus_path: str = field(
        default=None, metadata={"help": "corpus dataset path"}
    )
    passage_field_separator: str = field(default=' ')
    dataset_proc_num: int = field(
        default=12, metadata={"help": "number of proc used in dataset preprocess"}
    )
    train_n_passages: int = field(default=8)
    positive_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first positive passage"})
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first negative passages"})

    encode_in_path: List[str] = field(default=None, metadata={"help": "Path to data to encode"})
    encodedq_save_path: str = field(default=None, metadata={"help": "where to save the encoded query"})
    encodedp_save_path: str = field(default=None, metadata={"help": "where to save the encoded passage"})
    encode_is_qry: bool = field(default=False)
    encode_num_shard: int = field(default=1)
    encode_shard_index: int = field(default=0)

    q_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    p_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    data_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the data downloaded from huggingface"}
    )

    def __post_init__(self):
        if self.dataset_name is not None:
            info = self.dataset_name.split('/')
            self.dataset_split = info[-1] if len(info) == 3 else 'train'
            self.dataset_name = "/".join(info[:-1]) if len(info) == 3 else '/'.join(info)
            self.dataset_language = 'default'
            if ':' in self.dataset_name:
                self.dataset_name, self.dataset_language = self.dataset_name.split(':')
        else:
            self.dataset_name = 'json'
            self.dataset_split = 'train'
            self.dataset_language = 'default'
        if self.data_dir is not None:
            if os.path.isdir(self.data_dir):
                files = os.listdir(self.data_dir)
                # change all train directory paths to absolute
                self.data_dir = os.path.join(os.path.abspath(os.getcwd()), self.data_dir)
                for f in files :
                    if f.endswith('train.jsonl') or f.endswith('train.json'):
                        train_file = os.path.join(self.data_dir, f)
                    elif f.endswith('test.jsonl') or f.endswith('test.json'):
                        test_file = os.path.join(self.data_dir, f)
                    elif f.endswith('dev.jsonl') or f.endswith('dev.json'):
                        dev_file = os.path.join(self.data_dir, f)
                self.data_path = {
                    "train": train_file,
                    "test": test_file,
                    "dev": dev_file
                }
            else:
                self.data_path = [self.data_dir]
        else:
            self.data_path = None
        self.corpus_name = "json" if self.corpus_name is None else self.corpus_name


@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    local_rank: int = field(default=0)
    warmup_ratio: float = field(default=0.1)
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})

    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)
    eval_method: str = field(default='metrics')
    optimizer: str = field(default='adam')
    scheduler: str = field(default=None)
    learning_rate: float = field(default=1e-5)
    optimizer_kwargs: dict = field(default_factory=dict)
    adafactor_kwargs: dict = field(default_factory=dict)
    scheduler_kwargs: dict = field(default_factory=dict)
    train_batch_size: int = field(default=128)
    eval_batch_size: int = field(default=128)
    test_batch_size: int = field(default=128)
    corpus_batch_size: int = field(default=128)
    max_epochs: int = field(default=5)
    decimal_place: int = field(default=2)
    topk: str = field(default="5,10,20")
    retrieve_num: int = field(default=100)
    retrieve_dir: str = field(default="")
    eval_per_train: int = field(default=5)
    index_order_dir: str = field(default="")
    rr_result_dir: str = field(default="")
    encode_corpus_dir: str = field(default="")
    loss_fn: str = field(default="SimpleContrastiveLoss")
    index_file: str = field(default="")
    cache_train_dir: str = field(default="/home/wangyuhao/DRT_cache/")
    save_per_train: int = field(default=10, metadata={"help": "Save checkpoint every X epoches."})

    def __post_init__(self):
        if self.index_file == '':
            self.index_file = os.path.join(self.cache_train_dir, 'index_1phrase')
        if self.retrieve_dir == '':
            self.retrieve_dir = os.path.join(self.cache_train_dir, 'retrieve')
        if self.index_order_dir == '':
            self.index_order_dir = os.path.join(self.cache_train_dir, 'idx')
        if self.rr_result_dir == '':
            self.rr_result_dir = os.path.join(self.cache_train_dir, 'rr')
        if self.encode_corpus_dir == '':
            self.encode_corpus_dir = os.path.join(self.cache_train_dir, 'encoded_p')
        for dir in [self.retrieve_dir, self.encode_corpus_dir,  self.rr_result_dir, self.index_order_dir]:
            os.makedirs(dir, exist_ok=True)
        if self.save_per_train > self.max_epochs:
            self.save_per_train = self.max_epochs


@dataclass
class RRTrainingArguments(TrainingArguments):
    loss_fn: str = field(default="mr")
    margin: float = field(default=1.0)
