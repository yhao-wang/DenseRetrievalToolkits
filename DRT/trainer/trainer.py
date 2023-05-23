# trainer/trainer.py
import logging
import os
import torch
import json
import torch.distributed as dist
from torch import optim
from DRT.evaluator.index import BaseFaissIPRetriever
from ..arguments import TrainingArguments
from DRT.evaluator.metrics import get_metrics
from torch.nn.parallel import DistributedDataParallel as DDP
import collections
import faiss
import transformers
from .losses import get_loss_function
from tqdm import tqdm
import numpy as np
from ..evaluator.nq_eval import has_answers
from .scheduler import (
    AbstractScheduler, InverseSquareRootScheduler, CosineScheduler, LinearScheduler, ConstantScheduler
)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, training_args: TrainingArguments, model, corpus_dataloader=None, train_loader=None, eval_loader=None, test_loader=None):
        # --__init__: 初始化，设置模型、损失函数、数据类型，设置训练过程与评测方法
        self.training_args = training_args
        self.model = model
        self._wrapper_model()
        self.loss_fn = get_loss_function(training_args)
        self.train_loader = train_loader
        self._get_get_optimizer_and_scheduler()
        self.corpus_dataloader = corpus_dataloader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.start_epoch = 0
        self.eval_method = training_args.eval_method
        if eval_loader is not None:
            training_args.topk = [int(k) for k in training_args.topk.split(",")]

    def _wrapper_model(self):

        self.rank = int(os.environ["RANK"]) if hasattr(os.environ, "RANK") else 0
        self.local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else -1
        torch.cuda.set_device(self.rank % torch.cuda.device_count())
        self.device = torch.device("cuda", self.local_rank)

        dist.barrier()
        logger.info(f"[init] == local rank: {self.local_rank}, global rank: {self.rank} ==")
        dist.barrier()

        # 1. define network
        self.model = self.model.to(self.device)
        # DistributedDataParallel
        torch.cuda.set_device(self.local_rank)
        if torch.cuda.device_count() > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

    def _get_get_optimizer_and_scheduler(self):
        # --get_optimizer_and_scheduler: 设置训练相关组件
        self._get_trainable_parameters()
        optimizer = self.training_args.optimizer
        scheduler = self.training_args.scheduler
        learning_rate = self.training_args.learning_rate
        optimizer_kwargs = {'lr': self.training_args.learning_rate}
        optimizer_kwargs.update(self.training_args.optimizer_kwargs)
        adafactor_kwargs = self.training_args.adafactor_kwargs
        scheduler_kwargs = self.training_args.scheduler_kwargs
        optimizer_class = collections.defaultdict(
            lambda: optim.AdamW, {
                'adam': optim.Adam,
                'adamw': optim.AdamW,
                'sgd': optim.SGD,
                'adagrad': optim.Adagrad,
                'rmsprop': optim.RMSprop,
                'adafactor': transformers.Adafactor,
            }
        )
        scheduler_class = {
            'inverse': InverseSquareRootScheduler,
            'cosine': CosineScheduler,
            'linear': LinearScheduler,
            'constant': ConstantScheduler,
        }

        # dealing with adafactor
        if optimizer == 'adafactor':
            # using adafactor_kwargs in overall.yaml
            if self.grad_clip is not None:
                self.grad_clip = None
                logger.warning(
                    "Additional optimizer operations like gradient clipping "
                    "should not be used alongside Adafactor."
                )
            optimizer_kwargs.update(adafactor_kwargs)

        # get optimizer (use default value of pytorch if self.optimizer_kwargs is empty)
        logger.debug(f'Using optimizer {optimizer}')
        optimizer = optimizer_class[optimizer](params=self._trainable_parameters, **optimizer_kwargs)

        # scheduling
        if scheduler is not None and scheduler in scheduler_class:
            assert isinstance(scheduler_kwargs, dict), "Please specify scheduler_kwargs"
            logger.debug(f'Using scheduler {scheduler}.')
            scheduler_kwargs.setdefault("max_lr", learning_rate)
            optimizer = scheduler_class[scheduler](base_optimizer=optimizer, **scheduler_kwargs)
        self.optimizer = optimizer

    def _get_trainable_parameters(self):
        self._trainable_parameters = filter(lambda x: x.requires_grad, self.model.parameters())

    def train_step(self, inputs):
        # --train_step: 基于每个batch数据，更新模型参数，支持双塔单独训练与单塔双塔蒸馏
        inputs = {
            "query": inputs[0],
            "passage": inputs[1],
        }
        encoded = self.model(**inputs)
        # q_rep = encoded.q_reps
        # p_rep = encoded.p_reps
        # inputs = {
        #     "x": q_rep.contiguous(),
        #     "y": p_rep.contiguous(),
        # }
        # loss = self.loss_fn(**inputs)
        return encoded.loss

    def train(self):
        # --train: 基于训练数据，迭代训练模型，输出训练后的模型
        self.model.train()
        max_epochs = self.training_args.max_epochs

        for ep in range(self.start_epoch, max_epochs):
            # set sampler
            iter_bar = tqdm(self.train_loader, desc='Training XX (loss=X.XXX)')
            if torch.cuda.device_count() > 1:
                self.train_loader.sampler.set_epoch(ep)

            for idx, batch in enumerate(iter_bar):
                prepared = []
                for data in batch:
                    data = {
                        k: v.to(self.device) if v is not None else None for k, v in data.items()}
                    prepared.append(data)
                loss = self.train_step(prepared)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                iter_bar.set_description('Training %d (loss=%5.3f)' % (ep + 1, loss.item()))
            dist.barrier()
            if (ep + 1) % self.training_args.save_per_train == 0:
                self.save(ep + 1)
            if (ep + 1) % self.training_args.eval_per_train == 0:
                self.evaluate(self.eval_loader, ep + 1)
                # self.save(ep + 1)
        self.evaluate(self.test_loader, -1)


    # def evaluate_step(self, batch):
    #     # --evaluate_step: 基于每个batch数据，得到评测结果并存储
    #     inputs = {
    #         "query": batch[0],
    #         "passage": batch[1],
    #     }
    #     encoded = self.model(**inputs)
    #     q_reps = encoded.q_reps.detach().cpu().numpy()
    #     p_reps = encoded.p_reps.detach().cpu().numpy()
    #     eval_num += q_reps.shape[0]
    #     s = 0
    #     pos_len = np.zeros(p_reps.shape[0], dtype=np.int32)
    #     pos_index = np.zeros([p_reps.shape[0], max(self.training_args.topk)], dtype=np.int8)
    #     for q_idx in range(0, p_reps.shape[0]):
    #         e = s + batch[2][q_idx]
    #         scores, indices = self.retriever.search(q_reps[q_idx: q_idx+1], max(self.training_args.topk))
    #         pos_len[i] = batch[2][q_idx]
    #         for i, doc in enumerate(self.corpus[indices]):
    #             for p in p_reps[s, e]:
    #                 if doc == p:
    #                     pos_index[q_idx][i] = 1
    #         s = e
    #         metrics = get_metrics(pos_len, pos_index, self.training_args)
    #         for k, v in m_all.items():
    #             m_all[k] = v + metrics[k]

    def _encoding_corpus(self, ep):
        self.index = None
        iter_bar = tqdm(self.corpus_dataloader, desc='Encoding corpus: ')
        end = []
        idx = []
        for batch in iter_bar:
            data = {
                k: v.to(self.device) if v is not None else None for k, v in batch[1].items()}
            inputs = {
                "query": None,
                "passage": data,
            }
            encoded = self.model(**inputs)
            p_reps = encoded.p_reps.detach().cpu().numpy()
            end.append(p_reps)
            idx.append(batch[0])
            if self.index is None:
                self.index = BaseFaissIPRetriever(p_reps)
            # self.index.add(p_reps.astype(np.float32))
        end = np.concatenate(end)
        np.save(os.path.join(self.training_args.encode_corpus_dir, str(ep) + '.' + str(self.local_rank) + ".npy"), end)
        f = open(os.path.join(self.training_args.encode_corpus_dir, str(ep) + '.' + str(self.local_rank) + ".json"), 'w', encoding='utf-8')
        for i in tqdm(range(len(idx)), desc='Saving encoded corpus: '):
            json.dump({'id': idx[i]}, f, ensure_ascii=False)
            f.write('\n')
        f.close()
        # faiss.write_index(self.index.index, self.training_args.index_file + str(ep) + '.' + str(self.local_rank))
        dist.barrier()

    def _index_corpus(self, ep):
        # self.index = BaseFaissIPRetriever(768)
        if self.local_rank == 0:
            file_list = os.listdir(self.training_args.encode_corpus_dir)
            fl = []
            for file in file_list:
                if file.startswith(str(ep) + '.') and file.endswith('json'):
                    fl.append(file)
            file_list = fl
            self.idx = []
            for file in tqdm(file_list, desc='Indexing files: '):
                desc = "Indexing " + file + ": "
                file = os.path.join(self.training_args.encode_corpus_dir, file)
                npy_file = file[:-4] + "npy"
                print("loading " + npy_file + " and indexing...")
                self.index.add(np.load(npy_file))
                f = open(file, 'r', encoding='utf-8')
                iter_bar = tqdm(f.readlines(), desc=desc)
                for line in iter_bar:
                    dic = json.loads(line)
                    self.idx.extend(dic['id'])
                f.close()
            # for file in file_list:
            #     file = os.path.join(self.training_args.encode_corpus_dir, file)
            #     os.remove(file)
            faiss.write_index(self.index.index, self.training_args.index_file + str(ep))
            f = open(os.path.join(self.training_args.index_order_dir,  str(ep) + '.' + "docid.txt"), 'w', encoding='utf-8')
            json.dump({"id": self.idx}, f, ensure_ascii=False)
            f.close()
            print("Index and id orders have been saved.")
        dist.barrier()

    def _load_index(self, ep):
        if self.local_rank != 0:
            # self.index = BaseFaissIPRetriever(768)
            print(str(self.local_rank) + " is loading index...")
            self.index = BaseFaissIPRetriever(768)
            self.index.index = faiss.read_index(self.training_args.index_file + str(ep))
            f = open(os.path.join(self.training_args.index_order_dir, str(ep) + '.' + "docid.txt"), 'r', encoding='utf-8')
            for line in f.readlines():
                self.idx = json.loads(line)['id']
            f.close()
        dist.barrier()
        # if self.local_rank == 0:
        #     path_list = [os.path.join(self.training_args.index_order_dir, str(ep) + '.' + "docid.txt"),
        #                  self.training_args.index_file + str(ep)]
        #     for file in path_list:
        #         os.remove(file)

    def evaluate(self, query_loader, ep):
        self.model.eval()
        self._encoding_corpus(ep)
        self._index_corpus(ep)
        self._load_index(ep)
        m_all = {}
        eval_num = 0
        for metric in ["MRR", "NDCG", "Recall"]:
            for k in self.training_args.topk:
                m_all.update({"{}@{}".format(metric, k): 0.0})
        if torch.cuda.device_count() > 1:
            query_loader.sampler.set_epoch(0)
        iter_bar = tqdm(query_loader, desc='Evaluation: ')
        documents = []
        queries = []
        answers = []
        qid = []
        did = []
        for idx, batch in enumerate(iter_bar):
            data = {
                k: v.to(self.device) if v is not None else None for k, v in batch[1].items()}
            inputs = {
                "query": data,
                "passage": None,
            }
            encoded = self.model(**inputs)
            q_reps = encoded.q_reps.detach().cpu().numpy()
            k = self.training_args.retrieve_num
            indices = self.index.search(q_reps, k)
            pos_index = np.zeros([len(q_reps), k], dtype=np.int8)
            docs = []
            doc_ids = []

            for i, indice in enumerate(indices):
                doc_id = []
                eval_num += 1
                doc = []
                for j, id in enumerate(indice):
                    d = self.corpus_dataloader.dataset[self.idx[id]]['original']
                    doc_id.append(self.idx[id])
                    doc.append(d)
                    if has_answers(d, batch[2][i]):
                        pos_index[i][j] = 1
                docs.append(doc)
                doc_ids.append(doc_id)
            documents.extend(docs)
            qid.extend(batch[0])
            answers.extend(batch[2])
            queries.extend(batch[3])
            did.extend(doc_ids)
            metrics = get_metrics(pos_index, self.training_args.topk)
            for k, v in m_all.items():
                m_all[k] = v + metrics[k]

        f = open(os.path.join(self.training_args.retrieve_dir, str(ep) + '.' + str(self.local_rank) + ".json"), 'w',
                 encoding='utf-8')
        for i in range(len(did)):
            for doc, d in zip(documents[i], did[i]):
            # for d in did[i]:
                dic = {
                    'doc_id': d,
                    'query_id': qid[i],
                    'query': queries[i],
                    'document': doc,
                    'answers': answers[i],
                }
                json.dump(dic, f, ensure_ascii=False)
                f.write('\n')
        f.close()
        for k, v in m_all.items():
            m_all[k] = v / eval_num
            print(k, m_all[k])
        m_all['query_num'] = eval_num
        f = open(os.path.join(self.training_args.cache_train_dir, str(ep) + '.' + str(self.local_rank) + "_metrics"),
                 'w', encoding='utf-8')
        json.dump(m_all, f, ensure_ascii=False)
        f.close()
        dist.barrier()

    def save(self, i_epoch):
        if self.local_rank == 0:
            path = os.path.join(self.training_args.cache_train_dir, "result" + str(i_epoch))
            os.makedirs(path, exist_ok=True)

            if torch.cuda.device_count() > 1:
                self.model.module.save(path)
            else:
                self.model.save(path)
        # # --save: 保存模型参数到硬盘，包括Optimizer和Scheduler，train_step等，方便断点训练
        # if self.local_rank == -1 or torch.distributed.get_rank() == 0:
        #     ckpt = self._get_checkpoint(i_epoch)
        #     output_path = os.path.abspath(self.training_args.output_dir)
        #     if not os.path.exists(output_path):
        #         os.mkdir(output_path)
        #     file_name = str(i_epoch) + ".bin"
        #     output_file = os.path.join(output_path, file_name)
        #     torch.save(ckpt, output_file)

    def _get_checkpoint(self, i_epoch):
        # construct state_dict and parameters
        if torch.cuda.device_count() > 1:
            _state_dict = self.model.module.get_model_ckpt()
        else:
            _state_dict = self.model.get_model_ckpt()

        # get optimizer, config and validation summary
        checkpoint = {
            # parameters that needed to be loaded
            'state_dict': _state_dict,
            'optimizer': self.optimizer.state_dict(),
            'epoch': i_epoch,
        }
        return checkpoint

    def load(self, filename, ckpt_type=None):
        # --load: 从huggingface或硬盘读取模型参数并初始化，支持他人已预训练的ckpt如下所示
        checkpoint = torch.load(filename, map_location=self.device)
        if ckpt_type == None:
            self.start_epoch = checkpoint['epoch'] + 1
            self.model.load(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])


class RRTrainer(Trainer):

    def train_step(self, inputs):
        # --train_step: 基于每个batch数据，更新模型参数，支持双塔单独训练与单塔双塔蒸馏
        inputs = {
            "pos_pairs": inputs[0],
            "neg_pairs": inputs[1],
        }
        rr = self.model(**inputs)
        return rr.loss

    def evaluate(self, pair_loder, ep):
        m_all = {}
        eval_num = 0
        for metric in ["MRR", "NDCG", "Recall"]:
            for k in self.training_args.topk:
                m_all.update({"{}@{}".format(metric, k): 0.0})
        if torch.cuda.device_count() > 1:
            pair_loder.sampler.set_epoch(0)
        iter_bar = tqdm(pair_loder, desc='Evaluation: ')
        result = {}
        for idx, batch in enumerate(iter_bar):
            data = {
                k: v.to(self.device) if v is not None else None for k, v in batch[1].items()}
            inputs = {
                "pos_pairs": data,
                "neg_pairs": None,
            }
            scores = self.model(**inputs).detach().cpu().numpy()
            qid = batch[0]
            answers = batch[2]
            doc = batch[3]
            dids = batch[4]
            for q, a, d, s, did in zip(qid, answers, doc, scores, dids):
                if q not in result.keys():
                    result[q] = [], [], [], []
                result[q][0].append(float(s[0]))
                result[q][1].append(int(has_answers(d, a)))
                result[q][2].append(d)
                result[q][3].append(did)
        f = open(os.path.join(self.training_args.rr_result_dir, str(ep) + '.' + str(self.local_rank)+".json"), "w", encoding='utf-8')
        for qid, (scores, is_true, ds, dids) in result.items():
            for s, j, d, did in zip(scores, is_true, ds, dids):
                dic = {
                    'qid': qid,
                    'did': did,
                    'score': s,
                    'match': j,
                    'document': d,
                }
                json.dump(dic, f, ensure_ascii=False)
                f.write("\n")
        f.close()
        # timeout = os.environ["NCCL_NET_GDR_TIMEOUT"]
        # os.environ["NCCL_NET_GDR_TIMEOUT"] = "60000000"
        dist.barrier()
        if self.local_rank == 0:
            result = {}
            file_list = os.listdir(self.training_args.rr_result_dir)
            fl = []
            for file in file_list:
                if file.startswith(str(ep) + '.'):
                    fl.append(file)
            file_list = fl
            file_list = [os.path.join(self.training_args.rr_result_dir, f) for f in file_list]
            for file in file_list:
                f = open(file, 'r', encoding='utf-8')
                for line in f.readlines():
                    dic = json.loads(line)
                    if dic['qid'] not in result.keys():
                        result[dic['qid']] = [], []
                    result[dic['qid']][0].append(dic['score'])
                    result[dic['qid']][1].append(dic['match'])
                f.close()
            for qid, (scores, is_true) in result.items():
                eval_num += 1
                scores = np.array(scores)
                pos_index = [np.array(is_true)[np.argsort(-scores)]]
                metrics = get_metrics(pos_index, self.training_args.topk)
                for k, v in m_all.items():
                    m_all[k] = v + metrics[k]

            m_all['query_num'] = eval_num
            for k, v in m_all.items():
                m_all[k] = v / eval_num
                print(k, m_all[k])
            f = open(os.path.join(self.training_args.cache_train_dir, str(ep) + '.' + str(self.local_rank) + "_RR_metrics"),
                'w',
                encoding='utf-8')
            json.dump(m_all, f, ensure_ascii=False)
            f.close()

        dist.barrier()






    # --train_step: 基于每个batch数据，更新模型参数，支持双塔单独训练与单塔双塔蒸馏
# --evaluate: 基于评测数据，评测已训练模型，输出评测结果
# --evaluate_step: 基于每个batch数据，得到评测结果并存储
# --save: 保存模型参数到硬盘，包括Op，方便断点训练
# --load: 从huggingface或硬盘读取模型参数并初始化，支持他人已预训练的ckpt如下所示：
