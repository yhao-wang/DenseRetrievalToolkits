# trainer/trainer.py
import logging
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import collections
import transformers
from transformers import AutoModelForCausalLM, AutoConfig
from ..model.biencoder import DRModelForInference
from tqdm import tqdm
from .scheduler import (
    AbstractScheduler, InverseSquareRootScheduler, CosineScheduler, LinearScheduler, ConstantScheduler
)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config, model, train_set=None, eval_set=None, test_set=None):
        # --__init__: 初始化，设置模型、损失函数、数据类型，设置训练过程与评测方法
        self.config = config
        self._wrapper_model(model)
        self.loss_fn = get_loss_function(config)
        if config.do_train:
            self.train_set = train_set
        if config.do_eval:
            self.eval_set = eval_set
        if config.do_test:
            self.test_set = test_set
        self.eval_method = config.eval_method

    def _wrapper_model(self):
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.rank % torch.cuda.device_count())
        dist.init_process_group(backend="nccl")
        self.device = torch.device("cuda", self.local_rank)

        logger.info(f"[init] == local rank: {self.local_rank}, global rank: {self.rank} ==")

        # 1. define network
        self.model = self.model.to(self.device)
        # DistributedDataParallel
        self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

        # 2. define data_loader
        if self.config.do_train:
            self.train_batch_size = int(
                self.config.train_batch_size / self.config.gradient_accumulation_steps)
            train_sampler = DistributedSampler(self.train_set, shuffle=True)
            self.train_loader = torch.utils.data.DataLoader(
                self.train_set,
                batch_size=self.train_batch_size,
                num_workers=self.config.num_workers,
                pin_memory=True,
                sampler=train_sampler,
            )
        if self.config.do_eval:
            eval_sampler = DistributedSampler(self.eval_set, shuffle=False)
            self.eval_loader = torch.utils.data.DataLoader(
                self.eval_set,
                batch_size=self.config.eval_batch_size,
                num_workers=self.config.num_workers,
                pin_memory=True,
                sampler=eval_sampler,
            )

    def _get_get_optimizer_and_scheduler(self):
        # --get_optimizer_and_scheduler: 设置训练相关组件
        self._get_trainable_parameters()
        optimizer = self.config['optimizer']
        scheduler = self.config['scheduler']
        learning_rate = self.config['learning_rate']
        optimizer_kwargs = {'lr': self.config ['learning_rate']}
        optimizer_kwargs.update(self.config ['optimizer_kwargs'])
        adafactor_kwargs = self.config['adafactor_kwargs']
        scheduler_kwargs = self.config['scheduler_kwargs']
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

    def train_step(self, batch):
        # --train_step: 基于每个batch数据，更新模型参数，支持双塔单独训练与单塔双塔蒸馏
        encoder_outputs = self.model(batch)
        loss = self.loss_fn(encoder_outputs)
        return loss

    def train(self):
        # --train: 基于训练数据，迭代训练模型，输出训练后的模型
        self.model.train()
        max_epoch = self.config.max_epoch
        iter_bar = tqdm(self.train_loader, desc='Iter (loss=X.XXX)')
        for ep in range(1, max_epoch + 1):
            # set sampler
            self.train_loader.sampler.set_epoch(ep)

            for idx, batch in enumerate(iter_bar):
                batch = {
                    k: v.to(self.device) if v is not None else None for k, v in batch.items()}
                loss = self.train_step(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                iter_bar.set_description('Iter (loss=%5.3f)' % loss.item())

    def evaluate_step(self, batch):
        # --evaluate_step: 基于每个batch数据，得到评测结果并存储
        pass

    def evaluate(self, eval_loader):
        # --evaluate: 基于评测数据，评测已训练模型，输出评测结果
        pass

    def save(self, i_epoch):
        # --save: 保存模型参数到硬盘，包括Optimizer和Scheduler，train_step等，方便断点训练
        if self.local_rank == -1 or torch.distributed.get_rank() == 0:
            logger.info("** ** * Saving fine-tuned model and optimizer ** ** * ")
            model_to_save = self.model.lm_q
            output_model_file = os.path.join(self.config['output_dir'], "model.{0}.bin".format(i_epoch))
            torch.save(model_to_save.state_dict(), output_model_file)
            output_optim_file = os.path.join(self.config['output_dir'], "optim.{0}.bin".format(i_epoch))
            torch.save(self.optimizer.state_dict(), output_optim_file)

    def load(self):
        # --load: 从huggingface或硬盘读取模型参数并初始化，支持他人已预训练的ckpt如下所示
        if self.config['model_type'] == 'biencoder':
            config_path = self.config['config_path'] if hasattr(self.config, 'config_path') else self.config['model_path']
            config_kwargs = self.config['config_kwargs'] or {}
            configuration = AutoConfig.from_pretrained(config_path, **config_kwargs)
            transformer = self.model = AutoModelForCausalLM.from_pretrained(self.config['model_path'], config=configuration)
            self.model = DRModelForInference(transformer, transformer)


# --train_step: 基于每个batch数据，更新模型参数，支持双塔单独训练与单塔双塔蒸馏
# --evaluate: 基于评测数据，评测已训练模型，输出评测结果
# --evaluate_step: 基于每个batch数据，得到评测结果并存储
# --save: 保存模型参数到硬盘，包括Optimizer和Scheduler，train_step等，方便断点训练
# --load: 从huggingface或硬盘读取模型参数并初始化，支持他人已预训练的ckpt如下所示：
