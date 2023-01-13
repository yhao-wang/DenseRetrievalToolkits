import os
import sys
from DRT.trainer.trainer import Trainer
from DRT.arguments import DataArguments, ModelArguments, TrainingArguments
from transformers import HfArgumentParser, AutoTokenizer, AutoConfig
from DRT.dataset.data_collator import QPCollator
from DRT.dataset.abstract_dataset import ExactMatchDataset, RelevancyDataset, EXACTMATCH_DATASET
from DRT.dataloader.exactmatch_dataloader import ExactMatch_dataloader
from DRT.dataloader.relevancy_dataloader import Relevancy_dataloader
from DRT.trainer.sampler import RandomSampleNegatives
from DRT.model.biencoder import DRModel
from DRT.trainer.trainer import Trainer


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    num_labels = 1

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )

    model = DRModel.build(
        model_args=model_args,
        data_args=data_args,
        train_args=training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    dataset = ExactMatchDataset(data_args, tokenizer, cache_dir=data_args.data_cache_dir or model_args.cache_dir) if data_args.dataset in EXACTMATCH_DATASET else RelevancyDataset(data_args, tokenizer, cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    rnd_sampler = RandomSampleNegatives(data_args)
    dataloader = ExactMatch_dataloader(data_args, dataset, tokenizer, rnd_sampler, batch_size=training_args.train_batch_size) if data_args.dataset in EXACTMATCH_DATASET else Relevancy_dataloader(dataset, batch_size=training_args.train_batch_size)
    train_dataloader = dataloader.get_train_dataloader()

    trainer = Trainer(training_args, model, train_dataloader, eval_loader=None, test_loader=None)
    trainer.train()
    trainer.save()

    ########encode query and corpus########
    # model_args.model_name_or_path = training_args.output_dir
    # model = DRModelForInference.build(
    #     model_args=model_args,
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    # )

    # query_dataloader = dataloader.get_query_dataloader()

    # encoded_q = []
    # lookup_indices = []
    # model = model.to(training_args.device)
    # model.eval()

    # for (batch_ids, batch) in tqdm(query_dataloader):
    #     lookup_indices.extend(batch_ids)
    #     with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
    #         with torch.no_grad():
    #             for k, v in batch.items():
    #                 batch[k] = v.to(training_args.device)
    #             model_output: DROutput = model(query=batch)
    #             encoded_q.append(model_output.q_reps.cpu().detach().numpy())

    # encoded_q = np.concatenate(encoded_q)

    # with open(data_args.encodedq_save_path, 'wb') as f:
    #     pickle.dump((encoded_q, lookup_indices), f)

    # corpus_dataloader = dataloader.get_corpus_dataloader()
    # encoded_p = []
    # lookup_indices = []

    # for (batch_ids, batch) in tqdm(corpus_dataloader):
    #     lookup_indices.extend(batch_ids)
    #     with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
    #         with torch.no_grad():
    #             for k, v in batch.items():
    #                 batch[k] = v.to(training_args.device)
    #             model_output: DROutput = model(passage=batch)
    #             encoded_p.append(model_output.p_reps.cpu().detach().numpy())

    # encoded_p = np.concatenate(encoded_p)

    # with open(data_args.encodedp_save_path, 'wb') as f:
    #     pickle.dump((encoded_p, lookup_indices), f)

    print("ok!")


if __name__ == '__main__':
    main()
