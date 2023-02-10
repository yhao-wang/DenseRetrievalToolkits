python3 -m torch.distributed.launch \
run_toolkits.py \
--train_batch_size 8 \
--eval_batch_size 8 \
--test_batch_size 8 \
--output_dir model_nq \
--model_name_or_path bert-base-uncased \
--save_steps 20000 \
--dataset nq \
--data_cache_dir /root/autodl-tmp/DRT_cache \
--dataset_name Tevatron/wikipedia-nq \
--corpus_name xxazz/nq-corpus \
--positive_passage_no_shuffle \
--train_n_passages 8 \
--learning_rate 1e-5 \
--q_max_len 32 \
--p_max_len 156 \
--max_epochs 3 \
--encodedq_save_path query_emb.pkl \
--encodedp_save_path corpus_emb.pkl