CUDA_VISIBLE_DEVICES=5,6,7,8,9 \
python3 -m torch.distributed.launch \
--nproc_per_node=5 --node_rank=0 \
run_toolkits.py \
--train_batch_size 8 \
--eval_batch_size 8 \
--test_batch_size 8 \
--output_dir /mnt/wangyuhao/model_nq \
--model_name_or_path bert-base-uncased \
--save_steps 20000 \
--dataset nq \
--data_cache_dir /mnt/wangyuhao/DRT_cache \
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

OMP_NUM_THREADS=40 \
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m torch.distributed.launch \
--nproc_per_node=2 --node_rank=0 \
run_random_sampling.py \
--train_batch_size 16 \
--eval_batch_size 16 \
--corpus_batch_size 16 \
--eval_per_train 30 \
--test_batch_size 32 \
--topk 5,10,20,50,100 \
--retrieve_num 100 \
--output_dir /home/wangyuhao/model_nq \
--model_name_or_path /home/wangyuhao/DRT_cache/result30 \
--save_per_train 10 \
--untie_encoder \
--dataset nq \
--data_cache_dir /home/wangyuhao/DRT_cache \
--dataset_name Tevatron/wikipedia-nq \
--corpus_name xxazz/nq-corpus \
--positive_passage_no_shuffle \
--train_n_passages 2 \
--learning_rate 1e-5 \
--q_max_len 32 \
--p_max_len 156 \
--max_epochs 30 \
--encodedp_save_path corpus_emb.pkl

OMP_NUM_THREADS=40 \
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m torch.distributed.launch \
--nproc_per_node=2 --node_rank=0 \
run_random_sampling.py \
--train_batch_size 16 \
--eval_batch_size 16 \
--corpus_batch_size 16 \
--eval_per_train 40 \
--test_batch_size 32 \
--topk 5,10,20,50,100 \
--retrieve_num 100 \
--output_dir /home/wangyuhao/model_nq \
--model_name_or_path bert-base-uncased \
--save_per_train 10 \
--untie_encoder \
--dataset nq \
--data_cache_dir /home/wangyuhao/DRT_cache \
--dataset_name Tevatron/wikipedia-nq \
--corpus_name xxazz/nq-corpus \
--positive_passage_no_shuffle \
--train_n_passages 2 \
--learning_rate 1e-5 \
--q_max_len 32 \
--p_max_len 156 \
--max_epochs 40 \
--encodedp_save_path corpus_emb.pkl

OMP_NUM_THREADS=40 \
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m torch.distributed.launch \
--nproc_per_node=2 --node_rank=0 \
run_reranker.py \
--train_batch_size 16 \
--eval_batch_size 32 \
--test_batch_size 16 \
--output_dir /home/wangyuhao/model_nq \
--model_name_or_path bert-base-uncased \
--save_steps 20000 \
--dataset nq \
--train_n_passages 2 \
--data_cache_dir /home/wangyuhao/DRT_cache \
--dataset_name Tevatron/wikipedia-nq \
--corpus_name xxazz/nq-corpus \
--positive_passage_no_shuffle \
--learning_rate 1e-5 \
--q_max_len 32 \
--p_max_len 156 \
--max_epochs 3 \
--encodedq_save_path query_emb.pkl

CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m torch.distributed.launch \
--nproc_per_node=2 --node_rank=0 \
run_BM25_negative.py \
--train_batch_size 4 \
--eval_batch_size 4 \
--test_batch_size 4 \
--output_dir /home/wangyuhao/model_nq \
--model_name_or_path bert-base-uncased \
--save_steps 20000 \
--dataset nq \
--train_n_passages 8 \
--data_cache_dir /home/wangyuhao/DRT_cache \
--dataset_name Tevatron/wikipedia-nq \
--corpus_name xxazz/nq-corpus \
--positive_passage_no_shuffle \
--learning_rate 1e-5 \
--q_max_len 32 \
--p_max_len 156 \
--max_epochs 3 \
--encodedq_save_path query_emb.pkl


python3 run_BM25_negative.py \
--train_batch_size 8 \
--eval_batch_size 8 \
--test_batch_size 8 \
--output_dir /mnt/d/DRT-files/model_nq \
--model_name_or_path bert-base-uncased \
--save_steps 20000 \
--dataset nq \
--data_cache_dir /mnt/d/DRT-files/DRT_cache \
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



python3 test.py \
--train_batch_size 16 \
--eval_batch_size 16 \
--corpus_batch_size 16 \
--eval_per_train 30 \
--test_batch_size 32 \
--topk 5,10,20,50,100 \
--retrieve_num 100 \
--output_dir /mnt/wangyuhao/model_nq \
--model_name_or_path /mnt/wangyuhao/DRT_cache/result30 \
--save_per_train 10 \
--untie_encoder \
--dataset nq \
--data_cache_dir /mnt/wangyuhao/DRT_cache \
--dataset_name Tevatron/wikipedia-nq \
--corpus_name xxazz/nq-corpus \
--positive_passage_no_shuffle \
--train_n_passages 2 \
--learning_rate 1e-5 \
--q_max_len 32 \
--p_max_len 156 \
--max_epochs 30 \
--encodedp_save_path corpus_emb.pkl
