export CUDA_VISIBLE_DEVICES=0

WANDB_MODE=offline accelerate launch --num_machines=1 --num_processes=1 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network train_timeartist_align.py config=configs/stage2/timeartist_b64_512_align.yaml