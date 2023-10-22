#!/bin/sh
env="MPE"
scenario="simple_tag"
num_good_agents=1
num_adversaries=3
num_landmarks=2
algo="rmappo" #"mappo" "ippo"
exp="check"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents 4 --num_good_agents 1 --num_adversaries 3 \
    --num_landmarks ${num_landmarks} --seed ${seed} --share_policy \
    --n_training_threads 1 --n_rollout_threads 1 --use_render True --save_gifs True --episode_length 25 --render_episodes 5 --use_wandb \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
    --model_dir "results/MPE/simple_tag/rmappo/check/run17/models"
done
