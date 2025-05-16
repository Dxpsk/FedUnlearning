#!/bin/bash

commands=(

# "python ./federated.py --agent_frac 0.25 --method rlr --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr avg --device cuda:0 --rounds  400  --non_iid --poison_frac 0.2 --rlr_theta 3" 

# "python ./federated.py --agent_frac 0.25 --method rlr --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr avg --device cuda:1 --rounds  400  --non_iid --poison_frac 0.2 --rlr_theta 4"

# "python ./federated.py --agent_frac 0.25 --method rlr --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr avg --device cuda:2 --rounds  400  --non_iid --poison_frac 0.2 --rlr_theta 5" 

# "python ./federated.py --agent_frac 0.25 --method rlr --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr avg --device cuda:3 --rounds  400   --rlr_theta 3" 

# "python ./federated.py --agent_frac 0.25 --method rlr --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr avg --device cuda:4 --rounds  400   --rlr_theta 4"

# "python ./federated.py --agent_frac 0.25 --method rlr --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr avg --device cuda:0 --rounds  400   --rlr_theta 5" 

# "python ./federated.py --agent_frac 0.25 --method rlr --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr avg --device cuda:1 --rounds  400  --non_iid  --rlr_theta 3" 

# "python ./federated.py --agent_frac 0.25 --method rlr --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr avg --device cuda:2 --rounds  400  --non_iid  --rlr_theta 4"

# "python ./federated.py --agent_frac 0.25 --method rlr --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr avg --device cuda:3 --rounds  400  --non_iid  --rlr_theta 5" 

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr mul_krum --device cuda:5 --rounds  400  --non_iid  --mul_krum_remain_cls 4 --poison_frac 0.2" 

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr mul_krum --device cuda:2 --rounds  400  --non_iid  --mul_krum_remain_cls 3 --poison_frac 0.2" 

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr mul_krum --device cuda:1 --rounds  400  --non_iid  --mul_krum_remain_cls 2 --poison_frac 0.2" 

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr mul_krum --device cuda:3 --rounds  400  --non_iid  --mul_krum_remain_cls 4 " 

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr mul_krum --device cuda:4 --rounds  400  --non_iid  --mul_krum_remain_cls 3 " 

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr mul_krum --device cuda:0 --rounds  400  --non_iid  --mul_krum_remain_cls 2 " 



"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr snowball- --device cuda:0 --rounds  400  --non_iid --poison_frac 0.2 --snow_cluster_th 1 --snow_minus_remain_cls 1" 

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr snowball- --device cuda:1 --rounds  400  --non_iid --poison_frac 0.2 --snow_cluster_th 2 --snow_minus_remain_cls 1" 

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr snowball- --device cuda:2 --rounds  400  --non_iid --poison_frac 0.2 --snow_cluster_th 3 --snow_minus_remain_cls 1"

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr snowball- --device cuda:3 --rounds  400  --non_iid --poison_frac 0.2 --snow_cluster_th 5 --snow_minus_remain_cls 1"   

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr snowball- --device cuda:4 --rounds  400  --non_iid --poison_frac 0.2 --snow_cluster_th 1 --snow_minus_remain_cls 2" 

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr snowball- --device cuda:0 --rounds  400  --non_iid --poison_frac 0.2 --snow_cluster_th 2 --snow_minus_remain_cls 2" 

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr snowball- --device cuda:1 --rounds  400  --non_iid --poison_frac 0.2 --snow_cluster_th 3 --snow_minus_remain_cls 2"

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr snowball- --device cuda:2 --rounds  400  --non_iid --poison_frac 0.2 --snow_cluster_th 5 --snow_minus_remain_cls 2"

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr snowball- --device cuda:3 --rounds  400  --non_iid --poison_frac 0.2 --snow_cluster_th 1 --snow_minus_remain_cls 3" 

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr snowball- --device cuda:4 --rounds  400  --non_iid --poison_frac 0.2 --snow_cluster_th 2 --snow_minus_remain_cls 3" 

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr snowball- --device cuda:5 --rounds  400  --non_iid --poison_frac 0.2 --snow_cluster_th 3 --snow_minus_remain_cls 3"

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr snowball- --device cuda:6 --rounds  400  --non_iid --poison_frac 0.2 --snow_cluster_th 5 --snow_minus_remain_cls 3"

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr snowball- --device cuda:7 --rounds  400  --non_iid --poison_frac 0.2 --snow_cluster_th 5 --snow_minus_remain_cls 3"

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr deepsight --device cuda:0 --rounds  400   --ds_tau 0.333333 --ds_num_seed 3 --non_iid "

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr deepsight --device cuda:1 --rounds  400   --ds_tau 0.333333 --ds_num_seed 5 --non_iid "

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr deepsight --device cuda:2 --rounds  400   --ds_tau 0.333333 --ds_num_seed 3 --non_iid --poison_frac 0.2"

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr deepsight --device cuda:3 --rounds  400   --ds_tau 0.333333 --ds_num_seed 5 --non_iid --poison_frac 0.2"

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr deepsight --device cuda:4 --rounds  400   --ds_tau 0.2 --ds_num_seed 3 --non_iid "

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr deepsight --device cuda:0 --rounds  400   --ds_tau 0.2 --ds_num_seed 5 --non_iid "

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr deepsight --device cuda:1 --rounds  400   --ds_tau 0.2 --ds_num_seed 3 --non_iid --poison_frac 0.2"

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr deepsight --device cuda:2 --rounds  400   --ds_tau 0.2 --ds_num_seed 5 --non_iid --poison_frac 0.2"

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr deepsight --device cuda:3 --rounds  400   --ds_tau 0.1 --ds_num_seed 3 --non_iid "

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr deepsight --device cuda:4 --rounds  400   --ds_tau 0.1 --ds_num_seed 5 --non_iid "

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr deepsight --device cuda:0 --rounds  400   --ds_tau 0.1 --ds_num_seed 3 --non_iid --poison_frac 0.2"

"python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr deepsight --device cuda:1 --rounds  400   --ds_tau 0.1 --ds_num_seed 5 --non_iid --poison_frac 0.2"


# "python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr mul_metric --device cuda:0 --rounds  400   --snow_cluster_th 5 --mul_metric_remain_cls 1"

# "python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr mul_metric --device cuda:1 --rounds  400   --snow_cluster_th 5 --mul_metric_remain_cls 2"

# "python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr mul_metric --device cuda:2 --rounds  400   --snow_cluster_th 5 --mul_metric_remain_cls 3"

# "python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr mul_metric --device cuda:3 --rounds  400   --snow_cluster_th 5 --mul_metric_remain_cls 1 --non_iid"

# "python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr mul_metric --device cuda:4 --rounds  400   --snow_cluster_th 5 --mul_metric_remain_cls 2 --non_iid"

# "python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr mul_metric --device cuda:0 --rounds  400   --snow_cluster_th 5 --mul_metric_remain_cls 3 --non_iid"

# "python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr mul_metric --device cuda:1 --rounds  400   --snow_cluster_th 5 --mul_metric_remain_cls 1 --non_iid --poison_frac 0.2"

# "python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr mul_metric --device cuda:2 --rounds  400   --snow_cluster_th 5 --mul_metric_remain_cls 2 --non_iid --poison_frac 0.2"

# "python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr mul_metric --device cuda:3 --rounds  400   --snow_cluster_th 5 --mul_metric_remain_cls 3 --non_iid --poison_frac 0.2"


# "python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr flame --device cuda:2 --rounds  400   --min_claster_size 4 --noise_sigma 0.001 --non_iid --poison_frac 0.2"

# "python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr flame --device cuda:3 --rounds  400   --min_claster_size 2 --noise_sigma 0.001 --non_iid --poison_frac 0.2"

# "python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr flame --device cuda:4 --rounds  400   --min_claster_size 6 --noise_sigma 0.001 --non_iid --poison_frac 0.2"

# "python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr flame --device cuda:0 --rounds  400   --min_claster_size 2 --noise_sigma 0.001 --non_iid "

# "python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr flame --device cuda:1 --rounds  400   --min_claster_size 4 --noise_sigma 0.001 --non_iid "

# "python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr flame --device cuda:2 --rounds  400   --min_claster_size 6 --noise_sigma 0.001 --non_iid "

# "python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr flame --device cuda:3 --rounds  400   --min_claster_size 2 --noise_sigma 0.001  "

# "python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr flame --device cuda:4 --rounds  400   --min_claster_size 4 --noise_sigma 0.001  "

# "python ./federated.py --agent_frac 0.25 --method fed --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr flame --device cuda:5 --rounds  400   --min_claster_size 6 --noise_sigma 0.001  "



# "python ./federated.py --agent_frac 0.25 --method lockdown --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr avg --device cuda:5 --rounds  400  --anneal_factor 0.1"

# "python ./federated.py --agent_frac 0.25 --method lockdown --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr avg --device cuda:5 --rounds  400  --anneal_factor 0.01"

# "python ./federated.py --agent_frac 0.25 --method lockdown --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr avg --device cuda:5 --rounds  400 --non_iid --anneal_factor 0.1"

# "python ./federated.py --agent_frac 0.25 --method lockdown --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr avg --device cuda:5 --rounds  400 --non_iid --anneal_factor 0.01"

# "python ./federated.py --agent_frac 0.25 --method lockdown --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr avg --device cuda:5 --rounds  400 --non_iid --poison_frac 0.2 --anneal_factor 0.1"

# "python ./federated.py --agent_frac 0.25 --method lockdown --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr avg --device cuda:5 --rounds  400 --non_iid --poison_frac 0.2 --anneal_factor 0.01"

# "python ./federated.py --agent_frac 0.25 --method lockdown --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr avg --device cuda:5 --rounds  400  --anneal_factor 0.1 --theta 22"

# "python ./federated.py --agent_frac 0.25 --method lockdown --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr avg --device cuda:5 --rounds  400  --anneal_factor 0.01 --theta 22"

# "python ./federated.py --agent_frac 0.25 --method lockdown --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr avg --device cuda:5 --rounds  400 --non_iid --anneal_factor 0.1 --theta 22" 

# "python ./federated.py --agent_frac 0.25 --method lockdown --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr avg --device cuda:5 --rounds  400 --non_iid --anneal_factor 0.01 --theta 22"

# "python ./federated.py --agent_frac 0.25 --method lockdown --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr avg --device cuda:5 --rounds  400 --non_iid --poison_frac 0.2 --anneal_factor 0.1 --theta 22"

# "python ./federated.py --agent_frac 0.25 --method lockdown --attack badnet --data cifar10 --model resnet_ci --num_corrupt 8  --local_ep 4  --fix_attack  --aggr avg --device cuda:5 --rounds  400 --non_iid --poison_frac 0.2 --anneal_factor 0.01 --theta 22"


)

for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    $cmd &
done

wait  # 等待所有后台任务完成

