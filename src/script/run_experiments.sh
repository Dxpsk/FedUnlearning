#!/bin/bash

commands=(
"python ./federated.py --agent_frac 0.25 --method FDCR_adp --attack neurotoxin  --num_corrupt 8  --local_ep 4  --model resnet_ci --fix_attack  --aggr FDCR --device cuda:7 --rounds 400 --non_iid "

"python ./federated.py --agent_frac 0.25 --method FDCR_adp --attack badnet  --num_corrupt 8  --local_ep 4  --model resnet_ci --fix_attack  --aggr FDCR --device cuda:6 --rounds 400 "

"python ./federated.py --agent_frac 0.25 --method FDCR_adp --attack DBA  --num_corrupt 8  --local_ep 4  --model resnet_ci --fix_attack  --aggr FDCR --device cuda:5 --rounds 400 "

"python ./federated.py --agent_frac 0.25 --method FDCR_adp --attack neurotoxin  --num_corrupt 8  --local_ep 4  --model resnet_ci --fix_attack  --aggr FDCR --device cuda:4 --rounds 400 " 

"python ./federated.py --agent_frac 0.25 --method lockdown_adp --attack badnet  --num_corrupt 8  --local_ep 4  --model resnet_ci --fix_attack  --aggr avg --device cuda:3 --rounds 400 "

"python ./federated.py --agent_frac 0.25 --method lockdown_adp --attack DBA  --num_corrupt 8  --local_ep 4  --model resnet_ci --fix_attack  --aggr avg --device cuda:2 --rounds 400 " 

"python ./federated.py --agent_frac 0.25 --method lockdown_adp --attack neurotoxin  --num_corrupt 8  --local_ep 4  --model resnet_ci --fix_attack  --aggr avg --device cuda:1 --rounds 400 "

"python ./federated.py --agent_frac 0.25 --method lockdown_adp --attack badnet  --num_corrupt 8  --local_ep 4  --model resnet_ci --fix_attack  --aggr avg --device cuda:7 --rounds 400 --non_iid"

"python ./federated.py --agent_frac 0.25 --method lockdown_adp --attack DBA  --num_corrupt 8  --local_ep 4  --model resnet_ci --fix_attack  --aggr avg --device cuda:6 --rounds 400 --non_iid" 

"python ./federated.py --agent_frac 0.25 --method lockdown_adp --attack neurotoxin  --num_corrupt 8  --local_ep 4  --model resnet_ci --fix_attack  --aggr avg --device cuda:5 --rounds 400 --non_iid"
)

for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    $cmd &
done

wait  # 等待所有后台任务完成

