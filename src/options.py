import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data', type=str, default='cifar10',
                        help="dataset we want to train on")
    parser.add_argument('--num_agents', type=int, default=40,
                        help="number of agents:K")
    parser.add_argument('--agent_frac', type=float, default=1,
                        help="fraction of agents per round:C")
    parser.add_argument('--num_corrupt', type=int, default=4,
                        help="number of corrupt agents")
    parser.add_argument('--rounds', type=int, default=200,
                        help="number of communication rounds:R")
    parser.add_argument('--aggr', type=str, default='avg',
                        help="aggregation function to aggregate agents' local weights")
    parser.add_argument('--local_ep', type=int, default=2,
                        help="number of local epochs:E")
    parser.add_argument('--bs', type=int, default=64,
                        help="local batch size: B")
    parser.add_argument('--client_lr', type=float, default=0.1,
                        help='clients learning rate')
    parser.add_argument('--server_lr', type=float, default=1,
                        help='servers learning rate for signSGD')
    parser.add_argument('--target_class', type=int, default=7,
                        help="target class for backdoor attack")
    parser.add_argument('--poison_frac', type=float, default=0.5,
                        help="fraction of dataset to corrupt for backdoor attack")
    parser.add_argument('--pattern_type', type=str, default='plus',
                        help="shape of bd pattern")
    parser.add_argument('--snap', type=int, default=1,
                        help="do inference in every num of snap rounds")
    parser.add_argument('--device',  default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help="To use cuda, set to a specific GPU ID.")
    parser.add_argument('--num_workers', type=int, default=0, 
                        help="num of workers for multithreading")
    parser.add_argument('--method', type=str, default="lockdown",
                        help="method for fed setting")
    parser.add_argument('--non_iid', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--alpha',type=float, default=0.5)
    parser.add_argument('--attack',type=str, default="badnet")
    parser.add_argument('--lr_decay',type=float, default= 1)
    parser.add_argument('--wd', type=float, default= 1e-4)
    parser.add_argument('--cease_poison', type=float, default=100000)
    parser.add_argument('--start_poison', type=float, default=0)
    parser.add_argument('--fix_attack', action='store_true', default=False)
    parser.add_argument('--model', type=str, default="resnet_ci")
    parser.add_argument('--norm_log', action='store_true', default=False)

    
    #unlearning_params
    parser.add_argument('--unl_mode', type=str, default="joint")
    parser.add_argument('--pert_lr', type=float, default=10)
    parser.add_argument('--limit_norm', type=float, default=30)
    parser.add_argument('--fea_w', type=float, default=1)
    parser.add_argument('--unl_weight', type=float, default=1)
    parser.add_argument('--portion', type=float, default=0.01, help="portion of data added with perturbation when doing backdoor unlearning")
    parser.add_argument('--regu_weight', type=float, default=0.001, help="weight for regularization loss in for pert generation")
    parser.add_argument('--norm_limit', action='store_true', default=False, help="norm limit for perturbation") 
    parser.add_argument('--pert_rounds', type=int, default=1, help="number of rounds for perturbation generation")
    parser.add_argument('--mc_norm', type=float, default=30)
    parser.add_argument('--mc_adv_train',action='store_true', default=False)
    parser.add_argument('--pure_poison', type=float, default=0)
    
    #lockdown
    parser.add_argument('--theta', type=int, default=20,
                        help="break ties when votes sum to 0")
    parser.add_argument('--dense_ratio', type=float, default=0.25,)
    parser.add_argument('--anneal_factor', type=float, default=0.0001,)
    parser.add_argument('--same_mask', type=int, default= 1)
    parser.add_argument('--mask_init', type=str, default="ERK")
    parser.add_argument('--dis_check_gradient', action='store_true', default=False)
    parser.add_argument('--se_threshold', type=float, default=1e-4,
                        help="num of workers for multithreading")

    #rlr
    parser.add_argument('--rlr_theta', type=int, default=5)
    
    #multi_krum
    parser.add_argument('--mul_krum_remain_cls', type=int, default=4)
    
    #flame
    parser.add_argument('--min_claster_size', type=int, default=4)
    parser.add_argument('--noise_sigma', type=float, default=0.001)
    
    #snowball-
    parser.add_argument('--snow_cluster_th', type=int, default=2)
    parser.add_argument('--snow_minus_remain_cls', type=int, default=3)
    #snowball
    parser.add_argument('--snow_remain_cls', type=int, default=5)
    parser.add_argument('--vae_hidden', type=int, default=256)
    parser.add_argument('--vae_latent', type=int, default=64)
    parser.add_argument('--vae_initial', type=int, default=270)
    parser.add_argument('--vae_tuning', type=int, default=30)
    
    #deepsight
    parser.add_argument('--ds_tau', type=float, default=0.333333)
    parser.add_argument('--ds_num_seed', type=int, default=3)
    
    #mul_metric
    parser.add_argument('--mul_metric_remain_cls', type=int, default=3)

    #flip
    parser.add_argument('--prob', type=float, default=0.4)
    args = parser.parse_args()
    return args