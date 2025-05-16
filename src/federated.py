import utils
import models
import math
import copy
import numpy as np
from agent import Agent
from agent_sparse import Agent as Agent_s
from agent_sparse_adaptive import Agent as Agent_s_a
from agent_unlearning import Agent as Agent_unl
from options import args_parser
from aggregation import Aggregation
import torch
import random
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
import logging                                          
from agent_FDCR import Agent as Agent_FDCR
from agent_FDCR_adaptive import Agent as Agent_FDCR_a
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
from A3FL import Attacker
import os

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    args = args_parser()
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    
    if args.non_iid:
        logPath = "non_iid_logs"
    else:
        logPath = "logs"  
    if args.method == "unlearning":
        fileName = "AckR{}_{}_{}_{}_{}_{}_alpha{}_Epo{}_inject{}_Agg{}_noniid{}_att{}_fix_att{}_mode{}"\
                    "pert_lr{}_patt{}_norm_l{}_fea_w{}_portion{}_per_r{}_regu_w{}_norm_l{}_start_p{}_mc_adv{}_mc_n{}_poi{}".format(args.num_corrupt, args.num_agents, args.agent_frac,
            args.method, args.data, args.model, args.alpha, args.local_ep, args.poison_frac,  args.aggr, args.non_iid, args.attack, 
            args.fix_attack, args.unl_mode, args.pert_lr, args.pattern_type, args.limit_norm, args.fea_w, args.portion, args.pert_rounds, args.regu_weight
            , args.norm_limit, args.start_poison, args.mc_adv_train, args.mc_norm, args.pure_poison)
    elif args.method == "rlr":
        fileName = "AckRatio{}_{}_{}_{}_{}_{}_alpha{}_Epoch{}_inject{}_Agg{}_noniid{}_attack{}_fix_att{}_pattern{}_start_p{}_rlr_theta{}".format(
            args.num_corrupt, args.num_agents, args.agent_frac, args.method, args.data, args.model, args.alpha, args.local_ep, args.poison_frac,
            args.aggr, args.non_iid, args.attack, args.fix_attack, args.pattern_type, args.start_poison, args.rlr_theta)
    elif args.method == "lockdown":
        fileName = "AckRatio{}_{}_{}_{}_{}_{}_alpha{}_Epoch{}_inject{}_Agg{}_noniid{}_attack{}_fix_att{}_pattern{}_start_p{}_theta{}_anneal{}_dense_ratio{}".format(
            args.num_corrupt, args.num_agents, args.agent_frac, args.method, args.data, args.model, args.alpha, args.local_ep, args.poison_frac,
            args.aggr, args.non_iid, args.attack, args.fix_attack, args.pattern_type, args.start_poison, args.theta,args.anneal_factor, args.dense_ratio)
    elif args.aggr == "mul_krum":
        fileName = "AckRatio{}_{}_{}_{}_{}_{}_alpha{}_Epoch{}_inject{}_Agg{}_noniid{}_attack{}_fix_att{}_pattern{}_start_p{}_remain_clients{}".format(
            args.num_corrupt, args.num_agents, args.agent_frac, args.method, args.data, args.model, args.alpha, args.local_ep, args.poison_frac,
            args.aggr, args.non_iid, args.attack, args.fix_attack, args.pattern_type, args.start_poison, args.mul_krum_remain_cls)
    elif args.aggr == "flame":
        fileName = "AckRatio{}_{}_{}_{}_{}_{}_alpha{}_Epoch{}_inject{}_Agg{}_noniid{}_attack{}_fix_att{}_pattern{}_start_p{}_min_claster_size{}_noise_sig{}".format(
            args.num_corrupt, args.num_agents, args.agent_frac, args.method, args.data, args.model, args.alpha, args.local_ep, args.poison_frac,
            args.aggr, args.non_iid, args.attack, args.fix_attack, args.pattern_type, args.start_poison, args.min_claster_size, args.noise_sigma)
    elif args.aggr == "snowball-":
        fileName = "AckRatio{}_{}_{}_{}_{}_{}_alpha{}_Epoch{}_inject{}_Agg{}_noniid{}_attack{}_fix_att{}_pattern{}_start_p{}_snow_cluster_th{}_minus_remain_cls{}".format(
            args.num_corrupt, args.num_agents, args.agent_frac, args.method, args.data, args.model, args.alpha, args.local_ep, args.poison_frac,
            args.aggr, args.non_iid, args.attack, args.fix_attack, args.pattern_type, args.start_poison, args.snow_cluster_th, args.snow_minus_remain_cls)
    elif args.aggr == "snowball":
        fileName = "AckRatio{}_{}_{}_{}_{}_{}_alpha{}_Epoch{}_inject{}_Agg{}_noniid{}_attack{}_fix_att{}_pattern{}_start_p{}_snow_cluster_th{}_minus_remain_cls{}_remain_cls{}_hidden{}"\
                    "_latent{}_initial{}_tuning{}".format(
            args.num_corrupt, args.num_agents, args.agent_frac, args.method, args.data, args.model, args.alpha, args.local_ep, args.poison_frac,
            args.aggr, args.non_iid, args.attack, args.fix_attack, args.pattern_type, args.start_poison, args.snow_cluster_th, args.snow_minus_remain_cls
            , args.snow_remain_cls, args.vae_hidden ,args.vae_latent, args.vae_initial, args.vae_tuning)
    elif args.aggr == "deepsight":
        fileName = "AckRatio{}_{}_{}_{}_{}_{}_alpha{}_Epoch{}_inject{}_Agg{}_noniid{}_attack{}_fix_att{}_pattern{}_start_p{}_ds_tau{}_ds_num_seed{}".format(
            args.num_corrupt, args.num_agents, args.agent_frac, args.method, args.data, args.model, args.alpha, args.local_ep, args.poison_frac,
            args.aggr, args.non_iid, args.attack, args.fix_attack, args.pattern_type, args.start_poison, args.ds_tau, args.ds_num_seed)
    elif args.aggr == "mul_metric":
        fileName = "AckRatio{}_{}_{}_{}_{}_{}_alpha{}_Epoch{}_inject{}_Agg{}_noniid{}_attack{}_fix_att{}_pattern{}_start_p{}_mul_metric_remain_cls{}".format(
            args.num_corrupt, args.num_agents, args.agent_frac, args.method, args.data, args.model, args.alpha, args.local_ep, args.poison_frac,
            args.aggr, args.non_iid, args.attack, args.fix_attack, args.pattern_type, args.start_poison, args.mul_metric_remain_cls)
    elif args.method == "flip":
        fileName = "AckRatio{}_{}_{}_{}_{}_{}_alpha{}_Epoch{}_inject{}_Agg{}_noniid{}_attack{}_fix_att{}_pattern{}_start_p{}_prob{}".format(
            args.num_corrupt, args.num_agents, args.agent_frac, args.method, args.data, args.model, args.alpha, args.local_ep, args.poison_frac,
             args.aggr, args.non_iid, args.attack, args.fix_attack, args.pattern_type, args.start_poison, args.prob)     
    else:
        fileName = "AckRatio{}_{}_{}_{}_{}_{}_alpha{}_Epoch{}_inject{}_Agg{}_noniid{}_attack{}_fix_att{}_pattern{}_start_p{}".format(
            args.num_corrupt, args.num_agents, args.agent_frac, args.method, args.data, args.model, args.alpha, args.local_ep, args.poison_frac,
             args.aggr, args.non_iid, args.attack, args.fix_attack, args.pattern_type, args.start_poison)     
        # dir_name = ".".join(fileName.split(".")[:-1])
    dir_name = fileName
    if not os.path.exists(os.path.join("result", dir_name)):
        os.makedirs(os.path.join("result", dir_name))
    setattr(args, "result_dir", os.path.join("result", dir_name))
    if not args.debug:
        fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, fileName))
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)
    logging.info(args)

    cum_poison_acc_mean = 0
    
    # load dataset and user groups (i.e., user to data mapping)
    train_dataset, val_dataset = utils.get_datasets(args.data)
    if args.data == "cifar100":
        num_target = 100
    elif args.data == "tinyimagenet":
        num_target = 200
    elif args.data == "GTSRB":
        num_target = 43
    else:
        num_target = 10
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
                            pin_memory=False)
    if args.non_iid:
        user_groups, dict_user_per_class  = utils.distribute_data_dirichlet(train_dataset, args)
        print([len(data_idx) for data_idx in user_groups.values()])
    else:
        user_groups = utils.distribute_data(train_dataset, args, n_classes=num_target)
        # print(user_groups)
    idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
    # logging.info(idxs)
    if args.data != "tinyimagenet" and args.attack != "A3FL":
        # poison the validation dataset
        poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
        utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
    elif args.data != "tinyimagenet" and args.attack == "A3FL":
        poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
    else:
        poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs, runtime_poison=True, args=args)

    poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
                                     pin_memory=False)
    if args.data != "tinyimagenet":
        idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
        poisoned_val_set_only_x = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
        utils.poison_dataset(poisoned_val_set_only_x.dataset, args, idxs, poison_all=True, modify_label=False)
    else:
        poisoned_val_set_only_x = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs, runtime_poison=True, args=args, modify_label =False)

    poisoned_val_only_x_loader = DataLoader(poisoned_val_set_only_x, batch_size=args.bs, shuffle=False,
                                            num_workers=args.num_workers,
                                            pin_memory=False)

    # initialize a model, and the agents
    global_model = models.get_model(args.data, args.model).to(args.device)
    # global_model.load_state_dict(torch.load("AckRatio8_40_0.25_fed_cifar10_resnet_ci_alpha0.5_Rnd400_Epoch4_inject0.5_dense0.25_Aggflame_noniidFalse_attackbadnet_fix_attTrue_start_p0.pt")["model_state_dict"])
    
    start_round = 0
    load_checkpoint = False
    if load_checkpoint:
        start_round = 50
        if args.non_iid:
            checkpoint_dir = "non_iid_checkpoint"
        else:
            checkpoint_dir = "checkpoint"
            
        # load_path = checkpoint_dir + "/AckRatio{}_{}_{}_{}_{}_{}_alpha{}_Rnd{}_Epoch{}_inject{}_dense{}_Agg{}_noniid{}_attack{}_fix_att{}_pattern{}_start_p{}.pt".format(
        #             args.num_corrupt, args.num_agents, args.agent_frac, args.method, args.data, args.model, args.alpha, start_round ,args.local_ep, args.poison_frac,
        #             args.dense_ratio, args.aggr, args.non_iid, args.attack, args.fix_attack, args.pattern_type ,args.start_poison)
        # if os.path.exists(load_path):
        #     global_model.load_state_dict(torch.load(load_path)["model_state_dict"])
        #     logging.info(f"load model from {load_path}")
        # else:            
        #     load_path = checkpoint_dir + "/AckRatio{}_{}_{}_{}_{}_{}_alpha{}_Rnd{}_Epoch{}_inject{}_dense{}_Agg{}_noniid{}_attack{}_fix_att{}_start_p{}.pt".format(
        #                 args.num_corrupt, args.num_agents, args.agent_frac, args.method, args.data, args.model, args.alpha, start_round ,args.local_ep, args.poison_frac,
        #                 args.dense_ratio, args.aggr, args.non_iid, args.attack, args.fix_attack ,args.start_poison)
        #     global_model.load_state_dict(torch.load(load_path)["model_state_dict"])
        #     logging.info(f"load model from {load_path}")
        load_path = checkpoint_dir + "/AckRatio8_40_0.25_flip_fmnist_CNN_alpha0.5_Epoch4_inject0.2_Aggavg_noniidTrue_attackbadnet_fix_attTrue_patternplus_start_p0_prob0.0_Rnd50.pt"
        global_model.load_state_dict(torch.load(load_path)["model_state_dict"])
        logging.info(f"load model from {load_path}")
    
    
    global_mask = {}
    neurotoxin_mask = {}
    updates_dict = {}
    n_model_params = len(parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]))
    params = {name: copy.deepcopy(global_model.state_dict()[name]) for name in global_model.state_dict()}
    if args.method == "lockdown" or args.method == "lockdown_adp":
        sparsity = utils.calculate_sparsities(args, params, distribution=args.mask_init)
        mask = utils.init_masks(params, sparsity)
    agents, agent_data_sizes = [], {}
    for _id in range(0, args.num_agents):
        if args.method == "lockdown":
            if args.same_mask==0:
                agent = Agent_s(_id, args, train_dataset, user_groups[_id], mask=utils.init_masks(params, sparsity))
            else:
                agent = Agent_s(_id, args, train_dataset, user_groups[_id], mask=mask)
        elif args.method == "lockdown_adp":
            if args.same_mask==0:
                agent = Agent_s_a(_id, args, train_dataset, user_groups[_id], mask=utils.init_masks(params, sparsity))
            else:
                agent = Agent_s_a(_id, args, train_dataset, user_groups[_id], mask=mask)
        elif args.method == "FDCR":
            agent = Agent_FDCR(_id, args, train_dataset, user_groups[_id])
        elif args.method == "FDCR_adp":
            agent = Agent_FDCR_a(_id, args, train_dataset, user_groups[_id])
        elif args.method == "unlearning":
            agent = Agent_unl(_id, args, train_dataset, user_groups[_id])
        else:
            agent = Agent(_id, args, train_dataset, user_groups[_id])
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent)

        # aggregation server and the loss function

    aggregator = Aggregation(agent_data_sizes, n_model_params, poisoned_val_loader, args, None)
    criterion = nn.CrossEntropyLoss().to(args.device)
    agent_updates_list = []
    worker_id_list = []
    agent_updates_dict = {}
    mask_aggrement = []

    acc_vec = []
    asr_vec = []
    pacc_vec = []
    per_class_vec = []
    acc_vec_with_th = []
    asr_vec_with_th = []

    clean_asr_vec = []
    clean_acc_vec = []
    clean_pacc_vec = []
    clean_per_class_vec = []

    if args.attack == "A3FL":
        attacker = Attacker(args)
    else:
        attacker = None
    
    for rnd in range(start_round + 1, start_round + args.rounds + 1):
        logging.info("--------round {} ------------".format(rnd))
        # mask = torch.ones(n_model_params)
        rnd_global_params = parameters_to_vector([ copy.deepcopy(global_model.state_dict()[name]) for name in global_model.state_dict()])
        agent_updates_dict = {}
        local_fish_dict = {}
        # chosen = np.random.choice(args.num_agents, math.floor(args.num_agents * args.agent_frac), replace=False)
        chosen = utils.choose_agent(args)
        logging.info(f"choose agent : {[int(client) for client in chosen]}")
        if args.method == "lockdown" or args.method == "lockdown_adp":
            old_mask = [copy.deepcopy(agent.mask) for agent in agents]
        for agent_id in chosen:
            # logging.info(torch.sum(rnd_global_params))
            global_model = global_model.to(args.device)
            if args.method == "lockdown" or args.method == "lockdown_adp":
                update = agents[agent_id].local_train(global_model, criterion, rnd, global_mask=global_mask, neurotoxin_mask = neurotoxin_mask, updates_dict=updates_dict, attacker=attacker)
            elif args.method == "FDCR" or args.method == "FDCR_adp":
                update, fish_dict = agents[agent_id].local_train(global_model, criterion, rnd, neurotoxin_mask = neurotoxin_mask, attacker=attacker)
                local_fish_dict[agent_id] = fish_dict
            else:
                update = agents[agent_id].local_train(global_model, criterion, rnd, neurotoxin_mask=neurotoxin_mask, attacker=attacker)
            agent_updates_dict[agent_id] = update
            utils.vector_to_model(copy.deepcopy(rnd_global_params), global_model)
        if args.method == "lockdown" or args.method == "lockdown_adp":
            updates_dict,neurotoxin_mask = aggregator.aggregate_updates(global_model, agent_updates_dict, rnd)
        elif args.method == "FDCR" or args.method == "FDCR_adp":
            updates_dict,neurotoxin_mask = aggregator.aggregate_updates(global_model, agent_updates_dict, rnd, local_fish_dict)
        else:
            updates_dict,neurotoxin_mask = aggregator.aggregate_updates(global_model, agent_updates_dict, rnd)
        worker_id_list.append(agent_id + 1)

        # inference in every args.snap rounds
        if rnd % args.snap == 0:
            logging.info(f"global model metrics in round {rnd}")
            val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(global_model, criterion, val_loader,
                                                                                  args, rnd, num_target)
            logging.info(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
            logging.info(f'| Val_Per_Class_Acc: {val_per_class_acc} ')
            acc_vec.append(val_acc)
            per_class_vec.append(val_per_class_acc)

            if args.attack == "A3FL" and rnd >= args.start_poison:
                Attack_Loss, (asr, _), fail_samples = utils.get_loss_n_accuracy(global_model, criterion,
                                                                                poisoned_val_loader, args, rnd, num_target, attacker = attacker)
            else:
                Attack_Loss, (asr, _), fail_samples = utils.get_loss_n_accuracy(global_model, criterion,
                                                                                poisoned_val_loader, args, rnd, num_target)
            cum_poison_acc_mean += asr
            asr_vec.append(asr)
            logging.info(f'| Attack Loss/Attack Success Ratio: {Attack_Loss:.3f} / {asr:.3f} |')

            if args.attack == "A3FL" :
                pass
            else:
                poison_loss, (poison_acc, _), fail_samples = utils.get_loss_n_accuracy(global_model, criterion,
                                                                                   poisoned_val_only_x_loader, args,
                                                                                   rnd, num_target)
                pacc_vec.append(poison_acc)
                logging.info(f'| Poison Loss/Poison accuracy: {poison_loss:.3f} / {poison_acc:.3f} |')

            if args.method == "flip":
                val_acc_with_th = utils.get_loss_n_accuracy_with_prob(global_model, criterion, val_loader,
                                                                                  args.device,  num_target, prob=args.prob)
                logging.info(f'Val_Acc_with_th: {val_acc_with_th:.3f} |')
                acc_vec_with_th.append(val_acc_with_th)
                asr_with_th = utils.get_loss_n_accuracy_with_prob(global_model, criterion, poisoned_val_loader, args.device,
                                                                  num_target, prob=args.prob)
                logging.info(f'Attack Success Ratio with th: {asr_with_th:.3f} |')
                asr_vec_with_th.append(asr_with_th)

            if args.method == "lockdown" or args.method == "lockdown_adp":
                test_model = copy.deepcopy(global_model)
                for name, param in test_model.named_parameters():
                    mask = 0
                    for id, agent in enumerate(agents):
                        mask += old_mask[id][name].to(args.device)
                    param.data = torch.where(mask.to(args.device) >= args.theta, param,
                                             torch.zeros_like(param))
                    # logging.info(torch.sum(mask.to(args.device) >= args.theta) / torch.numel(mask))
                val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(test_model, criterion,
                                                                                      val_loader,
                                                                                      args, rnd, num_target)
                # writer.add_scalar('Clean Validation/Loss', val_loss, rnd)
                # writer.add_scalar('Clean Validation/Accuracy', val_acc, rnd)
                logging.info(f'| Clean Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
                logging.info(f'| Clean Val_Per_Class_Acc: {val_per_class_acc} ')
                clean_acc_vec.append(val_acc)
                clean_per_class_vec.append(val_per_class_acc)

                if args.attack == "A3FL" :
                    Attack_Loss, (asr, _), fail_samples = utils.get_loss_n_accuracy(test_model, criterion,
                                                                                    poisoned_val_loader, args, rnd, num_target, attacker = attacker)
                else:
                    Attack_Loss, (asr, _), fail_samples = utils.get_loss_n_accuracy(test_model, criterion,
                                                                                    poisoned_val_loader, args, rnd, num_target)
                clean_asr_vec.append(asr)
                cum_poison_acc_mean += asr
                logging.info(f'| Clean Attack Success Ratio: {Attack_Loss:.3f} / {asr:.3f} |')

                if args.attack == "A3FL" :
                    # poison_loss, (poison_acc, _), fail_samples = utils.get_loss_n_accuracy(test_model, criterion,
                    #                                                                 val_loader, args,
                    #                                                                 rnd, num_target, attacker=attacker)
                    pass 
                else:
                    poison_loss, (poison_acc, _), fail_samples = utils.get_loss_n_accuracy(test_model, criterion,
                                                                                    poisoned_val_only_x_loader, args,
                                                                                    rnd, num_target)
                    clean_pacc_vec.append(poison_acc)
                    logging.info(f'| Clean Poison Loss/Clean Poison accuracy: {poison_loss:.3f} / {poison_acc:.3f} |')
                # ask the guys to finetune the classifier
                del test_model

        save_frequency = 100 if "cifar" in args.data else 50
        if rnd % save_frequency == 0 or rnd == 1:
            if args.non_iid:
                checkpoint_dir = "non_iid_checkpoint"
            else:
                checkpoint_dir = "checkpoint"
            if args.method == "unlearning":
                PATH = checkpoint_dir + "/AckR{}_{}_{}_{}_{}_{}_alpha{}_Epo{}_inject{}_Agg{}_noniid{}_att{}_fix_att{}_mode{}"\
                    "pert_lr{}_pat{}_norm_l{}_fea_w{}_port{}_per_r{}_regu_w{}_norm_l{}_start_p{}_mc_adv{}_mc_n{}_poi{}_Rnd{}.pt".format(args.num_corrupt, args.num_agents, args.agent_frac,
            args.method, args.data, args.model, args.alpha, args.local_ep, args.poison_frac,  args.aggr, args.non_iid, args.attack, 
            args.fix_attack, args.unl_mode, args.pert_lr, args.pattern_type, args.limit_norm, args.fea_w, args.portion, args.pert_rounds, args.regu_weight
            , args.norm_limit, args.start_poison, args.mc_adv_train, args.mc_norm, args.pure_poison ,rnd)
            elif args.method == "rlr":
                PATH = checkpoint_dir + "/AckRatio{}_{}_{}_{}_{}_{}_alpha{}_Epoch{}_inject{}_Agg{}_noniid{}_attack{}_fix_att{}_pattern{}_start_p{}_rlr_theta{}_Rnd{}.pt".format(
                    args.num_corrupt, args.num_agents, args.agent_frac, args.method, args.data, args.model, args.alpha, args.local_ep, args.poison_frac,
                    args.aggr, args.non_iid, args.attack, args.fix_attack, args.pattern_type, args.start_poison, args.rlr_theta, rnd)
            elif args.method == "lockdown":
                PATH = checkpoint_dir + "/AckRatio{}_{}_{}_{}_{}_{}_alpha{}_Epoch{}_inject{}_Agg{}_noniid{}_attack{}_fix_att{}_pattern{}_start_p{}_theta{}_anneal{}_dense_ratio{}_Rnd{}.pt".format(
                    args.num_corrupt, args.num_agents, args.agent_frac, args.method, args.data, args.model, args.alpha, args.local_ep, args.poison_frac,
                    args.aggr, args.non_iid, args.attack, args.fix_attack, args.pattern_type, args.start_poison, args.theta,args.anneal_factor, args.dense_ratio, rnd)
            elif args.aggr == "mul_krum":
                PATH = checkpoint_dir + "/AckRatio{}_{}_{}_{}_{}_{}_alpha{}_Epoch{}_inject{}_Agg{}_noniid{}_attack{}_fix_att{}_pattern{}_start_p{}_remain_clients{}_Rnd{}.pt".format(
                    args.num_corrupt, args.num_agents, args.agent_frac, args.method, args.data, args.model, args.alpha, args.local_ep, args.poison_frac,
                    args.aggr, args.non_iid, args.attack, args.fix_attack, args.pattern_type, args.start_poison, args.mul_krum_remain_cls, rnd)
            elif args.aggr == "flame":
                PATH = checkpoint_dir + "/AckRatio{}_{}_{}_{}_{}_{}_alpha{}_Epoch{}_inject{}_Agg{}_noniid{}_attack{}_fix_att{}_pattern{}_start_p{}_min_claster_size{}_noise_sig{}_Rnd{}.pt".format(
                    args.num_corrupt, args.num_agents, args.agent_frac, args.method, args.data, args.model, args.alpha, args.local_ep, args.poison_frac,
                    args.aggr, args.non_iid, args.attack, args.fix_attack, args.pattern_type, args.start_poison, args.min_claster_size, args.noise_sigma, rnd)
            elif args.aggr == "snowball-":
                PATH = checkpoint_dir + "/AckRatio{}_{}_{}_{}_{}_{}_alpha{}_Epoch{}_inject{}_Agg{}_noniid{}_attack{}_fix_att{}_pattern{}_start_p{}_snow_cluster_th{}_minus_remain_cls{}_Rnd{}.pt".format(
                    args.num_corrupt, args.num_agents, args.agent_frac, args.method, args.data, args.model, args.alpha, args.local_ep, args.poison_frac,
                    args.aggr, args.non_iid, args.attack, args.fix_attack, args.pattern_type, args.start_poison, args.snow_cluster_th, args.snow_minus_remain_cls, rnd)
            elif args.aggr == "snowball":
                PATH = checkpoint_dir + "/AckRatio{}_{}_{}_{}_{}_{}_alpha{}_Epoch{}_inject{}_Agg{}_noniid{}_attack{}_fix_att{}_pattern{}_start_p{}_snow_cluster_th{}_minus_remain_cls{}_remain_cls{}_hidden{}"\
                        "_latent{}_initial{}_tuning{}_Rnd{}.pt".format(
                    args.num_corrupt, args.num_agents, args.agent_frac, args.method, args.data, args.model, args.alpha, args.local_ep, args.poison_frac,
                    args.aggr, args.non_iid, args.attack, args.fix_attack, args.pattern_type, args.start_poison, args.snow_cluster_th, args.snow_minus_remain_cls, args.snow_remain_cls, args.vae_hidden ,args.vae_latent, args.vae_initial, args.vae_tuning, rnd)
            elif args.aggr == "deepsight":
                PATH = checkpoint_dir + "/AckRatio{}_{}_{}_{}_{}_{}_alpha{}_Epoch{}_inject{}_Agg{}_noniid{}_attack{}_fix_att{}_pattern{}_start_p{}_ds_tau{}_ds_num_seed{}_Rnd{}.pt".format(
                    args.num_corrupt, args.num_agents, args.agent_frac, args.method, args.data, args.model, args.alpha, args.local_ep, args.poison_frac,
                    args.aggr, args.non_iid, args.attack, args.fix_attack, args.pattern_type, args.start_poison, args.ds_tau, args.ds_num_seed, rnd)   
            elif args.aggr == "mul_metric":
                PATH = checkpoint_dir + "/AckRatio{}_{}_{}_{}_{}_{}_alpha{}_Epoch{}_inject{}_Agg{}_noniid{}_attack{}_fix_att{}_pattern{}_start_p{}_mul_metric_remain_cls{}_Rnd{}.pt".format(
                    args.num_corrupt, args.num_agents, args.agent_frac, args.method, args.data, args.model, args.alpha, args.local_ep, args.poison_frac,
                    args.aggr, args.non_iid, args.attack, args.fix_attack, args.pattern_type, args.start_poison, args.mul_metric_remain_cls, rnd)   
            elif args.method == "flip":
                PATH= checkpoint_dir + "/AckRatio{}_{}_{}_{}_{}_{}_alpha{}_Epoch{}_inject{}_Agg{}_noniid{}_attack{}_fix_att{}_pattern{}_start_p{}_prob{}_Rnd{}.pt".format(
                    args.num_corrupt, args.num_agents, args.agent_frac, args.method, args.data, args.model, args.alpha,  args.local_ep, args.poison_frac,
                     args.aggr, args.non_iid, args.attack, args.fix_attack, args.pattern_type, args.start_poison,args.prob ,rnd)
            else:
                PATH = checkpoint_dir + "/AckRatio{}_{}_{}_{}_{}_{}_alpha{}_Epoch{}_inject{}_Agg{}_noniid{}_attack{}_fix_att{}_pattern{}_start_p{}_Rnd{}.pt".format(
                    args.num_corrupt, args.num_agents, args.agent_frac, args.method, args.data, args.model, args.alpha,  args.local_ep, args.poison_frac,
                     args.aggr, args.non_iid, args.attack, args.fix_attack, args.pattern_type, args.start_poison ,rnd)
            if args.method == "lockdown" or args.method == "lockdown_adp":
                torch.save({
                    'option': args,
                    'model_state_dict': global_model.state_dict(),
                    # 'masks': [agent.mask for agent in agents],
                    'acc_vec': acc_vec,
                    "asr_vec": asr_vec,
                    'pacc_vec ': pacc_vec,
                    "per_class_vec": per_class_vec,
                    "clean_asr_vec": clean_asr_vec,
                    'clean_acc_vec': clean_acc_vec,
                    'clean_pacc_vec ': clean_pacc_vec,
                    'clean_per_class_vec': clean_per_class_vec,
                }, PATH)
            elif args.method == "unlearning":
                torch.save({
                    'option': args,
                    'model_state_dict': global_model.state_dict(),
                    'acc_vec': acc_vec,
                    'asr_vec': asr_vec,
                    'pacc_vec ': pacc_vec,
                    'per_class_vec': per_class_vec,
                    'neurotoxin_mask': neurotoxin_mask, 
                    'norm_record': aggregator.norm_record,
                    'id_record': aggregator.id_record,
                    'unl_loss': [agent.unlloss_dict for agent in agents],
                    'bkd_loss': [agent.bckloss_dict for  agent in agents],
                }, PATH)
            elif args.method == "flip":
                torch.save({
                    'option': args,
                    'model_state_dict': global_model.state_dict(),
                    'acc_vec': acc_vec,
                    'asr_vec': asr_vec,
                    'pacc_vec ': pacc_vec,
                    'per_class_vec': per_class_vec,
                    'acc_vec_with_th': acc_vec_with_th,
                    'asr_vec_with_th': asr_vec_with_th
                }, PATH)
            else:
                torch.save({
                    'option': args,
                    'model_state_dict': global_model.state_dict(),
                    'acc_vec': acc_vec,
                    'asr_vec': asr_vec,
                    'pacc_vec ': pacc_vec,
                    'per_class_vec': per_class_vec,
                    'neurotoxin_mask': neurotoxin_mask, 
                    'norm_record': aggregator.norm_record,
                    'id_record': aggregator.id_record
                }, PATH)

    logging.info('Training has finished!')
