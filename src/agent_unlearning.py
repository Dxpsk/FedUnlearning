import copy
import math
import time
import random
import torch
import models
import utils
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
import torch.nn as nn
import logging
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from torch.utils.data import Dataset
from math import floor

class Agent():
    def __init__(self, id, args, train_dataset=None, data_idxs=None, mask=None, val_loader=None, poisoned_val_loader=None):
        self.id = id
        self.args = args
        self.error = 0
        self.val_loader = val_loader
        self.poisoned_val_loader = poisoned_val_loader
        self.unlloss_dict = {}
        self.bckloss_dict = {}
        
        # get datasets, fedemnist is handled differently as it doesn't come with pytorch
        if train_dataset is None:
            self.train_dataset = torch.load(f'../data/Fed_EMNIST/user_trainsets/user_{id}_trainset.pt')
            # for backdoor attack, agent poisons his local dataset
            if self.id < args.num_corrupt:
                utils.poison_dataset(self.train_dataset, args, data_idxs, agent_idx=self.id)

        else:
            if self.args.data != "tinyimagenet"  and self.id >= args.num_corrupt:
                self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)
            elif self.args.data != "tinyimagenet"  and  self.id < args.num_corrupt:
                if "adp" in self.args.unl_mode:
                    self.train_dataset = DatasetSplit(train_dataset, data_idxs)
                else:
                    self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)
                # for backdoor attack, agent poisons his local dataset
            if self.args.data != "tinyimagenet":
                if self.id < args.num_corrupt:
                    self.clean_backup_dataset = copy.deepcopy(self.train_dataset)
                    self.data_idxs = data_idxs
                    self.poison_idx = utils.poison_dataset(train_dataset, args, data_idxs, agent_idx=self.id)
                    self.poison_dataset = copy.deepcopy(self.train_dataset)
            else:
                self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs, runtime_poison=True, args=args,
                                                        client_id=id)
        # get dataloader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                       num_workers=args.num_workers, pin_memory=False, drop_last=True)
        # size of local dataset
        self.n_data = len(self.train_dataset)

    def check_poison_timing(self, round):
        if round < self.args.start_poison:
            self.train_dataset = self.clean_backup_dataset
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                           num_workers=self.args.num_workers, pin_memory=False, drop_last=True)
        elif round >= self.args.start_poison and round <= self.args.cease_poison:
            self.train_dataset = self.poison_dataset
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                           num_workers=self.args.num_workers, pin_memory=False, drop_last=True)
        else:
            self.train_dataset = self.clean_backup_dataset
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True, \
                                           num_workers=self.args.num_workers, pin_memory=False, drop_last=True)
            

    def _model_dist_norm_var(self, model, target_params_var, norm=2):
        size = 0
        for name, param in model.named_parameters():
            size += param.view(-1).shape[0]
        sum_var = torch.cuda.FloatTensor(size).fill_(0).to(self.args.device)
        size = 0
        for name, layer in model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (
            layer - target_params_var[name]).view(-1)
            size += layer.view(-1).shape[0]
            
        return torch.norm(sum_var, norm)
    
    def get_fea_mean(self, model):
        model.eval()
        features_list = []
        with torch.no_grad():
            for _, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(device=self.args.device, non_blocking=True), \
                                labels.to(device=self.args.device, non_blocking=True)
                features = model.get_features(inputs)
                features_list.append(features)
        features_tensor = torch.cat(features_list, dim=0)
        features_mean = torch.mean(features_tensor, dim=0)
        return features_mean
                

    def local_train(self, global_model, criterion, rnd=None, neurotoxin_mask=FileNotFoundError, attacker = None):
        """ Do a local training over the received global model, return the update """
        logging.info(f"client id: {self.id}")
        # global_model.eval()
        # self.local_test(global_model)
        env_config = utils.get_env(self.args)
        initial_global_model_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        # target_params_variables = dict()
        # for name, param in global_model.state_dict().items():
        #     target_params_variables[name] = param.clone()
        mse = torch.nn.MSELoss()
        
        if self.args.attack == "A3FL" and self.id < self.args.num_corrupt and rnd >= self.args.start_poison and rnd < self.args.cease_poison:
            if attacker.update_flag:
                logging.info(f"Agent {self.id} is searching for trigger")
                search_model = copy.deepcopy(global_model)
                search_dataloader = DataLoader(self.clean_backup_dataset, batch_size=self.args.bs, shuffle=True,
                                       num_workers=self.args.num_workers, pin_memory=False, drop_last=True)
                attacker.search_trigger(search_model, search_dataloader)
                attacker.update_flag = False
                self.train_loader = DataLoader(self.clean_backup_dataset, batch_size=self.args.bs, shuffle=True,
                                            num_workers=self.args.num_workers, pin_memory=False, drop_last=True)
            else:
                logging.info(f"Agent {self.id} will use the trigger")
                self.train_loader = DataLoader(self.clean_backup_dataset, batch_size=self.args.bs, shuffle=True,
                                               num_workers=self.args.num_workers, pin_memory=False, drop_last=True)
        elif self.args.attack == "A3FL" and self.id < self.args.num_corrupt and (rnd < self.args.start_poison or rnd >= self.args.cease_poison):
            self.train_loader = DataLoader(self.clean_backup_dataset, batch_size=self.args.bs, shuffle=True,
                                           num_workers=self.args.num_workers, pin_memory=False, drop_last=True)
        
        if self.id < self.args.num_corrupt and self.args.attack !="A3FL":
            self.check_poison_timing(rnd)
        global_model.train()
        optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr * (self.args.lr_decay) ** rnd,
                                    weight_decay=self.args.wd)
        if self.id >= self.args.num_corrupt:
            if self.args.unl_mode == "tt_unl" or self.args.unl_mode == "tt_unl_adp":
                start = time.time()
                for lc_rnd in range(self.args.local_ep):
                    batch_pert = torch.zeros([1, env_config["input_channel"], env_config["input_height"], env_config["input_width"]], requires_grad=True,
                                                device = self.args.device)
                    batch_opt = torch.optim.Adam(params = [batch_pert], lr = self.args.pert_lr) 
                    
                    global_model.eval()
                    for i in range(self.args.pert_rounds):
                        for images, labels in self.train_loader:
                            images = images.to(self.args.device)
                            labels = labels.to(self.args.device)
                            per_logits = global_model.forward(images + batch_pert)
                            loss_pert = F.cross_entropy(per_logits, labels, reduction='mean')
                            loss_regu = torch.pow(torch.norm(batch_pert),2)
                            loss_all = torch.mean(-loss_pert)  + self.args.regu_weight * loss_regu
                            batch_opt.zero_grad()
                            loss_all.backward(retain_graph = True)
                            batch_opt.step()
            
                    if self.args.norm_limit:
                        with torch.no_grad():
                            pert = batch_pert * min(1, self.args.limit_norm  / torch.norm(batch_pert))
                    else:
                        pert = batch_pert
                        
                    global_model.train()       
                    for images, labels in self.train_loader:
                        images = images.to(self.args.device)
                        labels = labels.to(self.args.device)
                        patching = torch.zeros_like(images, device=self.args.device)
                        image_num = images.shape[0]
                        rand_idx = random.sample(list(np.arange(image_num)), round(image_num*self.args.portion))
                        patching[rand_idx] = pert[0]
                        unlearn_imgs = images + patching
                        perb_logits = global_model.forward(unlearn_imgs)
                        
                        clean_features = global_model.get_features(images)
                        pert_features = global_model.get_features(images + pert)
                        loss_feature = mse(clean_features, pert_features)
                        
                        loss_unl = F.cross_entropy(perb_logits, labels, reduction='mean')
                        loss_tt = loss_unl + self.args.fea_w * loss_feature
                        
                        optimizer.zero_grad()
                        loss_tt.backward()
                        optimizer.step()     
        else:
            if self.args.mc_adv_train:
                global_model.train()
                optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.client_lr * (self.args.lr_decay) ** rnd,
                                    weight_decay=self.args.wd)
                if self.args.unl_mode == "tt_unl_adp":
                    start = time.time()
                    for lc_rnd in range(self.args.local_ep):
                        batch_pert = torch.zeros([1, env_config["input_channel"], env_config["input_height"], env_config["input_width"]], requires_grad=True,
                                                    device = self.args.device)
                        batch_opt = torch.optim.Adam(params = [batch_pert], lr = self.args.pert_lr) 
                        
                        global_model.eval()
                        for i in range(self.args.pert_rounds):
                            for _, images, labels in self.train_loader:
                                images = images.to(self.args.device)
                                labels = labels.to(self.args.device)
                                per_logits = global_model.forward(images + batch_pert)
                                loss_pert = F.cross_entropy(per_logits, labels, reduction='mean')
                                loss_regu = torch.pow(torch.norm(batch_pert),2)
                                loss_all = torch.mean(-loss_pert)  + self.args.regu_weight * loss_regu
                                batch_opt.zero_grad()
                                loss_all.backward(retain_graph = True)
                                batch_opt.step()
                
                        if self.args.norm_limit:
                            with torch.no_grad():
                                pert = batch_pert * min(1, self.args.mc_norm  / torch.norm(batch_pert))
                        else:
                            pert = batch_pert
                            
                        global_model.train()       
                        for idx, images, labels in self.train_loader:
                            images = images.to(self.args.device)
                            labels = labels.to(self.args.device)
                            if self.args.attack == "A3FL" and self.id < self.args.num_corrupt and rnd >= self.args.start_poison and rnd < self.args.cease_poison:
                                images, labels = attacker.poison_input(images, labels, eval=False)
                            patching = torch.zeros_like(images, device=self.args.device)
                            image_num = images.shape[0]

                            poison_sample_idx = [i for i in idx if i in self.poison_idx]
                            poison_num = len(poison_sample_idx)
                            if poison_num <= round(image_num*self.args.portion):
                                # pert_idx = poison_sample_idx
                                pert_idx = poison_sample_idx[:int(poison_num *(1 - self.args.pure_poison))]
                            else:
                                pert_idx = random.sample(poison_sample_idx, round(image_num*self.args.portion))
                            idx = list(idx)
                            patching[[idx.index(v) for v in pert_idx]] = pert[0]
                            unlearn_imgs = images + patching
                            perb_logits = global_model.forward(unlearn_imgs)
                            
                            clean_features = global_model.get_features(images)
                            pert_features = global_model.get_features(images + pert)
                            loss_feature = mse(clean_features, pert_features)
                            
                            loss_unl = F.cross_entropy(perb_logits, labels, reduction='mean')
                            loss_tt = loss_unl + self.args.fea_w * loss_feature
                            
                            optimizer.zero_grad()
                            loss_tt.backward()
                            if self.args.attack == "neurotoxin" and len(neurotoxin_mask) and self.id < self.args.num_corrupt:
                                for name, param in global_model.named_parameters():
                                    param.grad.data = neurotoxin_mask[name].to(self.args.device) * param.grad.data
                            optimizer.step()
                
            else:          
                for _ in range(self.args.local_ep):
                    start = time.time()
                    for _, (inputs, labels) in enumerate(self.train_loader):
                        optimizer.zero_grad()
                        inputs, labels = inputs.to(device=self.args.device, non_blocking=True), \
                                        labels.to(device=self.args.device, non_blocking=True)
                        if self.args.attack == "A3FL" and self.id < self.args.num_corrupt and rnd >= self.args.start_poison and rnd < self.args.cease_poison:
                            inputs, labels = attacker.poison_input(inputs, labels, eval=False)
                        outputs = global_model(inputs)
                        minibatch_loss = criterion(outputs, labels)
                        minibatch_loss.backward()
                        if self.args.attack == "neurotoxin" and len(neurotoxin_mask) and self.id < self.args.num_corrupt:
                            for name, param in global_model.named_parameters():
                                param.grad.data = neurotoxin_mask[name].to(self.args.device) * param.grad.data
                        if self.args.attack == "r_neurotoxin" and len(neurotoxin_mask) and self.id < self.args.num_corrupt:
                            for name, param in global_model.named_parameters():
                                param.grad.data = (torch.ones_like(neurotoxin_mask[name].to(self.args.device))-neurotoxin_mask[name].to(self.args.device) ) * param.grad.data
                        optimizer.step()
                




        with torch.no_grad():
            after_train = parameters_to_vector(
                [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
            self.update = after_train - initial_global_model_params
            return self.update
        
        
        

class DatasetSplit(Dataset):
    """ An abstract Dataset class wrapped around Pytorch Dataset class """

    def __init__(self, dataset, idxs, runtime_poison=False, args=None, client_id=-1, modify_label=True):
        self.dataset = dataset
        self.idxs = idxs
        self.targets = torch.Tensor([self.dataset.targets[idx] for idx in idxs])
        self.runtime_poison = runtime_poison
        self.args = args
        self.client_id = client_id
        self.modify_label = modify_label
        if client_id == -1:
            poison_frac = 1
        elif client_id < self.args.num_corrupt:
            poison_frac = self.args.poison_frac
        else:
            poison_frac = 0
        self.poison_sample = {}
        self.poison_idxs = []
        if runtime_poison and poison_frac > 0:
            self.poison_idxs = random.sample(self.idxs, floor(poison_frac * len(self.idxs)))
            for idx in self.poison_idxs:
                self.poison_sample[idx] = utils.add_pattern_bd(copy.deepcopy(self.dataset[idx][0]), None, dataset=self.args.data,
                                                         pattern_type='plus', agent_idx=client_id,
                                                         attack='badnet')
                # plt.imshow(self.poison_sample[idx].permute(1, 2, 0))
                # plt.show()

    def classes(self):
        return torch.unique(self.targets)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # print(target.type())
        if self.idxs[item] in self.poison_idxs:
            inp = self.poison_sample[self.idxs[item]]
            if self.modify_label:
                target = self.args.target_class
            else:
                target = self.dataset[self.idxs[item]][1]
        else:
            inp, target = self.dataset[self.idxs[item]]

        return self.idxs[item], inp, target