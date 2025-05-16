import copy
import math
import time

import torch
import models
import utils
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
import torch.nn as nn
import logging
import flip
import torch.nn.functional as F

class Agent():
    def __init__(self, id, args, train_dataset=None, data_idxs=None, mask=None):
        self.id = id
        self.args = args
        self.error = 0
        # get datasets, fedemnist is handled differently as it doesn't come with pytorch
        if train_dataset is None:
            self.train_dataset = torch.load(f'../data/Fed_EMNIST/user_trainsets/user_{id}_trainset.pt')

            # for backdoor attack, agent poisons his local dataset
            if self.id < args.num_corrupt:
                utils.poison_dataset(self.train_dataset, args, data_idxs, agent_idx=self.id)

        else:
            if self.args.data != "tinyimagenet":

                self.train_dataset = utils.DatasetSplit(train_dataset, data_idxs)
                # for backdoor attack, agent poisons his local dataset
                if self.id < args.num_corrupt:
                    self.clean_backup_dataset = copy.deepcopy(self.train_dataset)
                    self.data_idxs = data_idxs
                    utils.poison_dataset(train_dataset, args, data_idxs, agent_idx=self.id)
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

    def local_train(self, global_model, criterion, rnd=None, neurotoxin_mask=None, attacker = None):
        """ Do a local training over the received global model, return the update """
        initial_global_model_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
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
        end = time.time()
        # logging.info(end - start)

        fish_dict = self.fish_calculate(copy.deepcopy(global_model))
        
        with torch.no_grad():
            after_train = parameters_to_vector(
                [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
            self.update = after_train - initial_global_model_params
        return self.update, fish_dict

    def fish_calculate(self, net):
        loader = self.train_loader
        average_grads = {name: torch.zeros_like(param) for name, param in
                         net.named_parameters()}
        num_batches = 0
        for step, (image, label) in enumerate(loader):
            num_batches += 1
            image, label = image.to(self.args.device), label.to(self.args.device)
            output = net(image)
            pre = F.log_softmax(output, dim=1)
            log_liklihoods = (pre[:, label])
            net.zero_grad()
            log_liklihoods.mean().backward(retain_graph=True)
            for name, param in net.named_parameters():
                if param.grad is not None:
                    average_grads[name] += param.grad

            net.zero_grad()

        for name in average_grads:
            average_grads[name] /= num_batches
        fish_dict = {}
        for name, grad in average_grads.items():
            fish_dict[name] = average_grads[name].clone() ** 2

        return fish_dict