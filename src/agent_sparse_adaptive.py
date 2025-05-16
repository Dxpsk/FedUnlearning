import copy
import math
import torch
import utils
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import logging

class Agent():
    def __init__(self, id, args, train_dataset=None, data_idxs=None, mask=None):
        self.id = id
        self.args = args
        self.error = 0
        # self.val_loader = val_loader
        # self.poisoned_val_loader = poisoned_val_loader
        self.data_idxs = data_idxs
        # poisoned datasets, tinyimagenet is handled differently as the dataset is not loaded into memory
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

        self.mask = copy.deepcopy(mask)
        self.num_remove= None
        
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



    def screen_gradients(self, model):
        model.train()
        # # # train and update
        criterion = nn.CrossEntropyLoss()
        gradient = {name: 0 for name, param in model.named_parameters()}
        # # sample 10 batch  of data
        batch_num = 0
        
        if self.id < self.args.num_corrupt:
            train_dataset = self.clean_backup_dataset
            train_loader = DataLoader(train_dataset, batch_size=self.args.bs, shuffle=True,num_workers=self.args.num_workers, pin_memory=False, drop_last=True)
        else: 
            train_loader = self.train_loader
        
        for _, (x, labels) in enumerate(train_loader):
            batch_num+=1
            model.zero_grad()
            x, labels = x.to(self.args.device), labels.to(self.args.device)
            log_probs = model.forward(x)
            minibatch_loss = criterion(log_probs, labels.long())
            loss = minibatch_loss
            loss.backward()
            for name, param in model.named_parameters():
                gradient[name] += param.grad.data
        return gradient

    def update_mask(self, masks, num_remove, gradient=None):
        for name in gradient:
            if self.args.dis_check_gradient:
                temp = torch.where(masks[name] == 0, torch.ones_like(masks[name]), torch.zeros_like(masks[name]))
                idx = torch.multinomial(temp.flatten().to(self.args.device), num_remove[name], replacement=False)
                masks[name].view(-1)[idx] = 1
            else:
                temp = torch.where(masks[name].to(self.args.device) == 0, torch.abs(gradient[name]),
                                    -100000 * torch.ones_like(gradient[name]))
                sort_temp, idx = torch.sort(temp.view(-1), descending=True)
                masks[name].view(-1)[idx[:num_remove[name]]] = 1
        return masks
    
    # def init_mask(self,  gradient=None):
    #     for name in self.mask:
    #         num_init = torch.count_nonzero(self.mask[name])
    #         self.mask[name] = torch.zeros_like(self.mask[name])
    #         sort_temp, idx = torch.sort(torch.abs(gradient[name]).view(-1), descending=True)
    #         self.mask[name].view(-1)[idx[:num_init]] = 1
             
    # def local_test(self, model, rnd, attacker=None):
    #     env_config = utils.get_env(self.args)
    #     criterion = nn.CrossEntropyLoss()
    #     with torch.no_grad():
    #         val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(model, criterion, self.val_loader,
    #                                                                                 self.args, rnd, env_config["num_classes"])
    #         logging.info(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
    #         if self.args.attack == "A3FL" and rnd >= self.args.start_poison:
    #             poison_loss, (asr, _), fail_samples = utils.get_loss_n_accuracy(model, criterion,
    #                                                                         self.poisoned_val_loader, self.args, rnd, env_config["num_classes"], attacker)
    #         else:
    #             poison_loss, (asr, _), fail_samples = utils.get_loss_n_accuracy(model, criterion,
    #                                                                         self.poisoned_val_loader, self.args, rnd, env_config["num_classes"])
    #         logging.info(f'| Attack Loss/Attack Success Ratio: {poison_loss:.3f} / {asr:.3f} |')
    #     return val_loss, val_acc, poison_loss, asr

    def fire_mask(self, weights, masks, round):
        
        drop_ratio = self.args.anneal_factor / 2 * (1 + np.cos((round * np.pi) / (self.args.rounds)))
    
        # logging.info(drop_ratio)
        num_remove = {}
        for name in masks:
                num_non_zeros = torch.sum(masks[name].to(self.args.device))
                num_remove[name] = math.ceil(drop_ratio * num_non_zeros)
     
        for name in masks:
            if num_remove[name]>0 and  "track" not in name and "running" not in name: 
                if  self.id < self.args.num_corrupt:
                    temp_weights = torch.where(masks[name].to(self.args.device) > 0, torch.rand_like(weights[name]),
                                            100000 * torch.ones_like(weights[name]))
                else:
                    temp_weights = torch.where(masks[name].to(self.args.device) > 0, torch.abs(weights[name]),
                                            100000 * torch.ones_like(weights[name]))
                x, idx = torch.sort(temp_weights.view(-1).to(self.args.device))
                masks[name].view(-1)[idx[:num_remove[name]]] = 0
        return masks, num_remove



    def local_train(self, global_model, criterion, rnd=None, temparature=10, alpha=0.3, global_mask= None, neurotoxin_mask =None, updates_dict = None, attacker = None):
        """ Do a local training over the received global model, return the update """
        initial_global_model_params = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        if self.args.attack == "A3FL" and self.id < self.args.num_corrupt and rnd >= self.args.start_poison and rnd < self.args.cease_poison:
            if attacker.update_flag:
                logging.info(f"Agent {self.id} is searching for trigger")
                search_model = copy.deepcopy(global_model)
                for name, param in search_model.named_parameters():
                    mask_layer = self.mask[name].to(self.args.device)
                    param.data = param.data * mask_layer
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
        global_model.to(self.args.device)
        for name, param in global_model.named_parameters():
            self.mask[name] =self.mask[name].to(self.args.device)
            param.data = param.data * self.mask[name]
        if self.num_remove!=None:
            if self.id>=  self.args.num_corrupt or self.args.attack!="fix_mask" :
                gradient = self.screen_gradients(global_model)
                self.mask = self.update_mask(self.mask, self.num_remove, gradient)
        
        global_model.train()
        lr = self.args.client_lr* (self.args.lr_decay)**rnd
        optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, weight_decay=self.args.wd)

        for _ in range(self.args.local_ep):
            for _, (inputs, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device=self.args.device), \
                                 labels.to(device=self.args.device)
                if self.args.attack == "A3FL" and self.id < self.args.num_corrupt and rnd >= self.args.start_poison and rnd < self.args.cease_poison:
                    inputs, labels = attacker.poison_input(inputs, labels, eval=False)
                outputs = global_model(inputs)
                minibatch_loss = criterion(outputs, labels)
                loss = minibatch_loss
                loss.backward()
                for name, param in global_model.named_parameters():
                    param.grad.data = self.mask[name].to(self.args.device) * param.grad.data
                if self.args.attack == "neurotoxin" and self.id < self.args.num_corrupt and rnd >= self.args.start_poison and rnd < self.args.cease_poison:
                    if len(neurotoxin_mask):
                        for name, param in global_model.named_parameters():
                            param.grad.data = neurotoxin_mask[name].to(self.args.device) * param.grad.data
                optimizer.step()

        # global_model.eval()
        # val_loss, val_acc, poison_loss, asr = self.local_test(global_model, rnd)  
        

        if self.id <  self.args.num_corrupt:
            if self.args.attack=="fix_mask":
                self.mask = self.mask 

            elif self.args.attack == "omniscient":
                if len(global_mask):
                    self.mask = copy.deepcopy(global_mask)
                else:
                    self.mask = self.mask 
            # elif self.args.attack == "neurotoxin":
            #     if len(neurotoxin_mask):
            #         self.mask, self.num_remove = self.fire_mask(global_model.state_dict(), self.mask, rnd)
            #         for name in self.mask:
            #                 self.mask[name] = self.mask[name].to(self.args.device) * neurotoxin_mask[name].to(self.args.device)
            #     else:
            #         self.mask, self.num_remove = self.fire_mask(global_model.state_dict(), self.mask, rnd)
            else:
                self.mask, self.num_remove = self.fire_mask(global_model.state_dict(), self.mask, rnd)

        else:
            self.mask, self.num_remove = self.fire_mask(global_model.state_dict(), self.mask, rnd) 
            
        with torch.no_grad():
            after_train = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
            array_mask = parameters_to_vector([ self.mask[name].to(self.args.device) for name in global_model.state_dict()]).detach()
            self.update = ( array_mask *(after_train - initial_global_model_params))
            if "scale" in self.args.attack:
                logging.info("scale update for" + self.args.attack.split("_",1)[1] + " times")
                if self.id<  self.args.num_corrupt:
                    self.update=  int(self.args.attack.split("_",1)[1]) * self.update
        return self.update

