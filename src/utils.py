import copy
import logging
import os

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from math import floor
from collections import defaultdict
import random
import math
from sklearn.cluster import KMeans
from pytorch_cinic.dataset import CINIC10
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from typing import Any, Callable, Optional, Tuple, Union
import csv
import pathlib
from torchvision.datasets.vision import VisionDataset
from PIL import Image

class H5Dataset(Dataset):
    def __init__(self, dataset, client_id):
        self.targets = torch.LongTensor(dataset[client_id]['label'])
        self.inputs = torch.Tensor(dataset[client_id]['pixels'])
        shape = self.inputs.shape
        self.inputs = self.inputs.view(shape[0], 1, shape[1], shape[2])

    def classes(self):
        return torch.unique(self.targets)

    def __add__(self, other):
        self.targets = torch.cat((self.targets, other.targets), 0)
        self.inputs = torch.cat((self.inputs, other.inputs), 0)
        return self

    def to(self, device):
        self.targets = self.targets.to(device)
        self.inputs = self.inputs.to(device)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, item):
        inp, target = self.inputs[item], self.targets[item]
        return inp, target


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
                self.poison_sample[idx] = add_pattern_bd(copy.deepcopy(self.dataset[idx][0]), None, args.data,
                                                         pattern_type=args.pattern_type, agent_idx=client_id,
                                                         attack=args.attack)
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
                target = torch.LongTensor(self.args.target_class)
            else:
                target = torch.LongTensor(self.dataset[self.idxs[item]][1])
        else:
            inp, target = self.dataset[self.idxs[item]]

        return inp, target


def distribute_data_dirichlet(dataset, args):
    # sort labels
    labels_sorted = dataset.targets.sort()
    # create a list of pairs (index, label), i.e., at index we have an instance of  label
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    labels_dict = defaultdict(list)

    for k, v in class_by_labels:
        labels_dict[k].append(v)
    # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
    N = len(labels_sorted[1])
    K = len(labels_dict)
    logging.info((N, K))
    client_num = args.num_agents

    min_size = 0
    while min_size < 10:
        idx_batch = [[] for _ in range(client_num)]
        for k in labels_dict:
            idx_k = labels_dict[k]

            # get a list of batch indexes which are belong to label k
            np.random.shuffle(idx_k)
            # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
            # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
            proportions = np.random.dirichlet(np.repeat(args.alpha, client_num))

            # get the index in idx_k according to the dirichlet distribution
            proportions = np.array([p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            # generate the batch list for each client
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    # distribute data to users
    dict_users = defaultdict(list)
    for user_idx in range(args.num_agents):
        dict_users[user_idx] = idx_batch[user_idx]
        np.random.shuffle(dict_users[user_idx])

    dict_users_per_classes = defaultdict(list)
    for user_idx in range(args.num_agents):
        dict_users_per_classes[user_idx] = [0 for i in range(len(list(labels_dict.keys())))]
        for i in dict_users[user_idx]:
            dict_users_per_classes[user_idx][dataset.targets[i]] += 1
    # num = [ [ 0 for k in range(K) ] for i in range(client_num)]
    # for k in range(K):
    #     for i in dict_users:
    #         num[i][k] = len(np.intersect1d(dict_users[i], labels_dict[k]))
    # logging.info(num)
    # print(dict_users)
    # def intersection(lst1, lst2):
    #     lst3 = [value for value in lst1 if value in lst2]
    #     return lst3
    # # logging.info( [len(intersection (dict_users[i], dict_users[i+1] )) for i in range(args.num_agents)] )
    return dict_users, dict_users_per_classes


def distribute_data(dataset, args, n_classes=10):
    # logging.info(dataset.targets)
    # logging.info(dataset.classes)
    class_per_agent = n_classes

    if args.num_agents == 1:
        return {0: range(len(dataset))}

    def chunker_list(seq, size):
        return [seq[i::size] for i in range(size)]

    # sort labels
    labels_sorted = torch.tensor(dataset.targets).sort()
    # print(labels_sorted)
    # create a list of pairs (index, label), i.e., at index we have an instance of  label
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
    labels_dict = defaultdict(list)
    for k, v in class_by_labels:
        labels_dict[k].append(v)

    # split indexes to shards
    shard_size = len(dataset) // (args.num_agents * class_per_agent)
    slice_size = (len(dataset) // n_classes) // shard_size
    for k, v in labels_dict.items():
        labels_dict[k] = chunker_list(v, slice_size)
    hey = copy.deepcopy(labels_dict)
    # distribute shards to users
    dict_users = defaultdict(list)
    for user_idx in range(args.num_agents):
        class_ctr = 0
        for j in range(0, n_classes):
            if class_ctr == class_per_agent:
                break
            elif len(labels_dict[j]) > 0:
                dict_users[user_idx] += labels_dict[j][0]
                del labels_dict[j % n_classes][0]
                class_ctr += 1
        np.random.shuffle(dict_users[user_idx])
    # num = [ [ 0 for k in range(n_classes) ] for i in range(args.num_agents)]
    # for k in range(n_classes):
    #     for i in dict_users:
    #         num[i][k] = len(np.intersect1d(dict_users[i], hey[k]))
    # logging.info(num)
    # logging.info(args.num_agents)
    # def intersection(lst1, lst2):
    #     lst3 = [value for value in lst1 if value in lst2]
    #     return lst3
    # logging.info( len(intersection (dict_users[0], dict_users[1] )))

    return dict_users


def get_datasets(data):
    """ returns train and test datasets """
    train_dataset, test_dataset = None, None
    data_dir = '../data'

    if data == 'fmnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    if data == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    elif data == 'fedemnist':
        train_dir = '../data/Fed_EMNIST/fed_emnist_all_trainset.pt'
        test_dir = '../data/Fed_EMNIST/fed_emnist_all_valset.pt'
        train_dataset = torch.load(train_dir)
        test_dataset = torch.load(test_dir)

    elif data == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(
            test_dataset.targets)
    elif data == 'cifar100':
        transform = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                             std=[0.2675, 0.2565, 0.2761])])
        valid_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                                   std=[0.2675, 0.2565, 0.2761])])
        train_dataset = datasets.CIFAR100(data_dir,
                                          train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(data_dir,
                                         train=False, download=True, transform=valid_transform)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(
            test_dataset.targets)
    # elif data == "GTSRB":
    #     transform = transforms.Compose([
    #                                     transforms.Resize((32, 32)),
    #                                     transforms.RandomHorizontalFlip(),
    #                                     transforms.ToTensor(),
    #                                 ])
    #     valid_transform = transforms.Compose([
    #                                         transforms.Resize((32, 32)),
    #                                         transforms.ToTensor(),
    #                                     ])
    #     train_dataset = GTSRB_Modifiable(data_dir,
    #                                       split="train", download=True, transform=transform)
    #     test_dataset = GTSRB_Modifiable(data_dir,
    #                                      split="test", download=True, transform=valid_transform)
    #     train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(
    #         test_dataset.targets)
    # elif data == "cinic":
        
    elif data == "tinyimagenet":
        _data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor()
            ]),
            'val': transforms.Compose([
                transforms.ToTensor()
            ]),
        }
        _data_dir = '../data/tiny-imagenet-200/'
        train_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'train'),
                                             _data_transforms['train'])
        # print(train_dataset[0][0].shape)
        test_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'val'),
                                            _data_transforms['val'])
        train_dataset.targets = torch.tensor(train_dataset.targets)
        test_dataset.targets = torch.tensor(test_dataset.targets)
    
    return train_dataset, test_dataset


def get_loss_n_accuracy(model, criterion, data_loader, args, round, num_classes=10, attacker = None):
    """ Returns the loss and total accuracy, per class accuracy on the supplied data loader """

    # disable BN stats during inference
    model.eval()
    total_loss, correctly_labeled_samples = 0, 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
    not_correct_samples = []
    # forward-pass to get loss and predictions of the current batch
    all_labels = []
    # if round % 20 == 0:
    #     def hook(module, fea_n, fea_out):
    #         representation.append(fea_out.detach().cpu())
    #         return None
    #     for (name, module) in model.named_modules():
    #         if name == layer_name:
    #             handle = module.register_forward_hook(hook=hook)

    for _, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device=args.device, non_blocking=True), \
                         labels.to(device=args.device, non_blocking=True)
        # compute the total loss over minibatch
        if args.attack == "A3FL" and round >= args.start_poison and attacker != None:
            inputs, labels = attacker.poison_input(inputs, labels, eval=True)
        outputs = model(inputs)
        avg_minibatch_loss = criterion(outputs, labels)
        total_loss += avg_minibatch_loss.item() * outputs.shape[0]

        # get num of correctly predicted inputs in the current batch
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        all_labels.append(labels.cpu().view(-1))
        # correct_inputs = labels[torch.nonzero(torch.eq(pred_labels, labels) == 0).squeeze()]
        # not_correct_samples.append(  wrong_inputs )
        correctly_labeled_samples += torch.sum(torch.eq(pred_labels, labels)).item()
        # fill confusion_matrix
        for t, p in zip(labels.view(-1), pred_labels.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correctly_labeled_samples / len(data_loader.dataset)
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    return avg_loss, (accuracy, per_class_accuracy), not_correct_samples


def get_loss_n_accuracy_with_prob(model, criterion, data_loader, device,  num_classes=10, prob=0.4):
    """ Returns the loss and total accuracy, per class accuracy on the supplied data loader """
    # disable BN stats during inference
    model.eval()
    dataset_size = 0
    correct = 0
    was_correct = 0
    with torch.no_grad():
        for _, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device, non_blocking=True), \
                            labels.to(device, non_blocking=True)
            dataset_size += len(inputs)
            outputs = model(inputs)

            pred = outputs.data.max(1)[1]  # get the index of the max log-probability

            probabilities = torch.nn.functional.softmax(outputs)
            probabilities = torch.gather(probabilities, 1, pred.unsqueeze(1))
            
            for idx, is_correct in enumerate(pred.eq(labels.data.view_as(pred)).cpu()):
                if is_correct == True:
                    was_correct += 1
                    if probabilities[idx][0] >= prob:
                        correct += 1
                        
    acc = (float(correct) / float(dataset_size)) if dataset_size != 0 else 0
    return acc

def poison_dataset(dataset, args, data_idxs=None, poison_all=False, agent_idx=-1, modify_label=True):
    # if data_idxs != None:
    #     all_idxs = list(set(all_idxs).intersection(data_idxs))
    if data_idxs != None:
        all_idxs = (dataset.targets != args.target_class).nonzero().flatten().tolist()
        all_idxs = list(set(all_idxs).intersection(data_idxs))
    else:
        all_idxs = (dataset.targets != args.target_class).nonzero().flatten().tolist()
    poison_frac = 1 if poison_all else args.poison_frac
    poison_idxs = random.sample(all_idxs, floor(poison_frac * len(all_idxs)))
    for idx in poison_idxs:
        if args.data == 'fedemnist':
            clean_img = dataset.inputs[idx]
        elif args.data == "tinyimagenet":
            clean_img = dataset[idx][0]
        else:
            clean_img = dataset.data[idx]
        bd_img = add_pattern_bd(clean_img, dataset.targets[idx], args.data, pattern_type=args.pattern_type,
                                agent_idx=agent_idx, attack=args.attack)
        if args.data == 'fedemnist':
            dataset.inputs[idx] = torch.tensor(bd_img)
        elif args.data == "tinyimagenet" or args.data == "GTSRB":
            # don't do anything for tinyimagenet, we poison it in run time
            return
        else:
            dataset.data[idx] = torch.tensor(bd_img)
        if modify_label:
            dataset.targets[idx] = args.target_class
    return poison_idxs


def init_masks(params, sparsities):
    masks = {}
    for name in params:
        masks[name] = torch.zeros_like(params[name])
        dense_numel = int((1 - sparsities[name]) * torch.numel(masks[name]))
        if dense_numel > 0:
            temp = masks[name].view(-1)
            perm = torch.randperm(len(temp))
            perm = perm[:dense_numel]
            temp[perm] = 1
        masks[name] = masks[name].to("cpu")
    return masks


def vector_to_model(vec, model):
    # Pointer for slicing the vector for each parameter
    state_dict = model.state_dict()
    pointer = 0
    for name in state_dict:
        # The length of the parameter
        num_param = state_dict[name].numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        state_dict[name].data = vec[pointer:pointer + num_param].view_as(state_dict[name]).data
        # Increment the pointer
        pointer += num_param
    model.load_state_dict(state_dict)
    return state_dict


def calculate_sparsities(args, params, tabu=[], distribution="ERK"):
    spasities = {}
    if distribution == "uniform":
        for name in params:
            if name not in tabu:
                spasities[name] = 1 - args.dense_ratio
            else:
                spasities[name] = 0
    elif distribution == "ERK":
        logging.info('initialize by ERK')
        total_params = 0
        for name in params:
            total_params += params[name].numel()
        is_epsilon_valid = False
        # # The following loop will terminate worst case when all masks are in the
        # custom_sparsity_map. This should probably never happen though, since once
        # we have a single variable or more with the same constant, we have a valid
        # epsilon. Note that for each iteration we add at least one variable to the
        # custom_sparsity_map and therefore this while loop should terminate.
        dense_layers = set()

        density = args.dense_ratio
        while not is_epsilon_valid:
            # We will start with all layers and try to find right epsilon. However if
            # any probablity exceeds 1, we will make that layer dense and repeat the
            # process (finding epsilon) with the non-dense layers.
            # We want the total number of connections to be the same. Let say we have
            # for layers with N_1, ..., N_4 parameters each. Let say after some
            # iterations probability of some dense layers (3, 4) exceeded 1 and
            # therefore we added them to the dense_layers set. Those layers will not
            # scale with erdos_renyi, however we need to count them so that target
            # paratemeter count is achieved. See below.
            # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
            #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
            # eps * (p_1 * N_1 + p_2 * N_2) =
            #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
            # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name in params:
                if name in tabu or "running" in name or "track" in name :
                    dense_layers.add(name)
                n_param = np.prod(params[name].shape)
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density

                if name in dense_layers:
                    rhs -= n_zeros
                else:
                    rhs += n_ones
                    raw_probabilities[name] = (
                                                      np.sum(params[name].shape) / np.prod(params[name].shape)
                                              ) ** 1
                    divisor += raw_probabilities[name] * n_param
            epsilon = rhs / divisor
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        print(f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True

        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name in params:
            if name in dense_layers:
                spasities[name] = 0
            else:
                spasities[name] = (1 - epsilon * raw_probabilities[name])
    return spasities


def name_param_to_array(param):
    vec = []
    for name in param:
        # Ensure the parameters are located in the same device
        vec.append(param[name].view(-1))
    return torch.cat(vec)


def vector_to_name_param(vec, name_param_map):
    pointer = 0
    for name in name_param_map:
        # The length of the parameter
        num_param = name_param_map[name].numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        name_param_map[name].data = vec[pointer:pointer + num_param].view_as(name_param_map[name]).data
        # Increment the pointer
        pointer += num_param

    return name_param_map


def add_pattern_bd(x, y, dataset='cifar10', pattern_type='square', agent_idx=-1, attack="DBA"):
    """
    adds a trojan pattern to the image
    """

    # if cifar is selected, we're doing a distributed backdoor attack (i.e., portions of trojan pattern is split between agents, only works for plus)
    if dataset == 'cifar10' or dataset == "cifar100":
        x = np.array(x.squeeze())
        # logging.info(x.shape)
        row = x.shape[0]
        column = x.shape[1]

        if attack == "periodic_trigger":
            for d in range(0, 3):
                for i in range(row):
                    for j in range(column):
                        x[i][j][d] = max(min(x[i][j][d] + 20 * math.sin((2 * math.pi * j * 6) / column), 255), 0)
            # import matplotlib.pyplot as plt
            # plt.imsave("visualization/input_images/backdoor2.png", x)
            # print(y)
            # plt.show()
        else:
            if pattern_type == 'plus':
                start_idx = 5
                size = 6
                if agent_idx == -1:
                    # vertical line
                    for d in range(0, 3):
                        for i in range(start_idx, start_idx + size + 1):
                            if d == 2:
                                x[i, start_idx][d] = 0
                            else:
                                x[i, start_idx][d] = 255
                    # horizontal line
                    for d in range(0, 3):
                        for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                            if d == 2:
                                x[start_idx + size // 2, i][d] = 0
                            else:
                                x[start_idx + size // 2, i][d] = 255
                else:
                    if attack == "DBA":
                        # DBA attack
                        # upper part of vertical
                        if agent_idx % 4 == 0:
                            for d in range(0, 3):
                                for i in range(start_idx, start_idx + (size // 2) + 1):
                                    if d == 2:
                                        x[i, start_idx][d] = 0
                                    else:
                                        x[i, start_idx][d] = 255

                        # lower part of vertical
                        elif agent_idx % 4 == 1:
                            for d in range(0, 3):
                                for i in range(start_idx + (size // 2) + 1, start_idx + size + 1):
                                    if d == 2:
                                        x[i, start_idx][d] = 0
                                    else:
                                        x[i, start_idx][d] = 255

                        # left-part of horizontal
                        elif agent_idx % 4 == 2:
                            for d in range(0, 3):
                                for i in range(start_idx - size // 2, start_idx - size // 4 + 1):
                                    if d == 2:
                                        x[start_idx + size // 2, i][d] = 0
                                    else:
                                        x[start_idx + size // 2, i][d] = 255
                        # right-part of horizontal
                        elif agent_idx % 4 == 3:
                            for d in range(0, 3):
                                for i in range(start_idx - size // 4 + 1, start_idx + size // 2 + 1):
                                    if d == 2:
                                        x[start_idx + size // 2, i][d] = 0
                                    else:
                                        x[start_idx + size // 2, i][d] = 255
                    else:
                        # vertical line
                        for d in range(0, 3):
                            for i in range(start_idx, start_idx + size + 1):
                                if d == 2:
                                    x[i, start_idx][d] = 0
                                else:
                                    x[i, start_idx][d] = 255
                        # horizontal line
                        for d in range(0, 3):
                            for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                                if d == 2:
                                    x[start_idx + size // 2, i][d] = 0
                                else:
                                    x[start_idx + size // 2, i][d] = 255
            elif pattern_type == "plus_wh":
                start_idx = 5
                size = 6
                if agent_idx == -1:
                    # vertical line
                    for d in range(0, 3):
                        for i in range(start_idx, start_idx + size + 1):
                            if d == 2:
                                x[i, start_idx][d] = 255
                            else:
                                x[i, start_idx][d] = 255
                    # horizontal line
                    for d in range(0, 3):
                        for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                            if d == 2:
                                x[start_idx + size // 2, i][d] = 255
                            else:
                                x[start_idx + size // 2, i][d] = 255
                else:
                    if attack == "DBA":
                        # DBA attack
                        # upper part of vertical
                        if agent_idx % 4 == 0:
                            for d in range(0, 3):
                                for i in range(start_idx, start_idx + (size // 2) + 1):
                                    if d == 2:
                                        x[i, start_idx][d] = 255
                                    else:
                                        x[i, start_idx][d] = 255

                        # lower part of vertical
                        elif agent_idx % 4 == 1:
                            for d in range(0, 3):
                                for i in range(start_idx + (size // 2) + 1, start_idx + size + 1):
                                    if d == 2:
                                        x[i, start_idx][d] = 255
                                    else:
                                        x[i, start_idx][d] = 255

                        # left-part of horizontal
                        elif agent_idx % 4 == 2:
                            for d in range(0, 3):
                                for i in range(start_idx - size // 2, start_idx - size // 4 + 1):
                                    if d == 2:
                                        x[start_idx + size // 2, i][d] = 255
                                    else:
                                        x[start_idx + size // 2, i][d] = 255
                        # right-part of horizontal
                        elif agent_idx % 4 == 3:
                            for d in range(0, 3):
                                for i in range(start_idx - size // 4 + 1, start_idx + size // 2 + 1):
                                    if d == 2:
                                        x[start_idx + size // 2, i][d] = 255
                                    else:
                                        x[start_idx + size // 2, i][d] = 255
                    else:
                        # vertical line
                        for d in range(0, 3):
                            for i in range(start_idx, start_idx + size + 1):
                                if d == 2:
                                    x[i, start_idx][d] = 255
                                else:
                                    x[i, start_idx][d] = 255
                        # horizontal line
                        for d in range(0, 3):
                            for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                                if d == 2:
                                    x[start_idx + size // 2, i][d] = 255
                                else:
                                    x[start_idx + size // 2, i][d] = 255
            elif pattern_type == 'square':
                if agent_idx == -1:
                    for d in range(3):
                        if d == 2:
                            x[0, 0:6, d] = 255
                            x[0, 9:15, d] = 255
                            x[4, 0:6, d] = 255
                            x[4, 9:15, d] = 255
                        else:
                            x[0, 0:6, d] = 255
                            x[0, 9:15, d] = 255
                            x[4, 0:6, d] = 255
                            x[4, 9:15, d] = 255
                else:
                    if attack == "DBA":
                        if agent_idx % 4 == 0:
                            for d in range(3):
                                if d == 2:
                                    x[0, 0:6, d] = 255
                                else:
                                    x[0, 0:6, d] = 255
                        elif agent_idx % 4 == 1:
                            for d in range(3):
                                if d == 2:
                                    x[0, 9:15, d] = 255
                                else:
                                    x[0, 9:15, d] = 255
                        elif agent_idx % 4 == 2:
                            for d in range(3):
                                if d == 2:
                                    x[4, 0:6, d] = 255
                                else:
                                    x[4, 0:6, d] = 255
                        elif agent_idx % 4 == 3:
                            for d in range(3):
                                if d == 2:
                                    x[4, 9:15, d] = 255
                                else:
                                    x[4, 9:15, d] = 255
                    else:
                        for d in range(3):
                            if d == 2:
                                x[0, 0:6, d] = 255
                                x[0, 9:15, d] = 255
                                x[4, 0:6, d] = 255
                                x[4, 9:15, d] = 255
                            else:
                                x[0, 0:6, d] = 255
                                x[0, 9:15, d] = 255
                                x[4, 0:6, d] = 255
                                x[4, 9:15, d] = 255
            elif pattern_type == 'square_ye':
                if agent_idx == -1:
                    for d in range(3):
                        if d == 2:
                            x[0, 0:6, d] = 0
                            x[0, 9:15, d] = 0
                            x[4, 0:6, d] = 0
                            x[4, 9:15, d] = 0
                        else:
                            x[0, 0:6, d] = 255
                            x[0, 9:15, d] = 255
                            x[4, 0:6, d] = 255
                            x[4, 9:15, d] = 255
                else:
                    if attack == "DBA":
                        if agent_idx % 4 == 0:
                            for d in range(3):
                                if d == 2:
                                    x[0, 0:6, d] = 0
                                else:
                                    x[0, 0:6, d] = 255
                        elif agent_idx % 4 == 1:
                            for d in range(3):
                                if d == 2:
                                    x[0, 9:15, d] = 0
                                else:
                                    x[0, 9:15, d] = 255
                        elif agent_idx % 4 == 2:
                            for d in range(3):
                                if d == 2:
                                    x[4, 0:6, d] = 0
                                else:
                                    x[4, 0:6, d] = 255
                        elif agent_idx % 4 == 3:
                            for d in range(3):
                                if d == 2:
                                    x[4, 9:15, d] = 0
                                else:
                                    x[4, 9:15, d] = 255
                    else:
                        for d in range(3):
                            if d == 2:
                                x[0, 0:6, d] = 0
                                x[0, 9:15, d] = 0
                                x[4, 0:6, d] = 0
                                x[4, 9:15, d] = 0
                            else:
                                x[0, 0:6, d] = 255
                                x[0, 9:15, d] = 255
                                x[4, 0:6, d] = 255
                                x[4, 9:15, d] = 255
            elif pattern_type == 'square_mi':
                if agent_idx == -1:
                    for d in range(3):
                        if d == 2:
                            x[1, 2:6, d] = 255
                            x[1, 8:12, d] = 255
                            x[3, 2:6, d] = 255
                            x[3, 8:12, d] = 255
                        else:
                            x[1, 2:6, d] = 255
                            x[1, 8:12, d] = 255
                            x[3, 2:6, d] = 255
                            x[3, 8:12, d] = 255
                else:
                    if attack == "DBA":
                        if agent_idx % 4 == 0:
                            for d in range(3):
                                if d == 2:
                                    x[1, 2:6, d] = 255
                                else:
                                    x[1, 2:6, d] = 255
                        elif agent_idx % 4 == 1:
                            for d in range(3):
                                if d == 2:
                                    x[1, 8:12, d] = 255
                                else:
                                    x[1, 8:12, d] = 255
                        elif agent_idx % 4 == 2:
                            for d in range(3):
                                if d == 2:
                                    x[3, 2:6, d] = 255
                                else:
                                    x[3, 2:6, d] = 255
                        elif agent_idx % 4 == 3:
                            for d in range(3):
                                if d == 2:
                                    x[3, 8:12, d] = 255
                                else:
                                    x[3, 8:12, d] = 255
                    else:
                        for d in range(3):
                            if d == 2:
                                x[1, 2:6, d] = 255
                                x[1, 8:12, d] = 255
                                x[3, 2:6, d] = 255
                                x[3, 8:12, d] = 255
                            else:
                                x[1, 2:6, d] = 255
                                x[1, 8:12, d] = 255
                                x[3, 2:6, d] = 255
                                x[3, 8:12, d] = 255
                # import matplotlib.pyplot as plt
                
                # plt.imsave("visualization/input_images/backdoor2.png", x)
                # print(y)
                # plt.show()
            elif pattern_type == 'square_cl':
                if agent_idx == -1:
                    for d in range(3):
                        if d == 2:
                            x[1, 2:6, d] = 0
                            x[1, 8:12, d] = 0
                            x[3, 2:6, d] = 0
                            x[3, 8:12, d] = 0
                        else:
                            x[1, 2:6, d] = 255
                            x[1, 8:12, d] = 255
                            x[3, 2:6, d] = 255
                            x[3, 8:12, d] = 255
                else:
                    if attack == "DBA":
                        if agent_idx % 4 == 0:
                            for d in range(3):
                                if d == 2:
                                    x[1, 2:6, d] = 0
                                else:
                                    x[1, 2:6, d] = 255
                        elif agent_idx % 4 == 1:
                            for d in range(3):
                                if d == 2:
                                    x[1, 8:12, d] = 0
                                else:
                                    x[1, 8:12, d] = 255
                        elif agent_idx % 4 == 2:
                            for d in range(3):
                                if d == 2:
                                    x[3, 2:6, d] = 0
                                else:
                                    x[3, 2:6, d] = 255
                        elif agent_idx % 4 == 3:
                            for d in range(3):
                                if d == 2:
                                    x[3, 8:12, d] = 0
                                else:
                                    x[3, 8:12, d] = 255
                    else:
                        for d in range(3):
                            if d == 2:
                                x[1, 2:6, d] = 0
                                x[1, 8:12, d] = 0
                                x[3, 2:6, d] = 0
                                x[3, 8:12, d] = 0
                            else:
                                x[1, 2:6, d] = 255
                                x[1, 8:12, d] = 255
                                x[3, 2:6, d] = 255
                                x[3, 8:12, d] = 255
    # elif dataset == "GTSRB":
        # if pattern_type == "plus":
        #     start_idx = 5
        #     size = 6
        #     # vertical line
        #     if agent_idx == -1:
        #         for d in range(0, 3):
        #             for i in range(start_idx, start_idx + size + 1):
        #                 if d == 2:
        #                     x[d][i][start_idx] = 0
        #                 else:
        #                     x[d][i][start_idx] = 1
        #         # horizontal line
        #         for d in range(0, 3):
        #             for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
        #                 if d == 2:
        #                     x[d][start_idx + size // 2][i] = 0
        #                 else:
        #                     x[d][start_idx + size // 2][i] = 1
        #     else:
        #         if attack == "DBA":
        #             if agent_idx % 4 == 0:
        #                 # upper part of vertical
        #                 for d in range(0, 3):
        #                     for i in range(start_idx, start_idx + (size // 2) + 1):
        #                         if d == 2:
        #                             x[d][i][start_idx] = 0
        #                         else:
        #                             x[d][i][start_idx] = 1
        #             elif agent_idx % 4 == 1:
        #                 for d in range(0, 3):
        #                     # lower part of vertical
        #                     for i in range(start_idx + (size // 2) + 1, start_idx + size + 1):
        #                         if d == 2:
        #                             x[d][i][start_idx] = 0
        #                         else:
        #                             x[d][i][start_idx] = 1
        #             elif agent_idx % 4 == 2:
        #                 for d in range(0, 3):
        #                     # left-part of horizontal
        #                     for i in range(start_idx - size // 2, start_idx - size // 4 + 1):
        #                         if d == 2:
        #                             x[d][start_idx + size // 2][i] = 0
        #                         else:
        #                             x[d][start_idx + size // 2][i] = 1
        #             elif agent_idx % 4 == 3:
        #                 for d in range(0, 3):
        #                     # right-part of horizontal
        #                     for i in range(start_idx - size // 4 + 1, start_idx + size // 2 + 1):
        #                         if d == 2:
        #                             x[d][start_idx + size // 2][i] = 0
        #                         else:
        #                             x[d][start_idx + size // 2][i] = 1
                
    elif dataset == 'tinyimagenet':
        if pattern_type == 'plus':
            start_idx = 5
            size = 6
            # vertical line
            for d in range(0, 3):
                for i in range(start_idx, start_idx + size + 1):
                    if d == 2:
                        x[d][i][start_idx] = 0
                    else:
                        x[d][i][start_idx] = 1
            # horizontal line
            for d in range(0, 3):
                for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                    if d == 2:
                        x[d][start_idx + size // 2][i] = 0
                    else:
                        x[d][start_idx + size // 2][i] = 1
        elif pattern_type == 'square':
            for d in range(3):
                for i in range(4, 10):
                    for j in range(4, 10):
                        x[d][i][j] = 1
            
            # if agent_idx == -1:
            #     # plt.imsave("visualization/input_images/backdoor2.png", x)
            #     print(y)
            #     plt.show()
            # plt.savefig()

    elif dataset == 'fmnist':
        x = np.array(x.squeeze())
        if pattern_type == 'plus':
            start_idx = 5
            size = 6
            if agent_idx == -1:
                # vertical line
                for i in range(start_idx, start_idx + size + 1):
                    x[i, start_idx] = 255
                # horizontal line
                for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                    x[start_idx + size // 2, i] = 255
            else:
                if attack == "DBA":
                    # DBA attack
                    # upper part of vertical
                    if agent_idx % 4 == 0:
                        for i in range(start_idx, start_idx + (size // 2) + 1):
                            x[i, start_idx] = 255

                    # lower part of vertical
                    elif agent_idx % 4 == 1:
                        for i in range(start_idx + (size // 2) + 1, start_idx + size + 1):
                            x[i, start_idx] = 255

                    # left-part of horizontal
                    elif agent_idx % 4 == 2:
                        for i in range(start_idx - size // 2, start_idx - size // 4 + 1):
                            x[start_idx + size // 2, i] = 255

                    # right-part of horizontal
                    elif agent_idx % 4 == 3:
                        for i in range(start_idx - size // 4 + 1, start_idx + size // 2 + 1):
                            x[start_idx + size // 2, i] = 255
                else:
                    # vertical line
                    for i in range(start_idx, start_idx + size + 1):
                        x[i, start_idx] = 255
                    # horizontal line
                    for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                        x[start_idx + size // 2, i] = 255 

    elif dataset == 'mnist':
        x = np.array(x.squeeze())
        if pattern_type == 'plus':
            start_idx = 1
            size = 2
            if agent_idx == -1:
                # vertical line
                for i in range(start_idx, start_idx + size + 1):
                    x[i, start_idx] = 255
                # horizontal line
                for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                    x[start_idx + size // 2, i] = 255
            else:
                if attack == "DBA":
                    # DBA attack
                    # upper part of vertical
                    if agent_idx % 4 == 0:
                        for i in range(start_idx, start_idx + (size // 2) + 1):
                            x[i, start_idx] = 255

                    # lower part of vertical
                    elif agent_idx % 4 == 1:
                        for i in range(start_idx + (size // 2) + 1, start_idx + size + 1):
                            x[i, start_idx] = 255

                    # left-part of horizontal
                    elif agent_idx % 4 == 2:
                        for i in range(start_idx - size // 2, start_idx - size // 4 + 1):
                            x[start_idx + size // 2, i] = 255

                    # right-part of horizontal
                    elif agent_idx % 4 == 3:
                        for i in range(start_idx - size // 4 + 1, start_idx + size // 2 + 1):
                            x[start_idx + size // 2, i] = 255
                else:
                    # vertical line
                    for i in range(start_idx, start_idx + size + 1):
                        x[i, start_idx] = 255
                    # horizontal line
                    for i in range(start_idx - size // 2, start_idx + size // 2 + 1):
                        x[start_idx + size // 2, i] = 255
    # import matplotlib.pyplot as plt
    # if agent_idx == -1:
    #     # plt.imsave("visualization/input_images/backdoor2.png", x)
    #     plt.imshow(x)
    #     print(y)
    #     plt.show()
    return x

def gap_statistics(data, num_sampling, K_max, n):
    num_cluster = 0
    data = np.reshape(data, (data.shape[0], -1))
    # Linear transformation
    data_c = np.ndarray(shape=data.shape)
    for i in range(data.shape[1]):
        data_c[:,i] = (data[:,i] - np.min(data[:,i])) / \
                (np.max(data[:,i]) - np.min(data[:,i]))

    gap = []
    s = []
    for k in range(1, K_max + 1):
        k_means = KMeans(n_clusters=k, init='k-means++', n_init='auto').fit(data_c)
        predicts = (k_means.labels_).tolist()
        centers = k_means.cluster_centers_
        v_k = 0
        for i in range(k):
            for predict in predicts:
                if predict == i:
                    v_k += np.linalg.norm(centers[i] - \
                                data_c[predicts.index(predict)])
        # perform clustering on fake data
        v_kb = []
        for _ in range(num_sampling):
            data_fake = []
            for i in range(n):
                temp = np.ndarray(shape=(1,data.shape[1]))
                for j in range(data.shape[1]):
                    temp[0][j] = random.uniform(0,1)
                data_fake.append(temp[0])
            k_means_b = KMeans(n_clusters=k, init='k-means++', n_init='auto').fit(data_fake)
            predicts_b = (k_means_b.labels_).tolist()
            centers_b = k_means_b.cluster_centers_
            v_kb_i = 0
            for i in range(k):
                for predict in predicts_b:
                    if predict == i:
                        v_kb_i += np.linalg.norm(centers_b[i] - \
                                data_fake[predicts_b.index(predict)])
            v_kb.append(v_kb_i)
        # gap for k
        v = 0
        for v_kb_i in v_kb:
            v += math.log(v_kb_i)
        v /= num_sampling
        gap.append(v - math.log(v_k))
        sd = 0
        for v_kb_i in v_kb:
            sd += (math.log(v_kb_i) - v)**2
        sd = math.sqrt(sd / num_sampling)
        s.append(sd * math.sqrt((1 + num_sampling) / num_sampling))

    # select smallest k
    for k in range(1, K_max + 1):
        # print(gap[k - 1] - gap[k] + s[k])
        if k == K_max:
            num_cluster = K_max
            break
        if gap[k - 1] - gap[k] + s[k] > 0:
            num_cluster = k
            break

    return num_cluster

class NoiseDataset(torch.utils.data.Dataset):

    def __init__(self, size, num_samples):
        self.size = size
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        noise = torch.rand(self.size)
        noise = noise.cuda()
        return noise, 0

def dists_from_clust(clusters, N):
    pairwise_dists = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            pairwise_dists[i,j] = 0 if clusters[i] == clusters[j] else 1
    return pairwise_dists

def cluster(init_ids, data):
    clusterer = KMeans(n_clusters=len(init_ids), init=[data[i] for i in init_ids], n_init=1)
    cluster_labels = clusterer.fit_predict(data)
    return cluster_labels

def print_exp_details(args):
    print('======================================')
    print(f'    Dataset: {args.data}')
    print(f'    Global Rounds: {args.rounds}')
    print(f'    Aggregation Function: {args.aggr}')
    print(f'    Number of agents: {args.num_agents}')
    print(f'    Fraction of agents: {args.agent_frac}')
    print(f'    Batch size: {args.bs}')
    print(f'    Client_LR: {args.client_lr}')
    print(f'    Server_LR: {args.server_lr}')
    print(f'    Client_Momentum: {args.client_moment}')
    print(f'    RobustLR_threshold: {args.robustLR_threshold}')
    print(f'    Noise Ratio: {args.noise}')
    print(f'    Number of corrupt agents: {args.num_corrupt}')
    print(f'    Poison Frac: {args.poison_frac}')
    print(f'    Clip: {args.clip}')
    print('======================================')


def choose_agent(args):
    chosen = []
    if args.fix_attack == True:
        if args.num_agents - args.num_corrupt >= 0: 
            chosen.extend(random.sample(range(args.num_corrupt), int(args.num_corrupt * args.agent_frac)))
            chosen.extend(random.sample(range(args.num_corrupt, args.num_agents),
                                 int(args.num_agents * args.agent_frac) - int(args.num_corrupt * args.agent_frac)))
        else:
            print("args.num_agents * args.agent_frac < args.num_corrupt !!!")
    else:
        chosen.extend(np.random.choice(args.num_agents, math.floor(args.num_agents * args.agent_frac), replace=False))
    return chosen

def get_env(args):
    env_config = {
        "fmnist": {"input_channel": 1, "input_height": 28, "input_width": 28, "num_classes": 10},
        "cifar10": {"input_channel": 3, "input_height": 32, "input_width": 32, "num_classes": 10},
        "cifar100": {"input_channel": 3, "input_height": 32, "input_width": 32, "num_classes": 100},
        "tinyimagenet": {"input_channel": 3, "input_height": 64, "input_width": 64, "num_classes": 200},
        "GTSRB": {"input_channel": 3, "input_height": 32, "input_width": 32, "num_classes": 43},
    }
    
    if args.data not in env_config:
        raise ValueError("Unsupported dataset!")
    
    return env_config[args.data]


# class GTSRB_Modifiable(VisionDataset):
#     def __init__(
#         self,
#         root: Union[str, pathlib.Path],
#         split: str = "train",
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         download: bool = False,
#     ) -> None:
#         super().__init__(root, transform=transform, target_transform=target_transform)

#         self._split = verify_str_arg(split, "split", ("train", "test"))
#         self._base_folder = pathlib.Path(root) / "gtsrb"
#         self._target_folder = (
#             self._base_folder / "GTSRB" / ("Training" if self._split == "train" else "Final_Test/Images")
#         )

#         if download:
#             self.download()

#         if not self._check_exists():
#             raise RuntimeError("Dataset not found. You can use download=True to download it")

#         self.data = []
#         self.targets = []

#         if self._split == "train":
#             samples = list(self._target_folder.rglob("*.ppm"))
#             for path in samples:
#                 img = Image.open(path).convert("RGB")
#                 label = int(path.parent.name)
#                 if self.transform:
#                     img = self.transform(img)
#                 self.data.append(img)
#                 self.targets.append(label)
#         else:
#             with open(self._base_folder / "GT-final_test.csv") as csv_file:
#                 reader = csv.DictReader(csv_file, delimiter=";", skipinitialspace=True)
#                 for row in reader:
#                     img_path = self._target_folder / row["Filename"]
#                     img = Image.open(img_path).convert("RGB")
#                     label = int(row["ClassId"])
#                     if self.transform:
#                         img = self.transform(img)
#                     self.data.append(img)
#                     self.targets.append(label)

#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         img, target = self.data[index], self.targets[index]


#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         target = torch.tensor(target, dtype=torch.long)

#         return img, target

#     def __len__(self) -> int:
#         return len(self.data)

#     def _check_exists(self) -> bool:
#         return self._target_folder.is_dir()

#     def download(self) -> None:
#         if self._check_exists():
#             return

#         base_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"

#         if self._split == "train":
#             download_and_extract_archive(
#                 f"{base_url}GTSRB-Training_fixed.zip",
#                 download_root=str(self._base_folder),
#                 md5="513f3c79a4c5141765e10e952eaa2478",
#             )
#         else:
#             download_and_extract_archive(
#                 f"{base_url}GTSRB_Final_Test_Images.zip",
#                 download_root=str(self._base_folder),
#                 md5="c7e4e6327067d32654124b0fe9e82185",
#             )
#             download_and_extract_archive(
#                 f"{base_url}GTSRB_Final_Test_GT.zip",
#                 download_root=str(self._base_folder),
#                 md5="fe31e9c9270bbcd7b84b7f21a9d9d9e5",
#             )