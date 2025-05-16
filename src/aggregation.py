import copy

import torch
import models
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import numpy as np
from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
import logging
from utils import name_param_to_array,  vector_to_model, vector_to_name_param, gap_statistics, dists_from_clust, NoiseDataset, cluster
import hdbscan
from sklearn.cluster import KMeans
import sklearn.metrics.pairwise as smp
import math
from geom_median.torch import compute_geometric_median 
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from snowball import kl_loss, recon_loss, MyDST, _flatten_model, _init_weights, train_vae, obtain_dif, build_dif_set
from FDCR import FINCH

class Aggregation():
    def __init__(self, agent_data_sizes, n_params, poisoned_val_loader, args, writer):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.writer = writer
        self.server_lr = args.server_lr
        self.n_params = n_params
        self.poisoned_val_loader = poisoned_val_loader
        self.cum_net_mov = 0
        self.true_positive = 0
        self.true_negative = 0
        self.id_record = []
        self.norm_record = []
        
        self.false_positive = 0
        self.false_negative = 0
        if self.args.aggr == "fg":
            model_temp = models.get_model(args.data, args.model).to(args.device)
            self.update_history = []
            self.last_layer_name = "linear" if "cifar" in self.args.data else "fc2"
            if "cifar" in self.args.data:
                last_weight = model_temp.linear.weight.detach().clone().view(-1)
                last_bias = model_temp.linear.bias.detach().clone().view(-1)
            else:
                last_weight = model_temp.fc2.weight.detach().clone().view(-1)
                last_bias = model_temp.fc2.bias.detach().clone().view(-1)
            last_params = torch.cat((last_weight, last_bias))

            for _ in range(int(self.args.num_agents)):
                last_layer_params = torch.zeros_like(last_params)
                self.update_history.append(last_layer_params)
        
    def aggregate_updates(self, global_model, agent_updates_dict, rnd, local_fish_dict = None):
        self.rnd = rnd
        lr_vector = torch.Tensor([self.server_lr]*self.n_params).to(self.args.device)
        if self.args.method != "rlr":
            lr_vector=lr_vector
        else:
            lr_vector, _ = self.compute_robustLR(agent_updates_dict, global_model)
        # mask = torch.ones_like(agent_updates_dict[0])
        aggregated_updates = 0
        cur_global_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        if self.args.aggr=='avg':          
            aggregated_updates = self.agg_avg(agent_updates_dict, global_model)
        elif self.args.aggr== "clip_avg":
            for _id, update in agent_updates_dict.items():
                weight_diff_norm = torch.norm(update).item()
                logging.info(weight_diff_norm)
                update.data = update.data / max(1, weight_diff_norm / 2)
            aggregated_updates = self.agg_avg(agent_updates_dict, global_model)
            logging.info(torch.norm(aggregated_updates))
        elif self.args.aggr == "krum":
            aggregated_updates = self.agg_krum(agent_updates_dict)
        elif self.args.aggr == "mul_krum":
            aggregated_updates = self.agg_mul_krum(agent_updates_dict)
        elif self.args.aggr == "flame":
            aggregated_updates = self.agg_flame(agent_updates_dict, global_model)
        elif self.args.aggr == "rflbat":
            aggregated_updates = self.agg_rflbat(agent_updates_dict, global_model)
        elif self.args.aggr == "deepsight":
            aggregated_updates = self.agg_deepsight(agent_updates_dict, global_model)
        elif self.args.aggr == 'mul_metric':
            aggregated_updates = self.agg_mul_metric(agent_updates_dict, global_model)
        elif self.args.aggr == 'fg':
            aggregated_updates = self.agg_foolsgold(agent_updates_dict, global_model)
        elif self.args.aggr == "rfa":
            aggregated_updates = self.agg_rfa(agent_updates_dict)
        elif self.args.aggr == "snowball":
            aggregated_updates = self.agg_snowball(agent_updates_dict, global_model, rnd)
        elif self.args.aggr == "snowball-":
            aggregated_updates = self.agg_snowball(agent_updates_dict, global_model, rnd)
        elif self.args.aggr == "FDCR":
            aggregated_updates = self.agg_FDCR(agent_updates_dict, global_model, local_fish_dict)
         
        updates_dict = vector_to_name_param(aggregated_updates, copy.deepcopy(global_model.state_dict()))
        neurotoxin_mask = {}
        for name in updates_dict:
            updates = updates_dict[name].abs().view(-1)
            gradients_length = torch.numel(updates)
            _, indices = torch.topk(-1 * updates, int(gradients_length * 0.99))
            mask_flat = torch.zeros(gradients_length)
            mask_flat[indices.cpu()] = 1
            neurotoxin_mask[name] = (mask_flat.reshape(updates_dict[name].size()))

        cur_global_params = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float()
        vector_to_model(new_global_params, global_model)
        return    updates_dict, neurotoxin_mask


    def compute_robustLR(self, agent_updates_dict, global_model):

        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        mask=torch.zeros_like(sm_of_signs)
        theta = self.args.rlr_theta
        mask[sm_of_signs < theta] = 0
        mask[sm_of_signs >= theta] = 1
        sm_of_signs[sm_of_signs < theta] = -self.server_lr
        sm_of_signs[sm_of_signs >= theta] = self.server_lr
        return sm_of_signs.to(self.args.device), mask

    # def compute_robustLR(self, agent_updates_dict, global_model):
    #     theta = 1/2 * int(self.args.num_agents * self.args.agent_frac) 
    #     agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]  
    #     sm_of_signs = torch.abs(sum(agent_updates_sign))
    #     index = 0
    #     for name, param in global_model.state_dict().items():
    #         param_len = torch.numel(param)
    #         if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
    #             sm_of_signs[index:index+param_len] = self.server_lr
    #         else:
    #             sm_of_signs[index:index+param_len][sm_of_signs[index:index+param_len] < theta] = -self.server_lr
    #             sm_of_signs[index:index+param_len][sm_of_signs[index:index+param_len] >= theta] = self.server_lr
    #         index += param_len
    #     return sm_of_signs.to(self.args.device)

    def agg_krum(self, agent_updates_dict):
        krum_param_m = 1
        def _compute_krum_score( vec_grad_list, byzantine_client_num):
            krum_scores = []
            num_client = len(vec_grad_list)
            for i in range(0, num_client):
                dists = []
                for j in range(0, num_client):
                    if i != j:
                        dists.append(
                            torch.norm(vec_grad_list[i]- vec_grad_list[j])
                            .item() ** 2
                        )
                dists.sort()  # ascending
                score = dists[0: num_client - byzantine_client_num - 2]
                krum_scores.append(sum(score))
            return krum_scores

        # Compute list of scores
        __nbworkers = len(agent_updates_dict)
        krum_scores = _compute_krum_score(agent_updates_dict, self.args.num_corrupt)
        score_index = torch.argsort(
            torch.Tensor(krum_scores)
        ).tolist()  # indices; ascending
        score_index = score_index[0: krum_param_m]
        return_gradient = [agent_updates_dict[i] for i in score_index]
        return sum(return_gradient)/len(return_gradient)

    def agg_avg(self, agent_updates_dict, global_model):
        """ classic fed avg """
        if self.args.norm_log == True:
            id_list = []
            updates_state_dict_list = []
            global_model_state = parameters_to_vector([global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
            for id, update in agent_updates_dict.items():
                id_list.append(id)
                updates_state_dict_list.append(vector_to_name_param(update, copy.deepcopy(global_model.state_dict())))
                
            local_name_param = []
            for update in updates_state_dict_list:
                name_param = []    
                for name, param in update.items():
                    if "running_mean" not in name and "running_var" not in name and "num_batches_tracked" not in name:
                        name_param.append(param.view(-1))
                name_param = torch.cat(name_param)
                local_name_param.append(name_param)
            norm_list = [] 
            for i in range(len(local_name_param)):
                norm_list = np.append(norm_list,torch.norm(local_name_param[i]).item())
            
            logging.info(f"id list: {id_list}")
            logging.info(f"norm list: {norm_list}")
            self.id_record.append(id_list)
            self.norm_record.append(norm_list)
            
            
        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates +=  n_agent_data * update
            total_data += n_agent_data
        return  sm_updates / total_data

    def agg_deepsight(self, agent_updates_dict, global_model):
        self.sub_local_model = copy.deepcopy(global_model)
        self.sub_global_model = copy.deepcopy(global_model)
        local_model_vector = []
        local_model_state_dict = []
        id_list = []
        updates_state_dict_list = []
        global_model_params = parameters_to_vector([global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        for id, update in agent_updates_dict.items():
            id_list.append(id)
            local_model_vector.append(update + global_model_params)
            model_state_dict = vector_to_name_param(update + global_model_params, copy.deepcopy(global_model.state_dict()))
            local_model_state_dict.append(model_state_dict)
            updates_state_dict_list.append(vector_to_name_param(update, copy.deepcopy(global_model.state_dict())))
            
        num_seeds = self.args.ds_num_seed
        tau = self.args.ds_tau

        last_layer_name = "linear" if "cifar" in self.args.data else "fc2"
        if self.args.data.upper() == "CIFAR10":
            num_classes = 10
            dim = 32
            num_channel = 3
            num_samples = 20000
        elif self.args.data.upper() == "CIFAR100":
            num_classes = 100
            dim = 32
            num_channel = 3
            num_samples = 20000
        elif self.args.data.upper() == "EMNIST" or self.args.data.upper() == "FMNIST":
            num_classes = 10
            dim = 28
            num_channel = 1
            num_samples = 20000
        elif self.args.data.upper() == "TINYIMAGENET":
            num_classes = 200
            dim = 64
        
        
        ### computing NEUPs and TEs
        TEs, NEUPs, ed = [], [], []
        for enu_id, local_model_state_dict_sub in enumerate(local_model_state_dict):
            ### get updated params norm
            squared_sum = 0
            for name, value in local_model_state_dict_sub.items():
                if "tracked" in name or "running" in name:
                    continue
                squared_sum += torch.sum(torch.pow(value-global_model.state_dict()[name], 2)).item()
            update_norm = math.sqrt(squared_sum)
            ed = np.append(ed, update_norm)

            diff_bias = local_model_state_dict_sub[f"{last_layer_name}.bias"] \
                - global_model.state_dict()[f"{last_layer_name}.bias"]
            diff_weight = local_model_state_dict_sub[f"{last_layer_name}.weight"] \
                - global_model.state_dict()[f"{last_layer_name}.weight"]
            
            UPs = abs(diff_bias.cpu().numpy()) + np.sum(abs(diff_weight.cpu().numpy()), axis=1)
            NEUP = UPs**2/np.sum(UPs**2)
            TE = 0

            for j in NEUP:
                if j>= (1/num_classes)*np.max(NEUP):
                    TE += 1
            NEUPs = np.append(NEUPs, NEUP)
            TEs.append(TE)
        
        logging.info("Deepsight: Finish cauculating TE")
        labels = []
        for i in TEs:
            if i >= np.median(TEs)/2:
                labels.append(0)
            else:
                labels.append(1)
        logging.info(f"computed TEs:{TEs}")
        logging.info(f"computed labels:{labels}")
        
        ### computing DDifs
        DDifs = []
        for i, seed in enumerate(range(num_seeds)):
            torch.manual_seed(seed)
            dataset = NoiseDataset([num_channel, dim, dim], num_samples)
            loader = torch.utils.data.DataLoader(dataset, 1000, shuffle=False)

            for enu_id, local_model_state_dict_sub in enumerate(local_model_state_dict):
                self.sub_local_model.copy_params(local_model_state_dict_sub)
                self.sub_global_model.copy_params(global_model.state_dict())
                self.sub_local_model.eval()
                self.sub_global_model.eval()

                DDif = torch.zeros(num_classes).to(self.args.device)
                for y in loader:
                    x,_ = y 
                    x = x.to(self.args.device)
                    with torch.no_grad():
                        output_local = self.sub_local_model(x)
                        output_global = self.sub_global_model(x)

                        output_local = torch.softmax(output_local, dim=1)
                        output_global = torch.softmax(output_global, dim=1)

                    temp = torch.div(output_local, output_global+1e-30) # avoid zero-value
                    temp = torch.sum(temp, dim=0)
                    DDif.add_(temp)

                DDif /= num_samples
                DDifs = np.append(DDifs, DDif.cpu().numpy())

        DDifs = np.reshape(DDifs, (num_seeds, len(local_model_state_dict), -1))
        logging.info("Deepsight: Finish cauculating DDifs")

        ### compute cosine distance
        bias_name = f"{last_layer_name}.bias"
        cosine_distance = np.zeros((len(local_model_state_dict), len(local_model_state_dict)))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()

        for ind_0, local_model_state_dict_sub_0 in enumerate(local_model_state_dict):
            for ind_1, local_model_state_dict_sub_1 in enumerate(local_model_state_dict):
                update_0 =  local_model_state_dict_sub_0[bias_name] \
                    - global_model.state_dict()[bias_name]
                update_1 =  local_model_state_dict_sub_1[bias_name]\
                    - global_model.state_dict()[bias_name]
                # cosine_distance[ind_0, ind_1] = 1.0 - dot(update_0, update_1)/(norm(update_0)*norm(update_1))
                cosine_distance[ind_0, ind_1] = 1.0 - cos(update_0, update_1)
        logging.info("Deepsight: Finish cauculating cosine distance")

        # classification
        cosine_clusters = hdbscan.HDBSCAN(min_samples=1, allow_single_cluster=True).fit_predict(cosine_distance)
        logging.info(f"cosine cluster:{cosine_clusters}")
        cosine_cluster_dists = dists_from_clust(cosine_clusters, len(local_model_state_dict))

        NEUPs = np.reshape(NEUPs, (len(local_model_state_dict), num_classes))
        neup_clusters = hdbscan.HDBSCAN(min_samples=1, allow_single_cluster=True).fit_predict(NEUPs)
        logging.info(f"neup cluster:{neup_clusters}")
        neup_cluster_dists = dists_from_clust(neup_clusters, len(local_model_state_dict))

        ddif_clusters, ddif_cluster_dists = [],[]
        for i in range(num_seeds):
            DDifs[i] = np.reshape(DDifs[i], (len(local_model_state_dict), num_classes))
            ddif_cluster_i = hdbscan.HDBSCAN(min_samples=1, allow_single_cluster=True).fit_predict(DDifs[i])
            logging.info(f"ddif cluster:{ddif_cluster_i}")
            ddif_cluster_dists = np.append(ddif_cluster_dists,
                dists_from_clust(ddif_cluster_i, len(local_model_state_dict)))

        merged_ddif_cluster_dists = np.mean(np.reshape(ddif_cluster_dists,
                    (num_seeds, int(self.args.agent_frac * self.args.num_agents), int(self.args.agent_frac * self.args.num_agents))),
                    axis=0)
        
        merged_distances = np.mean([merged_ddif_cluster_dists,
                                    neup_cluster_dists,
                                    cosine_cluster_dists], axis=0)

        clusters = hdbscan.HDBSCAN(min_cluster_size=2 ,min_samples=1, allow_single_cluster=True).fit_predict(merged_distances)
        # clusters = hdbscan.HDBSCAN(min_samples=1, allow_single_cluster=True).fit_predict(merged_distances)
        logging.info(f"cluster label:{clusters}")

        final_clusters = np.asarray(clusters)
        cluster_list = np.unique(final_clusters)
        benign_index = []
        labels = np.asarray(labels)
        for cluster in cluster_list:
            if cluster == -1:
                indexes = np.argwhere(final_clusters==cluster).flatten()
                for i in indexes:
                    if labels[i] == 1:
                        continue
                    else:
                        benign_index.append(i)
            else:
                indexes = np.argwhere(final_clusters==cluster).flatten()
                amount_of_suspicious = np.sum(labels[indexes])/len(indexes)
                if amount_of_suspicious < tau:
                    for idx in indexes:
                        benign_index.append(idx)

        # Aggregate and norm-clipping
        clip_value = np.median(ed)
        benign_clients = [id_list[i] for i in benign_index]
        
        if benign_clients == []:
            return torch.zeros_like(list(agent_updates_dict.values())[0])
        self.cal_metrics(id_list, benign_clients)
        logging.info(f"all clients: {id_list}")
        logging.info(f"benign_clients: {benign_clients}")
        self.log_metrics()
        for index, update_name_param in enumerate(updates_state_dict_list):
            gama = clip_value/ed[index]
            for name, update_param in update_name_param.items():
                if "running_mean" not in name and "running_var" not in name and "num_batches_tracked" not in name:
                    update_param.data = update_param.data * gama
        global_update = copy.deepcopy(global_model.state_dict())
        for name, param in global_update.items():
            global_update[name] = torch.zeros_like(param)
        total_client = 0
        for id, update_name_param in zip(id_list, updates_state_dict_list):
            if id in benign_clients:
                total_client += 1
                for name, param in update_name_param.items():
                    global_update[name].data = global_update[name].data.float() + param.data
        for name, global_param in global_update.items():
            global_param.data /= total_client
        logging.info(f"clip_value: {clip_value}")
        
        return parameters_to_vector([global_update[name] for name in global_update.keys()])

    def agg_rflbat(self, agent_updates_dict, global_model):
        eps1 = 10
        eps2 = 4
        updates_state_dict_list = []
        id_list = []
        global_model_params = parameters_to_vector([global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        for id, update in agent_updates_dict.items():
            updates_state_dict_list.append(vector_to_name_param(update, copy.deepcopy(global_model.state_dict())))
            id_list.append(id)
        update_params = []
        for updata_state_dict in updates_state_dict_list:
            update_name_param = []
            for name, param in updata_state_dict.items():
                if "running_mean" not in name and "running_var" not in name and "num_batches_tracked" not in name:
                    update_name_param.append(param.view(-1))
            update_name_param = torch.cat(update_name_param)
            update_params.append(update_name_param)
        update_params_tensor = copy.deepcopy(update_params)
        for ind, update_param in enumerate(update_params):
            update_param = update_param.cpu().numpy().tolist()
            update_params[ind] = update_param
        
        update_params_tensor = torch.stack(update_params_tensor)
        U, S, V = torch.pca_lowrank(update_params_tensor)
        X_dr = torch.mm(update_params_tensor, V[:,:]).cpu().numpy()
        
        eu_list = []
        for i in range(len(X_dr)):
            eu_sum = 0
            for j in range(len(X_dr)):
                if i==j:
                    continue
                eu_sum += np.linalg.norm(X_dr[i]-X_dr[j])
            eu_list.append(eu_sum)
        
        accept = []
        x1 = []

        for i in range(len(eu_list)):
            if eu_list[i] < eps1 * np.median(eu_list):
                accept.append(i)
                x1 = np.append(x1, X_dr[i])
        
        x1 = np.reshape(x1, (-1, X_dr.shape[1]))
        num_clusters = gap_statistics(data=x1, num_sampling=5, \
                                           K_max=9, n=len(x1))
        logging.info(f"num of selected clusters:{num_clusters}")
        k_means = KMeans(n_clusters=num_clusters, init='k-means++', n_init='auto').fit(x1)
        predicts = k_means.labels_

        v_med = []
        for i in range(num_clusters):
            temp = []
            for j in range(len(predicts)):
                if predicts[j] == i:
                    temp.append(update_params[accept[j]])
            if len(temp) <= 1:
                v_med.append(1)
                continue
            v_med.append(np.median(np.average(smp\
                .cosine_similarity(temp), axis=1)))
        temp = []
        for i in range(len(accept)):
            if predicts[i] == v_med.index(min(v_med)):
                temp.append(accept[i])
        accept = temp

        # compute eu list again to exclude outliers
        temp = []
        for i in accept:
            temp.append(X_dr[i])
        X_dr = temp
        eu_list = []
        for i in range(len(X_dr)):
            eu_sum = 0
            for j in range(len(X_dr)):
                if i==j:
                    continue
                eu_sum += np.linalg.norm(X_dr[i]-X_dr[j])
            eu_list.append(eu_sum)
        temp = []
        for i in range(len(eu_list)):
            if eu_list[i] < eps2 * np.median(eu_list):
                temp.append(accept[i])

        accept = temp
        benign_clients = [id_list[i] for i in accept]
        self.cal_metrics(id_list, benign_clients)
        if benign_clients == None:
            return torch.zeros_like(agent_updates_dict.values()[0])
        logging.info(f"all clients: {id_list}")
        logging.info(f"benign_clients: {benign_clients}")
        self.log_metrics()
        return_gradient = [agent_updates_dict[i] for i in benign_clients]

        return sum(return_gradient)/len(return_gradient)
           
    def agg_flame(self, agent_updates_dict, global_model):
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
        cos_list = []
        local_model_vector = []
        id_list = []
        updates_state_dict_list = []
        global_model_state = parameters_to_vector([global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        for id, update in agent_updates_dict.items():
            id_list.append(id)
            local_model_vector.append(update + global_model_state)
            updates_state_dict_list.append(vector_to_name_param(update, copy.deepcopy(global_model.state_dict())))
            
        local_name_param = []
        for update in updates_state_dict_list:
            name_param = []    
            for name, param in update.items():
                if "running_mean" not in name and "running_var" not in name and "num_batches_tracked" not in name:
                    name_param.append(param.view(-1))
            name_param = torch.cat(name_param)
            local_name_param.append(name_param)
                    
        for i in range(len(local_name_param)):
            cos_i = []
            for j in range(len(local_name_param)):
                cos_ij = 1- cos(local_name_param[i], local_name_param[j])
                cos_i.append(round(cos_ij.item(),4))
                # cos_i.append(cos_ij.item())
            cos_list.append(cos_i)
        num_clients = max(int(self.args.agent_frac * self.args.num_agents), 1)
        num_malicious_clients = int(self.args.num_corrupt)
        num_benign_clients = num_clients - num_malicious_clients
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.args.min_claster_size, min_samples=1, allow_single_cluster=True).fit(cos_list)
        
        benign_index = []
        norm_list = np.array([])
        
        max_num_in_cluster=0
        max_cluster_index=0
        if clusterer.labels_.max() < 0:
            for i in range(len(local_model_vector)):
                benign_index .append(i)
                norm_list = np.append(norm_list,torch.norm(local_name_param[i]).item())
        else:
            for index_cluster in range(clusterer.labels_.max()+1):
                if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                    max_cluster_index = index_cluster
                    max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
            for i in range(len(clusterer.labels_)):
                if clusterer.labels_[i] == max_cluster_index:
                    benign_index .append(i)
        for i in range(len(local_model_vector)):
            norm_list = np.append(norm_list,torch.norm(local_name_param[i]).item())  
        benign_clients = [id_list[i] for i in benign_index]
        self.cal_metrics(id_list, benign_clients)
        logging.info(f"all clients: {id_list}")
        logging.info(f"benign_clients: {benign_clients}")
        self.log_metrics()
        if benign_clients == None:
            return torch.zeros_like(agent_updates_dict.values()[0])
        
        clip_value = np.median(norm_list)
        for index, update_name_param in enumerate(updates_state_dict_list):
            gama = clip_value/norm_list[index]
            for name, update_param in update_name_param.items():
                if "running_mean" not in name and "running_var" not in name and "num_batches_tracked" not in name:
                    update_param.data = update_param.data * gama
        global_update = copy.deepcopy(global_model.state_dict())
        for name, param in global_update.items():
            global_update[name] = torch.zeros_like(param)
        total_client = 0
        for id, update_name_param in zip(id_list, updates_state_dict_list):
            if id in benign_clients:
                total_client += 1
                for name, param in update_name_param.items():
                    global_update[name].data = global_update[name].data.float() + param.data
        for name, global_param in global_update.items():
            global_param.data /= total_client
        logging.info(f"clip_value: {clip_value}")
        for name, param in global_update.items():
            if "running_mean" not in name and "running_var" not in name and "num_batches_tracked" not in name:
                temp = copy.deepcopy(param.data)
                temp = temp.normal_(mean=0, std=math.pow(self.args.noise_sigma * clip_value, 2))                
                param.data = param.data + temp
        
        return parameters_to_vector([global_update[name] for name in global_update.keys()])

    def agg_mul_krum(self, agent_updates_dict):
        update_param_list = []
        agent_id = []
        for id, update in agent_updates_dict.items():
            agent_id.append(id)
            update_param_list.append(update)
        
        candidates = []
        candidate_indices = []
        remaining_updates = update_param_list
        all_indices = np.arange(len(update_param_list))
    
        while len(remaining_updates) > len(agent_id) - self.args.mul_krum_remain_cls:
            distances = []
            for update in remaining_updates:
                distance = []
                for update_ in remaining_updates:
                    distance.append(torch.norm((update - update_)) ** 2)
                distance = torch.Tensor(distance).float()
                distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
 
            distances = torch.sort(distances, dim=1)[0]
            scores = torch.sum(distances[:, :len(remaining_updates) - 2 - int(self.args.num_corrupt * self.args.agent_frac)], dim=1)
            indices = torch.argsort(scores)[:len(remaining_updates) - 2 - int(self.args.num_corrupt * self.args.agent_frac)]

            candidate_indices.append(all_indices[indices[0].cpu().numpy()]) # 添加一个下标
            all_indices = np.delete(all_indices, indices[0].cpu().numpy())
            candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
            remaining_updates.pop(indices[0])
            
        if candidate_indices == []:
            return torch.zeros_like(agent_updates_dict.values()[0])
        
        chosen_indice = [agent_id[i] for i in candidate_indices]
        self.cal_metrics(agent_id, chosen_indice)
        logging.info(f"all clients: {agent_id}")
        logging.info(f"benign_clients: {chosen_indice}")
        self.log_metrics()
        return_gradient = [agent_updates_dict[i] for i in chosen_indice]

        return sum(return_gradient)/len(return_gradient)

    def agg_rfa(self, agent_updates_dict):
        local_updates = []
        benign_id = []
        malicious_id = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update.cpu())
        
        n = len(local_updates)
        grads = torch.stack(local_updates, dim=0)
        weights = torch.ones(n)
        gw = compute_geometric_median(local_updates, weights).median
        for i in range(2):
            weights = torch.mul(weights, torch.exp(-1.0*torch.norm(grads-gw, dim=1)))
            if weights.sum() == 0:
                return gw.to(self.args.device)   
            gw = compute_geometric_median(local_updates, weights).median

        aggregated_model = gw
        return aggregated_model.to(self.args.device)    
        
    def agg_mul_metric(self, agent_updates_dict, global_model):
        cos_list = []
        local_model_vector = []
        id_list = []
        updates_state_dict_list = []
        global_model_state = parameters_to_vector([global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        for id, update in agent_updates_dict.items():
            id_list.append(id)
            local_model_vector.append(update + global_model_state)
            updates_state_dict_list.append(vector_to_name_param(update, copy.deepcopy(global_model.state_dict())))
            
        local_name_param = []
        for update in updates_state_dict_list:
            name_param = []    
            for name, param in update.items():
                if "running_mean" not in name and "running_var" not in name and "num_batches_tracked" not in name:
                    name_param.append(param.view(-1))
            name_param = torch.cat(name_param).cpu()
            local_name_param.append(name_param)
            
        vectorize_nets = local_name_param

        cos_dis = [0.0] * len(vectorize_nets)
        length_dis = [0.0] * len(vectorize_nets)
        manhattan_dis = [0.0] * len(vectorize_nets)
        for i, g_i in enumerate(vectorize_nets):
            for j in range(len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]

                    cosine_distance = float(
                        (1 - np.dot(g_i, g_j) / (np.linalg.norm(g_i) * np.linalg.norm(g_j))) ** 2)   #Compute the different value of cosine distance
                    manhattan_distance = float(np.linalg.norm(g_i - g_j, ord=1))    #Compute the different value of Manhattan distance
                    length_distance = np.abs(float(np.linalg.norm(g_i) - np.linalg.norm(g_j)))    #Compute the different value of Euclidean distance

                    cos_dis[i] += cosine_distance
                    length_dis[i] += length_distance
                    manhattan_dis[i] += manhattan_distance

        tri_distance = np.vstack([cos_dis, 
                                  manhattan_dis,
                                  length_dis]).T

        cov_matrix = np.cov(tri_distance.T)
        inv_matrix = np.linalg.inv(cov_matrix)

        ma_distances = []
        for i, g_i in enumerate(vectorize_nets):
            t = tri_distance[i]
            ma_dis = np.dot(np.dot(t, inv_matrix), t.T)
            ma_distances.append(ma_dis)

        scores = ma_distances
        # print(scores)

        p_num = self.args.mul_metric_remain_cls
        topk_ind = np.argpartition(scores, int(p_num))[:int(p_num)]   #sort

        # print(topk_ind)
        current_dict = {}
        
        benign_client = [id_list[i] for i in topk_ind]
        logging.info(f"all clients: {id_list}")
        logging.info(f"benign_clients: {benign_client}")
        
        self.cal_metrics(id_list, benign_client)
        self.log_metrics()
        for idx in topk_ind:
            current_dict[id_list[idx]] = agent_updates_dict[id_list[idx]]

        # return self.agg_avg_norm_clip(current_dict)
        update = self.agg_avg(current_dict, global_model)

        return update

    def agg_foolsgold(self, agent_updates_dict, global_model):
        def foolsgold(id_list):
            """
            :param grads:
            :return: compute similatiry and return weightings
            """
            selected_his = []
            num_clients = len(id_list)
            cs = np.zeros((num_clients, num_clients))
            for id in id_list:
                selected_his = np.append(selected_his, self.update_history[id].cpu().numpy())
            selected_his = np.reshape(selected_his, (num_clients, -1))
            for i in range(len(selected_his)):
                for j in range(len(selected_his)):
                    cs[i][j] = np.dot(selected_his[i], selected_his[j])/(np.linalg.norm(selected_his[i])*np.linalg.norm(selected_his[j]))
                                      
            cs = cs - np.eye(num_clients)
            maxcs = np.max(cs, axis=1) + 1e-5
            for i in range(num_clients):
                for j in range(num_clients):
                    if i==j:
                        continue
                    if maxcs[i] < maxcs[j]:
                        cs[i][j] = cs[i][j] * maxcs[i]/maxcs[j]
            
            wv = 1 - (np.max(cs,axis=1))
            wv[wv>1]=1
            wv[wv<0]=0
            wv = wv / np.max(wv)
            wv[(wv==1)] = .99
            wv = (np.log((wv/(1-wv)) + 1e-5 )+0.5)
            wv[(np.isinf(wv)+wv > 1)]=1
            wv[wv<0]=0
            
            return wv

        local_updates = []
        id_list = []

        for _id, update in agent_updates_dict.items():
            local_updates.append(update)
            id_list.append(_id)
        
        num_chosen_clients = len(id_list)

        client_grads = [vector_to_name_param(update, copy.deepcopy(global_model.state_dict())) for update in agent_updates_dict.values()]
        for i in range(num_chosen_clients):
            self.update_history[id_list[i]] += torch.cat((client_grads[i][f"{self.last_layer_name}.weight"].detach().clone().view(-1), 
                                   client_grads[i][f"{self.last_layer_name}.bias"].detach().clone().view(-1)))
            
        wv = foolsgold(id_list)  # Use FG

        # print(len(client_grads), len(wv))
        
        weighted_updates = [update * wv[i] for update, i in zip(agent_updates_dict.values(), range(len(wv)))]

        aggregated_model = torch.mean(torch.stack(weighted_updates, dim=0), dim=0)

        # print(aggregated_model.shape)

        return aggregated_model

    def agg_snowball(self, agent_updates_dict, global_model, rnd):
        local_model_vector = []
        id_list = []
        updates_state_dict_list = []
        global_model_state = parameters_to_vector([global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        for id, update in agent_updates_dict.items():
            id_list.append(id)
            local_model_vector.append(update + global_model_state)
            updates_state_dict_list.append(vector_to_name_param(update.detach().cpu(), copy.deepcopy(global_model.state_dict())))
            
        model_updates = updates_state_dict_list
        
        kernels = []
        for key in model_updates[0].keys():
            kernels.append([model_updates[idx_client][key] for idx_client in range(len(model_updates))])
                
        cnt = [0 for _ in range(len(model_updates))]
        for idx_layer, layer_name in enumerate(model_updates[0].keys()):
            if "cifar" in self.args.data:
                if ('conv1' not in layer_name and 'linear' not in layer_name) or "layer" in layer_name:
                # if layer_name != 'conv1.weight' and layer_name != 'layer1.0.conv1.weight' and 'linear' not in layer_name:
                    continue
            elif "mnist" in self.args.data:
                if 'cnn1' not in layer_name and 'fc2' not in layer_name:
                    continue
            benign_list_cur_layer = []
            score_list_cur_layer = []
            updates_kernel = [item.flatten().numpy() for item in kernels[idx_layer]]
            for idx_client in range(len(updates_kernel)):
                ddif = [updates_kernel[idx_client] - updates_kernel[i] for i in range(len(updates_kernel))]
                norms = np.linalg.norm(ddif, axis=1)
                norm_rank = np.argsort(norms)
                ct = self.args.snow_cluster_th
                suspicious_idx = norm_rank[-ct:]
                centroid_ids = [idx_client]
                centroid_ids.extend(suspicious_idx)
                cluster_result = cluster(centroid_ids, ddif)
                
                score_ = calinski_harabasz_score(ddif, cluster_result)
                benign_ids = np.argwhere(cluster_result == cluster_result[idx_client]).flatten() 

                benign_list_cur_layer.append(benign_ids)
                score_list_cur_layer.append(score_)
            
            score_list_cur_layer = np.array(score_list_cur_layer)
            std_, mean_ = np.std(score_list_cur_layer), np.mean(score_list_cur_layer)
            effective_ids = np.argwhere(score_list_cur_layer > 0).flatten()
            if len(effective_ids) < int(len(score_list_cur_layer) * 0.1):
                effective_ids = np.argsort(-score_list_cur_layer)[:int(len(score_list_cur_layer) * 0.1)]
                
            if np.max(score_list_cur_layer) - np.min(score_list_cur_layer) == 0:
                continue
            # score_list_cur_layer = (score_list_cur_layer - np.min(score_list_cur_layer)) / (np.max(score_list_cur_layer) - np.min(score_list_cur_layer))
            for idx_client in effective_ids:
                for idx_b in benign_list_cur_layer[idx_client]:
                    cnt[idx_b] += score_list_cur_layer[idx_client]
            
        cnt_rank = np.argsort(-np.array(cnt))
        selected_ids = cnt_rank[:math.ceil(self.args.snow_minus_remain_cls)].tolist()
        
        if ("cifar" in self.args.data and rnd <= 100) or self.args.aggr == "snowball-" or ("mnist" in self.args.data and rnd <= 30):
            benign_clients = [id_list[i] for i in selected_ids]
            self.cal_metrics(id_list, benign_clients)
            logging.info(f"all clients: {id_list}")
            logging.info(f"benign_clients: {benign_clients}")
            self.log_metrics()
            
            sm_updates, total_data = 0, 0
            for _id, update in agent_updates_dict.items():
                if _id in benign_clients:
                    n_agent_data = self.agent_data_sizes[_id]
                    sm_updates +=  n_agent_data * update
                    total_data += n_agent_data
            return  sm_updates / total_data
        
        else:
            if "cifar" in self.args.data:
                flatten_update_list = [_flatten_model(update, layer_list=['conv1', 'linear'], ignore='layer') for update in model_updates]
            elif "mnist" in self.args.data:
                flatten_update_list = [_flatten_model(update, layer_list=['cnn1', 'fc2'], ignore='layer') for update in model_updates]
            initial_round, tuning_round = self.args.vae_initial, self.args.vae_tuning
            vae = train_vae(None, build_dif_set([flatten_update_list[i] for i in selected_ids]), initial_round, device=torch.device(self.args.device), latent=self.args.vae_latent, hidden=self.args.vae_hidden)
            while len(selected_ids) < int(len(id_list) * 0.5):
                vae = train_vae(vae, build_dif_set([flatten_update_list[i] for i in selected_ids]), tuning_round, device=torch.device(self.args.device), latent=self.args.vae_latent, hidden=self.args.vae_hidden)
                vae.eval()
                with torch.no_grad():
                    rest_ids = [i for i in range(len(flatten_update_list)) if i not in selected_ids]
                    loss_ = []
                    for idx in rest_ids:
                        m_loss = 0.
                        loss_cnt = 0
                        for dif in obtain_dif([flatten_update_list[i] for i in selected_ids], flatten_update_list[idx]):
                            m_loss += vae.recon_prob(dif)
                            loss_cnt += 1
                        m_loss /= loss_cnt
                        loss_.append(m_loss)
                rank_ = np.argsort(loss_)
                selected_ids.extend(np.array(rest_ids)[rank_[:min(math.ceil(len(id_list) * 0.1), int(len(id_list) * self.args.snow_remain_cls) - len(selected_ids))]])
            benign_clients = [id_list[i] for i in selected_ids]
            self.cal_metrics(id_list, benign_clients)
            logging.info(f"all clients: {id_list}")
            logging.info(f"benign_clients: {benign_clients}")
            self.log_metrics()
            
            sm_updates, total_data = 0, 0
            for _id, update in agent_updates_dict.items():
                if _id in benign_clients:
                    n_agent_data = self.agent_data_sizes[_id]
                    sm_updates +=  n_agent_data * update
                    total_data += n_agent_data
            return  sm_updates / total_data

    def agg_FDCR(self, agent_updates_dict, global_model, local_fish_dict):
        local_model_vector = []
        id_list = []
        updates_state_dict_list = []
        global_model_state = parameters_to_vector([global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        for id, update in agent_updates_dict.items():
            id_list.append(id)
            local_model_vector.append(update + global_model_state)
            updates_state_dict_list.append(vector_to_name_param(update, copy.deepcopy(global_model.state_dict())))
            
        total_data = sum(self.agent_data_sizes.values())  # 计算总数据量
        freq = {client_id: data_size / total_data for client_id, data_size in self.agent_data_sizes.items()}

        local_name_param = []
        for update in updates_state_dict_list:
            name_param = []    
            for name, param in update.items():
                if "running_mean" not in name and "running_var" not in name and "num_batches_tracked" not in name:
                    name_param.append(param.view(-1))
            name_param = torch.cat(name_param)
            local_name_param.append(name_param)
        
        vectorize_nets_list = local_name_param
        nets_list = updates_state_dict_list
        prev_vectorize_net = torch.cat([p.view(-1) for p in global_model.parameters()]).detach()
        grad_list = []
        weight_list = []
        fish_list = []
        
        for query_index, _ in enumerate(local_name_param):
            grad_list.append((prev_vectorize_net - vectorize_nets_list[query_index]) / self.args.client_lr)
            query_fish_dict = local_fish_dict[id_list[query_index]]
            query_fish = torch.cat([p.view(-1) for p in query_fish_dict.values()]).detach()
            fish_list.append(query_fish)
            norm_fish = (query_fish - torch.min(query_fish)) / (torch.max(query_fish) - torch.min(query_fish))
            weight_list.append(norm_fish)

        weight_grad_list = []
        for query_index, _ in enumerate(local_name_param):
            query_grad = grad_list[query_index]
            query_weight = weight_list[query_index]
            weight_grad_list.append(torch.mul(query_grad, query_weight))

        weight_global_grad = torch.zeros_like(weight_grad_list[0])

        for weight_client_grad, client_freq in zip(weight_grad_list, freq):
            weight_global_grad += weight_client_grad * client_freq

        div_score = []
        for query_index, _ in enumerate(local_name_param):
            div_score.append(
                F.pairwise_distance(weight_grad_list[query_index].view(1, -1), weight_global_grad.view(1, -1), p=2))

        div_score = torch.tensor(div_score).view(-1, 1)
        fin = FINCH()
        fin.fit(div_score)

        if len(fin.partitions) == 0:
            benign_clients = id_list
            self.cal_metrics(id_list, benign_clients)
            logging.info(f"all clients: {id_list}")
            logging.info(f"benign_clients: {benign_clients}")
            self.log_metrics()
        else:
            select_partitions = (fin.partitions)['parition_0']
            evils_center = max(select_partitions['cluster_centers'])
            evils_center_idx = np.where(select_partitions['cluster_centers'] == evils_center)[0]
            evils_idx = select_partitions['cluster_core_indices'][int(evils_center_idx)]
            benign_idx = [i for i in range(len(local_name_param)) if i not in evils_idx]

            benign_clients = [id_list[i] for i in benign_idx]
            self.cal_metrics(id_list, benign_clients)
            logging.info(f"all clients: {id_list}")
            logging.info(f"benign_clients: {benign_clients}")
            self.log_metrics()
        
            for i in benign_idx:
                curr_net = nets_list[i]
                norm_weight = weight_list[i]
                index = 0
                for name, curr_param in curr_net.items():
                    prev_para = global_model.state_dict()[name].detach()
                    delta = (prev_para - curr_param.detach())
                    param_number = prev_para.numel()
                    param_size = prev_para.size()
                    
                    if "running_mean" not in name and "running_var" not in name and "num_batches_tracked" not in name :
                        weight_para = norm_weight[index:index + param_number].reshape(param_size).to(self.args.device)
                        weight_para = torch.nn.functional.sigmoid(weight_para) * 2
                        weight_delta = torch.mul(delta, weight_para)
                        index += param_number
                    else:
                        weight_delta = delta
                        
                    curr_param.data = (prev_para - weight_delta)
                nets_list[i] = curr_net
        
        global_update = copy.deepcopy(global_model.state_dict())
        total_data = 0
        for name, param in global_update.items():
            global_update[name] = torch.zeros_like(param)
        for id, update_name_param in zip(id_list, nets_list):
            if id in benign_clients:
                total_data += self.agent_data_sizes[id]
                for name, param in update_name_param.items():
                    global_update[name].data = global_update[name].data.float() + param.data * self.agent_data_sizes[id]
        for name, global_param in global_update.items():
            global_param.data /= total_data
        
        return parameters_to_vector([global_update[name] for name in global_update.keys()])
              
    def cal_metrics(self, id_list, chosen_agent):
        if self.rnd >= self.args.start_poison:
            for id in id_list:
                if id in chosen_agent and id >= self.args.num_corrupt:
                    self.true_negative += 1
                elif id in chosen_agent and id < self.args.num_corrupt:
                    self.false_negative += 1
                elif id not in chosen_agent and id >= self.args.num_corrupt:
                    self.false_positive += 1
                elif id not in chosen_agent and id < self.args.num_corrupt:
                    self.true_positive += 1
                else:
                    logging.info("error")
    
    def log_metrics(self):
        if self.rnd >= self.args.start_poison:
            total = self.true_positive + self.false_positive + self.true_negative + self.false_negative
            
            precision = self.true_positive / (self.true_positive + self.false_positive + 1e-10)  # 避免除零
            recall = self.true_positive / (self.true_positive + self.false_negative + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            fpr = self.false_positive / (self.false_positive + self.true_negative + 1e-10)
            fnr = self.false_negative / (self.true_positive + self.false_negative + 1e-10)
            accuracy = (self.true_positive + self.true_negative) / (total + 1e-10)

            logging.info(f"[Defense Metrics] "
                        f"Precision: {precision:.4f}, "
                        f"Recall: {recall:.4f}, "
                        f"F1: {f1:.4f}, "
                        f"FPR: {fpr:.4f}, "
                        f"FNR: {fnr:.4f}, "
                        f"Accuracy: {accuracy:.4f}")
            
            logging.info(f"[Raw Counts] "
                        f"TP: {self.true_positive}, FP: {self.false_positive}, "
                        f"TN: {self.true_negative}, FN: {self.false_negative}")



