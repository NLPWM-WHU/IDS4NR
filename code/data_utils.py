import torch
import random
import pandas as pd
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

#random.seed(2019)
#np.random.seed(2019)
def rbf_kernel(t_u, s_u, t_i, lamda=2):
    return np.exp(-abs(t_u-t_i)**2 / (2 * lamda * (s_u+1e-6)**2))

def cal_pointwise_weight(batch_u, batch_i, theta_u, sigma_u, theta_i):
    batch_theta_u, batch_sigma_u = theta_u[batch_u], sigma_u[batch_u]
    batch_theta_i = theta_i[batch_i]
    batch_pai_ui = rbf_kernel(batch_theta_u, batch_sigma_u, batch_theta_i)
    weight = (batch_pai_ui - min(batch_pai_ui)) / (max(batch_pai_ui) - min(batch_pai_ui))
    #weight = np.where(weight > 0, weight, 0)
    #weight = np.nan_to_num(weight, posinf=0, neginf=0)
    return weight

class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)

class UserItemRatingDataset_Weight(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor, target_tensor, weight_tensor):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor
        self.weight_tensor = weight_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index], self.weight_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)

def instance_a_train_loader(user_num, item_num, train_user_list, num_negatives, batch_size, cold_items=[None]):
        """instance train loader for one training epoch"""
        users, items, ratings = [], [], []
        for u, ilist in enumerate(train_user_list):
            #print(u)
            if len(ilist) == 0:
                continue
            for i in ilist:
                users.append(u)
                items.append(int(i))
                ratings.append(1)
                for _ in range(num_negatives):
                    neg_i = np.random.randint(item_num)
                    while (neg_i in ilist) or (neg_i in cold_items):    #负样本是没买过的非冷启动item
                        neg_i = np.random.randint(item_num)
                    users.append(u)
                    items.append(neg_i)
                    ratings.append(0)
                #neg_list = np.random.choice(list(set(range(item_num)) - ilist), num_negatives)
                #for neg_i in neg_list:
                #    users.append(u)
                #    items.append(neg_i)
                #    ratings.append(0) # negative samples get 0 rating

        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(np.array(users)),
                                        item_tensor=torch.LongTensor(np.array(items)),
                                        target_tensor=torch.FloatTensor(np.array(ratings)))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def instance_a_train_loader_weight(user_num, item_num, train_user_list, num_negatives, batch_size, theta_u, sigma_u, theta_i):
    """instance train loader for one training epoch"""
    users, items, ratings = [], [], []
    for u, ilist in enumerate(train_user_list):
        # print(u)
        if len(ilist) == 0:
            continue
        for i in ilist:
            users.append(u)
            items.append(int(i))
            ratings.append(1)
            for _ in range(num_negatives):
                neg_i = np.random.randint(item_num)
                while neg_i in ilist:
                    neg_i = np.random.randint(item_num)
                users.append(u)
                items.append(neg_i)
                ratings.append(0)
            # neg_list = np.random.choice(list(set(range(item_num)) - ilist), num_negatives)
            # for neg_i in neg_list:
            #    users.append(u)
            #    items.append(neg_i)
            #    ratings.append(0) # negative samples get 0 rating
    weights = cal_pointwise_weight(np.array(users), np.array(items), theta_u, sigma_u, theta_i)
    dataset = UserItemRatingDataset_Weight(user_tensor=torch.LongTensor(np.array(users)),
                                    item_tensor=torch.LongTensor(np.array(items)),
                                    target_tensor=torch.FloatTensor(np.array(ratings)),
                                    weight_tensor=torch.FloatTensor(weights))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class UserItemRatingDataset_pair(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor, negs_tensor):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.negs_tensor = negs_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.negs_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)

def instance_a_train_loader_pair(user_num, item_num, train_user_list, num_negatives, batch_size, cold_items=[None]):
        """instance train loader for one training epoch"""
        users, items, negs = [], [], []
        for u, ilist in enumerate(train_user_list):
            #print(u)

            if len(ilist) == 0:
                continue
            for i in ilist:
                # users.append(u)
                # items.append(int(i))
                # tmp_neg = []
                for _ in range(num_negatives):
                    neg_i = np.random.randint(item_num)
                    while (neg_i in ilist) or (neg_i in cold_items):
                        neg_i = np.random.randint(item_num)
                    users.append(u)
                    items.append(int(i))
                    negs.append([int(neg_i)])
                #negs.append(tmp_neg)

        dataset = UserItemRatingDataset_pair(user_tensor=torch.LongTensor(np.array(users)),
                                        item_tensor=torch.LongTensor(np.array(items)),
                                        negs_tensor=torch.LongTensor(np.array(negs)))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def instance_a_eval_loader(user_num, item_num, train_user_list, test_user_list, test_neg, batch_size):
    users, items, ratings = [], [], []
    for u, ilist in enumerate(test_user_list):
        if len(ilist) == 0:
            continue
        neg_list = np.random.choice(list(set(range(item_num)) - ilist - train_user_list[u]), test_neg-len(ilist))
        #test_list = np.append(ilist, neg_list)
        for i in ilist:
            users.append(u)
            items.append(i)
            ratings.append(1)
        for i in neg_list:
            users.append(u)
            items.append(i)
            ratings.append(0)

    dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(np.array(users)),
                                    item_tensor=torch.LongTensor(np.array(items)),
                                    target_tensor=torch.FloatTensor(np.array(ratings)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def instance_u_eval_loader_all(item_num, train_user_list, test_user_list, batch_size, u):
    users, items, ratings = [], [], []
    test_groundtruth = test_user_list[u]
    test_list = np.array(list(set(range(1, item_num)) - train_user_list[u]))
    for i in test_list:
        users.append(u)
        items.append(i)
        if i in test_groundtruth:
            ratings.append(1)
        else:
            ratings.append(0)

    dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(np.array(users)),
                                    item_tensor=torch.LongTensor(np.array(items)),
                                    target_tensor=torch.FloatTensor(np.array(ratings)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def instance_a_eval_loader(test_instance, batch_size):
    users, items, ratings = zip(*test_instance)
    dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(np.array(users)),
                                    item_tensor=torch.LongTensor(np.array(items)),
                                    target_tensor=torch.FloatTensor(np.array(ratings)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)