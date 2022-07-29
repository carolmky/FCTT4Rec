from .negative_samplers import negative_sampler_factory

import torch.utils.data as data_utils

from abc import *
import random
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time
from FCTT.utils import *


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        self.sampler_rng = random.Random(seed)  # share seed for now... (doesn't really matter)
        save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        self.dataset = dataset
        self.user2dict = dataset['user2dict']
        self.train_targets = dataset['train_targets']
        self.validation_targets = dataset['validation_targets']
        self.test_targets = dataset['test_targets']
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)

        train_items=dict()
        for tups in self.train_targets:
            user = tups[0]
            item_range = tups[1]
            items = self.user2dict[user]['items'][:item_range]
            train_items[user-1] = items              #用户索引从0开始
        trainUniqueUsers, trainItem, trainUser = [], [], []
        self.traindataSize = 0
        for a, l in train_items.items():
            if len(l) > 0:
                items = l
                uid = a
                trainUniqueUsers.append(uid)
                trainUser.extend([uid] * len(items))
                trainItem.extend(items)
                # self.m_item = max(self.m_item, max(items))
                # self.n_user = max(self.n_user, uid)
                self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        self.Graph = None
        # (users,items), bipartite graph

        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.user_count, self.item_count+1))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        self.local_export_root, remote_export_root, communicator = setup_train(args, MACHINE_IS_HOST)
        self.Graph = self.getSparseGraph()
        self.dataset['Graph'] = self.Graph

        # dynamically determine # of users/items
        # need to create Dataloader before anything else
        args.num_users = self.user_count
        args.num_items = self.item_count
        args.num_ratings = dataset['num_ratings']
        args.num_days = dataset['num_days']

        code = args.train_negative_sampler_code
        train_negative_sampler = negative_sampler_factory(code, self.user2dict,
                                                          self.user_count, self.item_count,
                                                          args.train_negative_sample_size,
                                                          args.train_negative_sampling_seed,
                                                          save_folder)
        code = args.test_negative_sampler_code
        test_negative_sampler = negative_sampler_factory(code, self.user2dict,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         save_folder)

        self.train_negative_samples = train_negative_sampler.get_negative_samples()
        self.test_negative_samples = test_negative_sampler.get_negative_samples()

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    def getSparseGraph(self):

        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz('./Data/graph' + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.user_count + self.item_count+1, self.user_count + self.item_count+1), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.user_count, self.user_count:] = R
                adj_mat[self.user_count:, :self.user_count] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz('./Data/graph' + '/s_pre_adj_mat.npz', norm_adj)


            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to('cuda')
            print("don't split the matrix")
        return self.Graph

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def get_pytorch_dataloaders(self):
        train_loaders = self._get_dataloaders('train')
        val_loaders = self._get_dataloaders('val')
        test_loaders = self._get_dataloaders('test')
        return train_loaders, val_loaders, test_loaders

    def _get_dataloaders(self, mode):
        batch_size = {'train':self.args.train_batch_size,
                      'val':self.args.val_batch_size,
                      'test':self.args.test_batch_size}[mode]

        dataset = self._get_dataset(mode)
        # shuffle = True if mode == 'train' else False
        # sampler = None
        shuffle = False
        sampler = CustomRandomSampler(len(dataset), self.sampler_rng) if mode == 'train' else None
        # drop_last = True if mode == 'train' else False
        drop_last = False
        dataloader = data_utils.DataLoader(dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           sampler=sampler,
                                           pin_memory=True,
                                           num_workers=self.args.num_workers,
                                           drop_last=drop_last)
        return dataloader

    @abstractmethod
    def _get_dataset(self, mode):
        pass


class CustomRandomSampler(data_utils.Sampler):
    def __init__(self, n, rng):
        super().__init__(data_source=[]) # dummy
        self.n = n
        self.rng = rng

    def __len__(self):
        return self.n

    def __iter__(self):
        indices = list(range(self.n))
        self.rng.shuffle(indices)
# 打乱的效果会更好一点吗
        return iter(indices)

    def get_rng_state(self):
        return self.rng.getstate()

    def set_rng_state(self, state):
        return self.rng.setstate(state)
