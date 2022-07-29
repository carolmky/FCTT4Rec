from FCTT.models.base import BaseModel

import torch.nn as nn
import torch

from abc import *


class BertBaseModel(BaseModel, metaclass=ABCMeta):
    def __init__(self, args):
        super().__init__(args)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, d,batch_idx,is_gcn):
        if is_gcn:
            self.emb_user,self.emb_item = self.gcnn()
            return self.emb_user,self.emb_item
        else:
            logits, info = self.get_logits(d,batch_idx,self.emb_user,self.emb_item)
            # torch.cuda.empty_cache()
            ret = {'logits':logits, 'info':info}
            if self.training:
                labels = d['labels']
                users= d['users']
                loss, loss_cnt = self.get_loss(d, logits, labels)
                ret['loss'] = loss
                ret['loss_cnt'] = loss_cnt
            else:
                # get scores (B x V) for validation
                last_logits = logits[:, -1, :]  # B x H
                uid = d['users'].squeeze(1)-1
                ret['scores'] = self.get_scores(d, last_logits,uid,logits)  # B x C
            # torch.cuda.empty_cache()
            return ret

    @abstractmethod
    def get_logits(self, d):
        pass

    @abstractmethod
    def get_scores(self, d, logits):  # logits : B x H or M x H, returns B x C or M x V
        pass

    # def get_loss(self, d, logits, labels):
    #     _logits = logits.view(-1, logits.size(-1))  # BT x H
    #     _labels = labels.view(-1)  # BT
    #
    #     valid = _labels > 0
    #     loss_cnt = valid.sum()  # = M
    #     valid_index = valid.nonzero().squeeze()  # M
    #
    #     valid_logits = _logits[valid_index]  # M x H
    #     valid_scores = self.get_scores(d, valid_logits)  # M x V
    #     valid_labels = _labels[valid_index]  # M
    #
    #     loss = self.ce(valid_scores, valid_labels)
    #     loss, loss_cnt = loss.unsqueeze(0), loss_cnt.unsqueeze(0)
    #     return loss, loss_cnt

    def get_loss(self, d, logits, labels):
        users = d['users']
        valid = labels > 0
        loss_cnt = valid.sum()  # = M
        valid_index = valid.nonzero().squeeze()  # M

        for i in range(len(valid_index)):
            idx=valid_index[i]
            iid = idx[1]
            if i==0:
                uid = users[idx[0]]-1
                valid_logits= torch.unsqueeze(logits[idx[0]][idx[1]],0)
            else:
                uidd = users[idx[0]]-1
                uid=torch.cat((uid,uidd),0)
                vv=torch.unsqueeze(logits[idx[0]][idx[1]],0)
                valid_logits=torch.cat((valid_logits,vv),0)

        valid_scores = self.get_scores(d, valid_logits,uid,logits)  # M x V
        torch.cuda.empty_cache()
        for i in range(len(valid_index)):
            a = valid_index[i][0]
            b = valid_index[i][1]
            if i ==0:
                valid_labels = torch.unsqueeze(labels[a][b],0)
            else:
                vvv=torch.unsqueeze(labels[a][b],0)
                valid_labels = torch.cat((valid_labels,vvv),0)
        # valid_labels = labels[valid_index]  # M

        loss = self.ce(valid_scores, valid_labels)
        loss, loss_cnt = loss.unsqueeze(0), loss_cnt.unsqueeze(0)
        return loss, loss_cnt

    def gcnn(self):
        users_emb = self.token_embedding.embedding_user.weight
        items_emb = self.token_embedding.embedding_item.weight[:-1, :]
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.args.num_users, self.args.num_items +1])
        items = torch.cat((items, items_emb[-1:,:]), 0)
        return users,items

