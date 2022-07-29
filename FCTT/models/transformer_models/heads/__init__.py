from ..utils import GELU

import torch
import torch.nn as nn
import torch.nn.functional as F


class BertLinearPredictionHead(nn.Module):
    def __init__(self, args, input_size=None):
        super().__init__()
        self.vocab_size = args.num_items + 1
        hidden = input_size if input_size is not None else args.hidden_units
        if args.head_use_ln:
            self.out = nn.Sequential(
                nn.Linear(hidden, hidden),
                GELU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, self.vocab_size)
            )
        else:
            self.out = nn.Sequential(
                nn.Linear(hidden, hidden),
                GELU(),
                nn.Linear(hidden, self.vocab_size)
            )

    def forward(self, x, candidates=None):
        x = self.out(x)  # B x V or M x V
        if candidates is not None:
            x = x.gather(1, candidates)  # B x C or M x C
        return x


class BertDotProductPredictionHead(nn.Module):
    def __init__(self, args, item_embeddings,user_embeddings, input_size=None):
        super().__init__()
        self.item_embeddings = item_embeddings
        self.user_embeddings = user_embeddings
        hidden = args.hidden_units
        if input_size is None:
            input_size = hidden
        self.vocab_size = args.num_items + 1
        if args.head_use_ln:
            self.out = nn.Sequential(
                nn.Linear(input_size, hidden),
                GELU(),
                nn.LayerNorm(hidden)
            )
        else:
            self.out = nn.Sequential(
                nn.Linear(input_size, hidden),
                GELU(),
            )
        self.bias = nn.Parameter(torch.zeros(1, self.vocab_size))
        self.linear_t = nn.Linear(hidden, int(hidden/2), bias=False)
        self.a = torch.nn.Parameter(torch.FloatTensor(hidden))

    def forward(self, x, uid,last_hidden,candidates=None):
        x = self.out(x)  # B x H or M x H
        if candidates is not None:  # x : B x H
            emb_itemss = self.item_embeddings(candidates)# B x C x H

            qt = self.linear_t(last_hidden)
            beta = F.softmax(emb_itemss @ qt.transpose(1, 2), -1)
            target = beta @ last_hidden

            emb_userss = self.user_embeddings(uid)
            emb_userss = emb_userss.unsqueeze(1).repeat(1,candidates.size()[1],1)
            item_cat_user = torch.cat((emb_itemss,emb_userss),-1)

            s = self.a*x.unsqueeze(1)+(1-self.a)*target
            logits = (s * item_cat_user).sum(-1)

            # logits1 = (x.unsqueeze(1) * item_cat_user).sum(-1)  # B x C
            # logits2 = (target * item_cat_user).sum(-1)
            # logits = self.a * logits1 + (1 - self.a) * logits2

            bias = self.bias.expand(logits.size(0), -1).gather(1, candidates)  # B x C
            logits += bias
        else:  # x : M x H
            uuid = list()    #1516mb
            m_last = -1
            m_ch=-1
            for i in range(len(uid)):
                m = uid[i]
                if m==m_last:
                    uuid.append(m_ch)
                else:
                    m_ch+=1
                    m_last = m
                    uuid.append(m_ch)

            emb_itemss = self.item_embeddings.weight[:self.vocab_size]#3417*64
            qt = self.linear_t(last_hidden)#1516  5*200*64
            beta = F.softmax(emb_itemss @ qt.transpose(1, 2), -1)#1530    5*3417*200
            target = beta @ last_hidden#1530   5*3417*128
            target = target[uuid]#1826   177*3417*128

            emb_userss = self.user_embeddings.weight
            user_cat = emb_userss[uid]
            user_cat = torch.unsqueeze(user_cat,1).repeat(1,self.vocab_size,1)#1974
            emb_itemss = torch.unsqueeze(emb_itemss,0).repeat(len(x),1,1)#2122
            item_cat_user = torch.cat((emb_itemss,user_cat),-1)#2418
            s = self.a * x.unsqueeze(1) + (1 - self.a) * target#3010
            logits = (s * item_cat_user).sum(-1)
            logits += self.bias

            # logits1  = (x.unsqueeze(1) * item_cat_user).sum(-1)
            # logits2 = (target * item_cat_user).sum(-1)
            # logits = self.a*logits1+(1-self.a)*logits2

            # emb = self.item_embeddings.weight[:self.vocab_size]  # V x H
            # emb= torch.cat((emb,emb),-1)
            # logits = torch.matmul(x, emb.transpose(0, 1))  # M x V
            # logits += self.bias

        return logits


            # emb_userss = self.user_embeddings.weight[:self.vocab_size]
            # embs = torch.cat((emb_itemss,emb_userss),1)
            # embs = torch.unsqueeze(embs, 0).repeat(len(x), 1, 1)
            # logits = (x.unsqueeze(1) * embs).sum(-1)
            # logits += self.bias
        # return logits

            # V x H
        #     emb_userss = self.user_embeddings.weight
        #     user_cat = emb_userss[uid]
        #     user_cat = torch.unsqueeze(user_cat,1).repeat(1,self.vocab_size,1)
        #     emb_itemss = torch.unsqueeze(emb_itemss,0).repeat(len(x),1,1)
        #     item_cat_user = torch.cat((emb_itemss,user_cat),-1)
        #     logits = torch.matmul(x.unsqueeze(1), item_cat_user.transpose(1, 2)).squeeze(1)
        #     logits += self.bias
        # return logits
            # logits = (x.unsqueeze(1) * item_cat_user).sum(-1)
            # print(logitss==logits)
            # logits = torch.matmul(x, item_cat_user.transpose(0, 1))  # M x V

            # del emb_userss,emb_itemss,user_cat,item_cat_user
            # torch.cuda.empty_cache()



class BertL2PredictionHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden = args.hidden_units
        self.vocab_size = args.num_items + 1
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden),
            GELU(),
            nn.LayerNorm(hidden)
        )
        self.bias = nn.Parameter(torch.zeros(1, 1, self.vocab_size))

    def forward(self, x, token_embeddings):
        # x = self.out(x).unsqueeze(2)  # B x T x 1 x H
        x = x.unsqueeze(2)  # B x T x 1 x H
        emb = token_embeddings.weight[:self.vocab_size].unsqueeze(0).unsqueeze(1)  # 1 x 1 x V x H
        diff = x - emb  # B x T x V x H
        dist = (diff ** 2).sum(-1).sqrt()  # B x T x V
        return (-dist) + self.bias


class BertDiscriminatorHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden = args.hidden_units
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden),
            GELU(),
            nn.LayerNorm(hidden)
        )
        self.w = nn.Parameter(torch.zeros(1, 1, hidden))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x : B x T x H
        x = self.out(x)
        x = (x * self.w).sum(-1)  # B x T
        return self.sigmoid(x)
