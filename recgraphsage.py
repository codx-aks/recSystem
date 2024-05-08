import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph

# Load Pytorch as backend
dgl.load_backend('pytorch')

import numpy as np
import pandas as pd
from scipy import stats
from scipy import sparse as spsp

from movielens import MovieLens
data = MovieLens('.')

ratings = data.ratings
user_id = np.array(ratings['user_idx'])
movie_id = np.array(ratings['movie_idx'])
user_movie_spm = spsp.coo_matrix((np.ones((len(user_id),)), (user_id, movie_id)))
num_users, num_movies = user_movie_spm.shape
print('#user-movie iterations:', len(movie_id))
print('#users:', num_users)
print('#movies:', num_movies)

ratings_train = ratings[~(ratings['valid_mask'] | ratings['test_mask'])]
user_latest_item_indices = (
        ratings_train.groupby('user_id')['timestamp'].transform(pd.Series.max) ==
        ratings_train['timestamp'])
user_latest_item = ratings_train[user_latest_item_indices]
user_latest_item = dict(
        zip(user_latest_item['user_idx'].values, user_latest_item['movie_idx'].values))

user_id = np.array(ratings_train['user_idx'])
movie_id = np.array(ratings_train['movie_idx'])
user_movie_spm = spsp.coo_matrix((np.ones((len(user_id),)), (user_id, movie_id)))
assert num_users == user_movie_spm.shape[0]
assert num_movies == user_movie_spm.shape[1]
train_size = len(user_id)
print('#training size:', train_size)

# The validation and testing dataset
users_valid = ratings[ratings['valid_mask']]['user_idx'].values
movies_valid = ratings[ratings['valid_mask']]['movie_idx'].values
users_test = ratings[ratings['test_mask']]['user_idx'].values
movies_test = ratings[ratings['test_mask']]['movie_idx'].values
valid_size = len(users_valid)
test_size = len(users_test)

from SLIM import SLIM, SLIMatrix
model = SLIM()
params = {'algo': 'cd', 'nthreads': 2, 'l1r': 2.0, 'l2r': 1.0}
trainmat = SLIMatrix(user_movie_spm.tocsr())
model.train(params, trainmat)

movie_spm = model.to_csr()
print('#edges:', movie_spm.nnz)
print('most similar:', np.max(movie_spm.data))
print('most unsimilar:', np.min(movie_spm.data))

g = dgl.DGLGraph(movie_spm, readonly=True)
g.edata['similarity'] = torch.tensor(movie_spm.data, dtype=torch.float32)

year = np.expand_dims(data.movie_data['year'], axis=1)
genre = data.movie_data['genre']
title = data.movie_data['title']
features = torch.tensor(np.concatenate((genre, title), axis=1), dtype=torch.float32)
print('#features:', features.shape[1])
in_feats = features.shape[1]

from sage_conv import SAGEConv

class GraphSAGEModel(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_dim,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGEModel, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type,
                                    feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type,
                                        feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(SAGEConv(n_hidden, out_dim, aggregator_type,
                                    feat_drop=dropout, activation=None))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h, g.edata['similarity'])
        return h


class EncodeLayer(nn.Module):
    def __init__(self, in_feats, num_hidden):
        super(EncodeLayer, self).__init__()
        self.proj = nn.Linear(in_feats, num_hidden)

    def forward(self, feats):
        return self.proj(feats)


beta = 0
gamma = 0


class FISM(nn.Module):
    def __init__(self, user_movie_spm, gconv_p, gconv_q, in_feats, num_hidden):
        super(FISM, self).__init__()
        self.encode = EncodeLayer(in_feats, num_hidden)
        self.num_users = user_movie_spm.shape[0]
        self.num_movies = user_movie_spm.shape[1]
        self.b_u = nn.Parameter(torch.zeros(num_users))
        self.b_i = nn.Parameter(torch.zeros(num_movies))
        self.user_deg = torch.tensor(user_movie_spm.dot(np.ones(num_movies)))
        values = user_movie_spm.data
        indices = np.vstack((user_movie_spm.row, user_movie_spm.col))
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(values)
        self.user_item_spm = torch.sparse_coo_tensor(indices, values, user_movie_spm.shape)
        self.users = user_movie_spm.row
        self.movies = user_movie_spm.col
        self.ratings = user_movie_spm.data
        self.gconv_p = gconv_p
        self.gconv_q = gconv_q

    def _est_rating(self, P, Q, user_idx, item_idx):
        bu = self.b_u[user_idx]
        bi = self.b_i[item_idx]
        user_emb = torch.sparse.mm(self.user_item_spm, P)
        user_emb = user_emb[user_idx] / torch.unsqueeze(self.user_deg[user_idx], 1)
        tmp = torch.mul(user_emb, Q[item_idx])
        r_ui = bu + bi + torch.sum(tmp, 1)
        return r_ui

    def est_rating(self, g, features, user_idx, item_idx, neg_item_idx):
        h = self.encode(features)
        P = self.gconv_p(g, h)
        Q = self.gconv_q(g, h)
        r = self._est_rating(P, Q, user_idx, item_idx)
        neg_sample_size = len(neg_item_idx) / len(user_idx)
        neg_r = self._est_rating(P, Q, np.repeat(user_idx, neg_sample_size), neg_item_idx)
        return torch.unsqueeze(r, 1), neg_r.reshape((-1, int(neg_sample_size)))

    def loss(self, P, Q, r_ui, neg_r_ui):
        diff = 1 - (r_ui - neg_r_ui)
        return torch.sum(torch.mul(diff, diff) / 2) \
            + beta / 2 * torch.sum(torch.mul(P, P) + torch.mul(Q, Q)) \
            + gamma / 2 * (torch.sum(torch.mul(self.b_u, self.b_u)) + torch.sum(torch.mul(self.b_i, self.b_i)))

    def forward(self, g, features, neg_sample_size):
        h = self.encode(features)
        P = self.gconv_p(g, h)
        Q = self.gconv_q(g, h)
        tot = len(self.users)
        pos_idx = np.random.choice(tot, int(tot / 10))
        user_idx = self.users[pos_idx]
        item_idx = self.movies[pos_idx]
        neg_item_idx = np.random.choice(self.num_movies, len(pos_idx) * neg_sample_size)
        r_ui = self._est_rating(P, Q, user_idx, item_idx)
        neg_r_ui = self._est_rating(P, Q, np.repeat(user_idx, neg_sample_size), neg_item_idx)
        r_ui = torch.unsqueeze(r_ui, 1)
        neg_r_ui = neg_r_ui.reshape((-1, int(neg_sample_size)))
        return self.loss(P, Q, r_ui, neg_r_ui)

def RecEvaluate(model, g, features, users_eval, movies_eval, neg_sample_size):
    model.eval()
    with torch.no_grad():
        neg_movies_eval = data.neg_valid[users_eval].flatten()
        r, neg_r = model.est_rating(g, features, users_eval, movies_eval, neg_movies_eval)
        hits10 = (torch.sum(neg_r > r, 1) <= 10).numpy()
        print('HITS@10:{:.4f}'.format(np.mean(hits10)))
        return np.mean(hits10)


# Model hyperparameters
n_hidden = 16
n_layers = 1
dropout = 0.5
aggregator_type = 'sum'

# create GraphSAGE model
gconv_p = GraphSAGEModel(n_hidden,
                         n_hidden,
                         n_hidden,
                         n_layers,
                         F.relu,
                         dropout,
                         aggregator_type)

gconv_q = GraphSAGEModel(n_hidden,
                         n_hidden,
                         n_hidden,
                         n_layers,
                         F.relu,
                         dropout,
                         aggregator_type)

model = FISM(user_movie_spm, gconv_p, gconv_q, in_feats, n_hidden)

# Training hyperparameters
weight_decay = 5e-4
n_epochs = 100
lr = 1e-3
neg_sample_size = 20

# use optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# initialize graph
dur = []
prev_acc = 0
for epoch in range(n_epochs):
    model.train()
    loss = model(g, features, neg_sample_size)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Epoch {:05d} | Loss {:.4f}".format(epoch, loss.item()))

    acc = RecEvaluate(model, g, features, users_valid, movies_valid, neg_sample_size)

print()
# Let's save the trained node embeddings.
RecEvaluate(model, g, features, users_test, movies_test, neg_sample_size)