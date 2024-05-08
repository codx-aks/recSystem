import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from sage_conv import SAGEConv
import numpy as np
import pandas as pd
from scipy import sparse as spsp
from sklearn.preprocessing import OneHotEncoder

val_size=30
test_size=1000

# Load ratings and events data
ratings = pd.read_csv('ratings.csv')
events = pd.read_csv('events.csv', usecols=['eventId', 'type'])

# Convert event types to one-hot encoding
encoder = OneHotEncoder()
event_types = encoder.fit_transform(events[['type']]).toarray()

# Prepare user-event sparse matrix
user_ids = ratings['userId'].values
event_ids = ratings['eventId'].values
user_event_spm = spsp.coo_matrix((np.ones_like(user_ids), (user_ids, event_ids)))
num_users, num_events = user_event_spm.shape
train_size = len(user_ids)
print('#training size:', train_size)

# Convert to DGL graph
g = dgl.DGLGraph(user_event_spm, readonly=True)

# Define GraphSAGE model for collaborative filtering
class GraphSAGEModel(nn.Module):
    def __init__(self, in_feats, n_hidden, out_dim):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_feats, n_hidden, 'mean')
        self.conv2 = SAGEConv(n_hidden, out_dim, 'mean')

    def forward(self, g, features):
        h = self.conv1(g, features)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

# Set model hyperparameters
in_feats = num_events + event_types.shape[1]  # Include one-hot encoded event types
n_hidden = 16
out_dim = 1

# Create feature tensor
features = torch.tensor(np.hstack((event_types, np.zeros((num_events, num_users)))), dtype=torch.float32)

# Create masks for training, validation, and testing
train_mask = torch.arange(train_size)
val_mask = torch.arange(train_size, train_size + val_size)
test_mask = torch.arange(train_size + val_size, train_size + val_size + test_size)

# Create labels (ratings) tensor
labels = torch.tensor(ratings['rating'].values, dtype=torch.float32)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the model
model = GraphSAGEModel(in_feats, n_hidden, out_dim).to(device)

# Training hyperparameters
weight_decay = 5e-4
n_epochs = 100
lr = 1e-3

# Use optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Training loop
for epoch in range(n_epochs):
    model.train()
    logits = model(g.to(device), features.to(device))
    loss = F.mse_loss(logits[train_mask.to(device)], labels[train_mask.to(device)])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Epoch {:05d} | Loss {:.4f}".format(epoch, loss.item()))

# Evaluate on validation set
model.eval()
with torch.no_grad():
    logits = model(g.to(device), features.to(device))
    val_loss = F.mse_loss(logits[val_mask.to(device)], labels[val_mask.to(device)])
    print("Validation loss {:.4f}".format(val_loss.item()))

# Evaluate on test set
logits = model(g.to(device), features.to(device))
test_loss = F.mse_loss(logits[test_mask.to(device)], labels[test_mask.to(device)])
print("Test loss {:.4f}".format(test_loss.item()))
