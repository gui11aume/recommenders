#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F


pyro.enable_validation()
pyro.set_rng_seed(123)

'''
https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/p305-li.pdf
'''

K   = 50     # Dimension of variables u, v and z.
XSZ = 20001  # Dimension of variable x (size of the vocabulary).


class InfNet(nn.Module):
   def __init__(self, tied_generator=None):
      super().__init__()
      # Layers.
      self.L1 = nn.Linear(XSZ, 200)
      self.L2 = nn.Linear(200, 100)
      self.L3 = nn.Linear(100,   K)

   def forward(self, x):
      act = F.relu(self.L1(x))
      act = F.relu(self.L2(act))
      return self.L3(act)


class VariationalAutoencoder(nn.Module):
   def __init__(self):
      super().__init__()
      self.generator = Generator()

   def model(self, X=None):
      # Register generator network for training.
      pyro.module('generator', self.generator)
      with pyro.plate('items', X.shape[0]):
         zero = X.new_zeros(X.shape[0], K)
         one = X.new_ones(X.shape[0], K)
         # Sample z, latent variable for item features.
         z = pyro.sample('z', dist.Normal(zero, one).to_event(1))
         # Generate item features.
         param = self.generator(z)
         pyro.sample('X', dist.Bernoulli(param).to_event(1), obs=X)

   def guide(self, X=None):
      with pyro.plate('items', X.shape[0]):
         param = self.infnet(X)
         pyro.sample('z', dist.Normal(*param).to_event(1))


class CollaborativeVariationalAutoencoder(nn.Module):
   def __init__(self):
      super().__init__()
      self.infnet = InfNet()

   def model(self, R, X):
      # Register inference network for training.
      pyro.module('infnet', self.infnet)
      # There are M users and N items.
      M, N = R.shape[:2]
      with pyro.plate('items', N):
         zero = R.new_zeros(N, K)
         one = R.new_ones(N, K)
         # Sample v-dagger, latent variable of collaborative profile.
         vd = pyro.sample('vd', dist.Normal(zero, one).to_event(1))
         v = vd + self.infnet(X)
      with pyro.plate('users', M): 
         zero = R.new_zeros(M, K)
         one = R.new_ones(M, K)
         # Sample u, latent variable of user profile.
         u = pyro.sample('u', dist.Normal(zero, one).to_event(1))
      with pyro.plate('mix1', M, dim=-2), pyro.plate('mix2', N, dim=-1):
         # Ratings are the dot products of user and item profiles.
         prob = torch.sigmoid(torch.mm(u, v.T))
         # The target is 20x more items than users have rated.
         pyro.sample('R', dist.Bernoulli(prob / 20), obs=R)

   def guide(self, R, X):
      # There are M users and N items.
      M, N = R.shape[:2]
      # Variational parameters for u.
      u_loc = pyro.param('u_loc', R.new_zeros(M, K))
      u_scl = pyro.param('u_scl', R.new_ones(M, K),
            constraint=torch.distributions.constraints.positive)
      # Variational parameters for vd.
      vd_loc = pyro.param('vd_loc', R.new_zeros(N, K))
      vd_scl = pyro.param('vd_scl', R.new_ones(N, K),
            constraint=torch.distributions.constraints.positive)
      with pyro.plate('items', N):
         pyro.sample('vd', dist.Normal(vd_loc, vd_scl).to_event(1))
      with pyro.plate('users', M):
         pyro.sample('u', dist.Normal(u_loc, u_scl).to_event(1))


class WordData:
   def __init__(self, bag_of_words_fname):
      X = list()
      with open(bag_of_words_fname) as f:
         for line in f:
            items = line.split()
            idx = torch.tensor([int(x.split(':')[0]) for x in items[1:]])
            # Use last word of the vocabulary for empty documents.
            if len(idx) == 0:
               idx = XSZ-1
            # Use 1 for presence, 0 for absence (ignore counts).
            x = torch.zeros(XSZ)
            x[idx] = 1
            X.append(x)
      self.X = torch.stack(X)

   def batches(self, batch_size=None, randomize=True):
      batch_size = batch_size or self.X.shape[0]
      idx = np.arange(len(self.X))
      if randomize:
         np.random.shuffle(idx)
      for ix in np.array_split(idx, len(idx) // batch_size): 
         yield self.X[ix,...]


class RatingData:
   def __init__(self, rating_fname, nitems):
      R = list()
      with open(rating_fname) as f:
         for line in f:
            items = line.split()
            idx = [int(x) for x in items[1:]]
            # Use 1 for yes, 0 for no or unaware.
            r = torch.zeros(nitems)
            r[idx] = 1
            R.append(r)
      self.R = torch.stack(R)

   def batches(self):
      yield self.R


if __name__ == '__main__':
   # Train on GPU if possible.
   device = 'cuda' if torch.cuda.is_available() else 'cpu'

   X = WordData('mult.dat').X.to(device)
   R = RatingData('users.dat', X.shape[0]).R.to(device)
   cvae = CollaborativeVariationalAutoencoder().to(device)

   optimizer = pyro.optim.Adam({'lr': 0.01})
   svi = pyro.infer.SVI(cvae.model, cvae.guide, optimizer,
         loss=pyro.infer.JitTrace_ELBO())

   loss = 0.
   for epoch in range(5500):
      loss += svi.step(R, X)
      if (epoch+1) % 100 == 0:
         print('epoch {0}, loss: {1}'.format(epoch+1, loss / 100))
         loss = 0.

   v = pyro.param('vd_loc') + cvae.infnet(X)
   u = pyro.param('u_loc')
   prob = torch.sigmoid(torch.mm(u, v.T))
