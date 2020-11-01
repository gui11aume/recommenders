#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F


pyro.enable_validation()
pyro.set_rng_seed(123)

# Min parameter value to prevent numeric instability.
EPSILON = 1e-3

SPARSITY = 1.

'''
https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/p305-li.pdf
'''

K   = 50      # Dimension of variables u, v and z.
XSZ = 20000   # Dimension of variable x (size of the vocabulary).


class Generator(nn.Module):
   def __init__(self):
      super().__init__()
      # Layers.
      self.A = nn.Linear(  K, 100)
      self.B = nn.Linear(100, 200)
      self.C = nn.Linear(200, XSZ) # Dirichlet parameter alpha.

   def forward(self, z):
      act = F.relu(self.A(z))
      act = F.relu(self.B(act))
      lmbd = F.softplus(self.C(act))
      # Parameters of a Dirichlet variable.
      return lmbd


class InfNet(nn.Module):
   def __init__(self, tied_generator=None):
      super().__init__()
      # Layers.
      self.C = nn.Linear(XSZ, 200)
      self.B = nn.Linear(200, 100)
      self.A = nn.Linear(100,   K) # Gaussian parameter mu.
      self.S = nn.Linear(100,   K) # Gaussian parameter sd.
      
      if tied_generator is not None:
         self.A.weight.data = tied_generator.A.weight.transpose(0,1)
         self.B.weight.data = tied_generator.B.weight.transpose(0,1)
         self.C.weight.data = tied_generator.C.weight.transpose(0,1)

   def forward(self, x):
      act = F.relu(self.C(x))
      act = F.relu(self.B(act))
      mu = self.A(act)
      sd = F.softplus(self.S(act))
      # Parameters of a Gaussian variable.
      return mu, sd


class CollaborativeVAE(nn.Module):
   def __init__(self):
      super().__init__()
      self.generator = Generator()
      self.infnet = InfNet(tied_generator=self.generator)

   def model(self, R, X, S):
      # Register generator for training.
      pyro.module('generator', self.generator)
      # There are M users and N items.
      M = R.shape[0]
      N = R.shape[1]
      with pyro.plate('items', N, subsample_size=16384) as indX:
         zero = X.new_zeros(16384, K)
         one = X.new_ones(16384, K)
         # Sample z, latent variable of item features.
         z = pyro.sample('z', dist.Normal(zero, one).to_event(1))
         # Generate item features (add epsilon to prevent underflow).
         lmbd = self.generator(z) * S[indX].unsqueeze(1) + EPSILON
         pyro.sample('X', dist.Poisson(lmbd).to_event(1), obs=X[indX])
         # Sample v-dagger, latent variable of collaborative profile.
         vd = pyro.sample('vd', dist.Normal(zero, one).to_event(1))
         #v = vd + z # Full latent representation of the items.
         v = vd # <======= DEBUG TEST XXX #
      with pyro.plate('users', M) as indR:
         zero = R.new_zeros(M, K)
         one = R.new_ones(M, K)
         # Sample u, latent variable of user profile.
         u = pyro.sample('u', dist.Normal(zero, one).to_event(1))
      with pyro.plate('mix1', M, dim=-2), pyro.plate('mix2', 16384, dim=-1):
         # Ratings are the dot products of user and item profiles.
         # The bias term is to push the average below 1%.
         prob = torch.sigmoid(torch.mm(u, v.T))
         pyro.sample('R', dist.Bernoulli(SPARSITY / .01 * prob), obs=R[indR][:,indX])


   def guide(self, R, X, S):
      # Register inference network for training.
      pyro.module('infnet', self.infnet)
      # There are M users and N items.
      M = R.shape[0]
      N = R.shape[1]
      # Variational parameters for u.
      u_loc = pyro.param('u_loc', R.new_zeros(M, K))
      u_scl = pyro.param('u_scl', R.new_ones(M, K),
            constraint=torch.distributions.constraints.positive)
      # Variational parameters for vd.
      vd_loc = pyro.param('vd_loc', X.new_zeros(N, K))
      vd_scl = pyro.param('vd_scl', X.new_ones(N, K),
            constraint=torch.distributions.constraints.positive)
      with pyro.plate('items', N, subsample_size=16384) as indX:
         pyro.sample('z', dist.Normal(*self.infnet(X[indX])).to_event(1))
         pyro.sample('vd', dist.Normal(vd_loc[indX], vd_scl[indX]).to_event(1))
      with pyro.plate('users', M) as indR:
         pyro.sample('u', dist.Normal(u_loc, u_scl).to_event(1))


def parse_CiteULike_data(bag_of_words_fname, ratings_fname):
   split = lambda s: [int(x) for x in s.split(':')]
   S = list()
   X = list()
   with open(bag_of_words_fname) as f:
      for line in f:
         x = torch.zeros(XSZ)
         items = line.split()
         pairs = [split(x) for x in items[1:]]
         scl = torch.tensor(float(items[0]))
         idx = torch.tensor([int(word) for word,count in pairs])
         cnt = torch.tensor([float(count) for word,count in pairs])
         # Use last word of the vocabulary for empty documents.
         if len(idx) == 0:
            scl = torch.tensor(1)
            idx = XSZ-1
            cnt = 1
         x[idx] = cnt
         S.append(scl)
         X.append(x)
   R = list()
   with open(ratings_fname) as f:
      for line in f:
         r = torch.zeros(len(X))
         items = line.split()
         idx = [int(x) for x in items[1:]]
         r[idx] = 1
         R.append(r)
   return torch.stack(R), torch.stack(X), torch.stack(S)


if __name__ == '__main__':
   # Data.
   R, X, S = parse_CiteULike_data('mult.dat', 'users.dat')
   R = R.to('cuda')
   X = X.to('cuda')
   S = S.to('cuda')
   SPARSITY = R.sum() / torch.prod(torch.tensor(R.shape))
   # Model.
   cvae = CollaborativeVAE()
   cvae.cuda()
   # Training.
   optimizer = torch.optim.Adam
   scheduler = pyro.optim.MultiStepLR({'optimizer': optimizer,
      'optim_args': {'lr': .001}, 'milestones': [250, 500]})
   svi = pyro.infer.SVI(cvae.model, cvae.guide, scheduler,
         loss=pyro.infer.Trace_ELBO())
   loss = 0
   for i in range(1400):
      loss += svi.step(R, X, S)
      if (i+1) % 10 == 0:
         print(i, loss)
         loss = 0.
   import pdb; pdb.set_trace()
   guide_trace = pyro.poutine.trace(cvae.guide).get_trace(R, X, S)
   model_trace = pyro.poutine.trace(pyro.poutine.replay(cvae.model, guide_trace)).get_trace(R, X, S)
