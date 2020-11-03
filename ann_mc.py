"""
ANN Monte Carlo algorithm, implemented starting from PPO-Clip example.
"""
import time

import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam



def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]



class Buffer:
    """
    A buffer for storing trajectories experienced by an agent interacting
    with the environment.

    For categoricals, action is a scalar (act_dim = None),
    while fbn_dim = the number of possible actions
    """

    def __init__(self, obs_dim, fbn_dim, size, gamma=0.99):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32) # Observation (state)
        # the Forbidden mask -- set to True where actions are disallowed at that moment
        self.fbn_buf = np.zeros(combined_shape(size, fbn_dim), dtype=np.bool8)
        self.act_buf = np.zeros(size, dtype=np.int64) # Action
        self.rew_buf = np.zeros(size, dtype=np.float32) # Reward (one-step)
        self.ret_buf = np.zeros(size, dtype=np.float32) # Return (discounted cumulative future rewards)
        self.gamma = gamma
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, fbn, act, rew):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.fbn_buf[self.ptr] = fbn
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute the rewards-to-go for each state.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def reset(self):
        self.ptr, self.path_start_idx = 0, 0

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        # assert self.ptr == self.max_size    # buffer has to be full before you can get
        assert 1 <= self.ptr < self.max_size # we will return a slice
        used = slice(0, self.ptr)
        # self.ptr, self.path_start_idx = 0, 0
        self.reset()
        data = dict(
            size=used.stop,
            obs=torch.as_tensor(self.obs_buf[used, ...], dtype=torch.float32),
            fbn=torch.as_tensor(self.fbn_buf[used, ...], dtype=torch.bool),
            act=torch.as_tensor(self.act_buf[used, ...], dtype=torch.int64),
            ret=torch.as_tensor(self.ret_buf[used],      dtype=torch.float32),
            )
        return data



class MCAlgo:
    def __init__(self, model, lr=1e-3):

        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        # setup_pytorch_for_mpi()

        # Random seed
        # seed += 10000 * proc_id()
        # torch.manual_seed(seed)
        # np.random.seed(seed)

        self.model = model

        # # Count variables
        # var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
        # logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

        # # Set up experience buffer
        # local_steps_per_epoch = int(steps_per_epoch / num_procs())
        # buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

        # Set up optimizers for policy and value function
        self.optimizer = Adam(self.model.parameters(), lr=lr)

        # # Set up model saving
        # logger.setup_pytorch_saver(ac)


    def update(self, buffer):

        # Set up function for computing loss
        def compute_loss(data):
            obs, act, ret, size = data['obs'], data['act'], data['ret'], data['size']
            idx = torch.arange(size)
            q = self.model(obs)[idx, act]
            # Cross entropy with sigmoid output, MSE with linear output (?)
            # https://susanqq.github.io/tmp_post/2017-09-05-crossentropyvsmes/
            # Except I think we actually want "multi-label classification" (roughly)
            # https://sebastianraschka.com/faq/docs/pytorch-crossentropy.html

            # MSE on linear outputs doesn't work at all -- no progress or convergence
            # MSE on sigmoid outputs works BEST so far, though theory says it shouldn't work (well)
            # binary_cross_entropy_with_logits() on linear outputs works, in accordance with theory
            # logsigmoid + kl_div sounded plausible but doesn't work that well

            # loss = F.mse_loss(q, ret)
            loss = F.mse_loss(torch.sigmoid(q), ret)
            # loss = F.binary_cross_entropy_with_logits(q, ret)
            # loss = F.kl_div(F.logsigmoid(q), ret)
            return loss

        data = buffer.get()

        for i in range(1):
            self.optimizer.zero_grad()
            loss = compute_loss(data)
            loss.backward()
            # mpi_avg_grads(ac.v)    # average grads across MPI processes
            self.optimizer.step()
