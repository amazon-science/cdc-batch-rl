from __future__ import  print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from misc.utils import get_action_info

def atanh(x):
    '''
        aratnh = 0.5 * log ((1+ x) / (1-x))
    '''
    one_plus_x = (1 + x).clamp(min=1e-6)
    one_minus_x = (1 - x).clamp(min=1e-6)
    return 0.5*torch.log(one_plus_x/ one_minus_x)

class ActorSAC(nn.Module):
    """
      This arch is standard based on SAC paper
    """
    def __init__(self,
                action_space,
                hidden_sizes = [256, 256],
                input_dim = None,
                max_action = None,
                LOG_STD_MAX = 2,
                LOG_STD_MIN = -20,
                device = 'cpu'
                ):

        super(ActorSAC, self).__init__()
        self.hsize_1 = hidden_sizes[0]
        self.hsize_2 = hidden_sizes[1]
        action_dim, action_space_type = get_action_info(action_space)

        if len(hidden_sizes) == 2:
            self.net = nn.Sequential(
                            nn.Linear(input_dim[0], self.hsize_1),
                            nn.ReLU(),
                            nn.Linear(self.hsize_1, self.hsize_2),
                            nn.ReLU()
                            )

        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim[0], hidden_sizes[0]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                nn.ReLU()
                )
        # add layer for to learn exploration noise
        self.fc_mean = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std = nn.Linear(hidden_sizes[-1], action_dim)

        ###
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        # Here are actions max/limit:
        # 1.0 HalfCheetah-v3
        # 1.0 Walker2d-v3
        # 1.0 Hopper-v3
        # 1.0 Swimmer-v3
        # 0.4000000059604645 Humanoid-v3
        # 1.0 Reacher-v2
        # 1.0 InvertedDoublePendulum-v2
        # 3.0 InvertedPendulum-v2
        # 1.0 Ant-v3
        self.act_limit  = max_action
        self.LOG_STD_MAX = LOG_STD_MAX
        self.LOG_STD_MIN = LOG_STD_MIN

    def get_logprob(self, s, action):
        '''
            get log_probs
        '''
        raw_actions = atanh(action) # [B, D] --> [B, D]
        x = self.net(s)
        mu  = self.fc_mean(x)
        log_std  = self.log_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # std and mu are [B, D]
        cat = torch.distributions.Normal(mu, std)
        logprobs  = cat.log_prob(raw_actions).sum(axis=-1) # sum([B, D]) ==> [B]
        # mle_logprobs: [B]
        logprobs -= (2*(np.log(2) - action - F.softplus(-2*action))).sum(axis=1)

        # mle_logprobs: [B] ==> [B, 1]
        logprobs = logprobs.unsqueeze(-1)

        return logprobs


    def forward(self, x, gt_actions = None, deterministic=False, with_logprob=True, with_no_squash=False, with_log_mle = False):
        '''
            input (x  : B * D where B is batch size and D is input_dim
        '''
        x = self.net(x)
        mu  = self.fc_mean(x)
        log_std  = self.log_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = torch.distributions.Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu

        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # SAC paper (arXiv 1801.01290) appendix C
            # This is a more numerically-stable equivalent to Eq 21.

            # pi_action [B, num_actions], logp_pi: [B]
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # logp_pi:  [B, num_actions].sum(-1) ==> [B]
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
            # logp_pi: [B] ==> [B, 1]
            logp_pi = logp_pi.unsqueeze(-1)

        else:
            logp_pi = None

        squashed_actions = self.act_limit * torch.tanh(pi_action)

        if with_no_squash == False:
            # check we need log_mle

            if with_log_mle == True:
                ######
                # log pi(a|s) = log rho(u|s) - \sum^D log (1 - tanh^2(u)) # D action dimention
                # gt_actions: [B, D]
                #####
                raw_actions = atanh(gt_actions) # [B, D] --> [B, D]
                mle_logprobs  = pi_distribution.log_prob(raw_actions).sum(axis=-1) # sum([B, D]) ==> [B]
                # mle_logprobs: [B]
                mle_logprobs -= (2*(np.log(2) - gt_actions - F.softplus(-2*gt_actions))).sum(axis=1)

                # mle_logprobs: [B] ==> [B, 1]
                mle_logprobs = mle_logprobs.unsqueeze(-1)

                return squashed_actions, logp_pi, mle_logprobs

            else:
                return squashed_actions, logp_pi

        else:
            return squashed_actions, logp_pi, pi_action

class CriticSACMulti(nn.Module):
    def __init__(self,
                action_space,
                hidden_sizes = [256, 256],
                input_dim = None,
                number_of_qs = 2,
                device = 'cpu'
                ):

        super(CriticSACMulti, self).__init__()
        action_dim, action_space_type = get_action_info(action_space)

        # It uses two different Q networks
        # Q1 architecture
        self.dim_i = input_dim[0] + action_dim
        self.hidden_sizes = hidden_sizes
        self.number_of_qs = number_of_qs

        self.q1 = self.create_net_layes()
        self.q2 = self.create_net_layes()
        if self.number_of_qs > 2:
            self.q3 = self.create_net_layes()

        if self.number_of_qs == 4:
            self.q4 = self.create_net_layes()

    def create_net_layes(self):
        '''
            create 3 or 4 layers networks
        '''
        return nn.Sequential(
                            nn.Linear(self.dim_i, self.hidden_sizes[0]),
                            nn.ReLU(),
                            nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
                            nn.ReLU(),
                            nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2]),
                            nn.ReLU(),
                            nn.Linear(self.hidden_sizes[2], 1),
                            )

    def forward(self, obs, a, q_id=None):
        '''
            input (x): B * D where B is batch size and D is input_dim
            input (u): B * A where B is batch size and A is action_dim
            pre_act_rew: B * (A + 1) where B is batch size and A + 1 is input_dim
        '''
        xu = torch.cat([obs, a], dim =-1)

        if q_id is None:
            x1 = self.q1(xu)
            # Q2
            x2 = self.q2(xu)

            if self.number_of_qs == 2:
                # each q is [b_size, 1] ==>cat ==> [num_qs, b_size, 1]
                all_qs = torch.cat(
                    [x1.unsqueeze(0), x2.unsqueeze(0)], 0)
                return all_qs

            elif self.number_of_qs == 3:
                # Q3
                x3 = self.q3(xu)
                all_qs = torch.cat(
                    [x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0)], 0)
                return all_qs

            else:
                # Q3
                x3 = self.q3(xu)
                # Q4
                x4 = self.q4(xu)

                all_qs = torch.cat(
                    [x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0), x4.unsqueeze(0)], 0)
                return all_qs

        else:
            return self.qs[q_id](xu)

    def get_Qi(self, obs, a, q_id):
        '''
            returns values for a specific qs
        '''
        return self(obs, a, q_id)
