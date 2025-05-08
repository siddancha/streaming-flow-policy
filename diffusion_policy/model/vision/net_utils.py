import robomimic
from torch import nn as nn


class AgentPosEncoder(robomimic.models.base_nets.Module):
    def __init__(self, agent_pos_emb_dim=32):
        """
        Simple MLP stacks to encode keypoint features and positions.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, agent_pos_emb_dim),
            nn.ReLU(),
            nn.Linear(agent_pos_emb_dim, agent_pos_emb_dim),
        )
        self.agent_pos_emb_dim = agent_pos_emb_dim

    def forward(self, x):
        return self.net(x)

    def output_shape(self, input_shape=None):
        # To make it an instance of robomimic.models.base_nets.Module
        return [self.agent_pos_emb_dim]
