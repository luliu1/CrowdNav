import torch
import torch.nn as nn
import numpy as np
import itertools
import logging
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import ObservableState, FullState
import crowd_nav.cadrl_utils.agent as agent
import crowd_nav.cadrl_utils.util as util
import crowd_nav.cadrl_utils.network as network


def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, mlp_dims):
        super().__init__()
        self.value_network = mlp(input_dim, mlp_dims)

    def forward(self, state):
        value = self.value_network(state)
        return value


class CADRL_ORIGINAL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'CADRL_ORIGINAL'
        self.trainable = True
        self.multiagent_training = None
        self.kinematics = None
        self.epsilon = None
        self.gamma = None
        self.sampling = None
        self.speed_samples = None
        self.rotation_samples = None
        self.query_env = None
        self.action_space = None
        self.speeds = None
        self.rotations = None
        self.action_values = None
        self.with_om = None
        self.cell_num = None
        self.cell_size = None
        self.om_channel_size = None
        self.self_state_dim = 6
        self.human_state_dim = 7
        self.joint_state_dim = self.self_state_dim + self.human_state_dim

    def configure(self, config):
        self.possible_actions = network.Actions()
        num_actions = self.possible_actions.num_actions
        self.net = network.NetworkVP_rnn(network.Config.DEVICE, 'network', num_actions)
        self.net.simple_load('../../checkpoints/network_01900000')
        logging.info('Policy: CADRL_Original without occupancy map')
        self.FOV_min_angle = config.getfloat('map', 'angle_min') * np.pi % (2*np.pi)
        self.FOV_max_angle = config.getfloat('map', 'angle_max') * np.pi % (2*np.pi)


    def predict(self, state):
        """
        Input state is the joint state of robot concatenated by the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')
        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)

        host_agent = agent.Agent(state.self_state.px, state.self_state.py, state.self_state.gx, state.self_state.gy, state.self_state.radius, state.self_state.v_pref, state.self_state.theta, 0)
        host_agent.vel_global_frame = np.array([state.self_state.vx, state.self_state.vy])

        other_agents = []
        for i, human_state in enumerate(state.human_states):
            if self.human_state_in_FOV(state.self_state, human_state):
                x = human_state.px
                y = human_state.py
                v_x = human_state.vx
                v_y = human_state.vy
                heading_angle = np.arctan2(v_y, v_x)
                pref_speed = np.linalg.norm(np.array([v_x, v_y]))
                goal_x = x + 5.0; goal_y = y + 5.0
                other_agents.append(agent.Agent(x, y, goal_x, goal_y, human_state.radius, pref_speed, heading_angle, i+1))
                other_agents[-1].vel_global_frame = np.array([v_x, v_y])
        obs = host_agent.observe(other_agents)[1:]
        obs = np.expand_dims(obs, axis=0)
        predictions = self.net.predict_p(obs, None)[0]
        raw_action = self.possible_actions.actions[np.argmax(predictions)]
        action = ActionRot(host_agent.pref_speed*raw_action[0], util.wrap(raw_action[1]))
        return action
        
    def human_state_in_FOV(self, self_state, human_state):
        rot = np.arctan2(human_state.py - self_state.py, human_state.px - self_state.px)
        angle = (rot - self_state.theta) % (2 * np.pi)
        if angle > self.FOV_min_angle or angle < self.FOV_max_angle or self.FOV_min_angle == self.FOV_max_angle:
            return True
        else:
            return False
