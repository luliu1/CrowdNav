from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.cadrl_original_data import CADRL_ORIGINAL
from crowd_nav.policy.lstm_rl import LstmRL
from crowd_nav.policy.sarl import SARL

policy_factory['cadrl'] = CADRL
policy_factory['cadrl_original'] = CADRL_ORIGINAL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
