from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player

import numpy as np
import torch
from ray.rllib.core import Columns
from ray.rllib.core.rl_module import RLModule, MultiRLModule
from ray.tune.registry import register_env

from poke_env.battle import AbstractBattle
from examples.reinforcement_learning import EnhancedPokemonEnv

class RLPlayer(Player):
    def __init__(self, checkpoint="", agent_id=None, **kwargs):
        super().__init__(**kwargs)
        register_env("showdown", EnhancedPokemonEnv.create_single_agent_env)
        
        self.multi_rl_module = RLModule.from_checkpoint(checkpoint)
        
        if not isinstance(self.multi_rl_module, MultiRLModule):
            raise TypeError(f"Expected MultiRLModule, got {type(self.multi_rl_module)}")
        
        # Get available agents
        available_agents = list(self.multi_rl_module._rl_modules.keys())
        
        # Set agent_id
        if agent_id is None:
            self.agent_id = available_agents[0]
        else:
            if agent_id not in available_agents:
                raise ValueError(f"Agent '{agent_id}' not found. Available: {available_agents}")
            self.agent_id = agent_id
        
        # Get the individual ActorCriticModule
        self.actor_critic_module = self.multi_rl_module._rl_modules[self.agent_id]
        
        # Set to eval mode
        if hasattr(self.actor_critic_module, 'eval'):
            self.actor_critic_module.eval()

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        try:
            env = EnhancedPokemonEnv()
            obs = env.embed_battle(battle)
            obs["observations"] = torch.tensor(obs["observations"], dtype=torch.float32).unsqueeze(0)
            obs["action_mask"] = torch.tensor(obs["action_mask"], dtype=torch.bool).unsqueeze(0)
            
            with torch.no_grad():
                batch = {Columns.OBS: obs}
                output = self.actor_critic_module._forward(batch)
                logits = output[Columns.ACTION_DIST_INPUTS]
                
                # Sample action
                action_dist = torch.distributions.Categorical(logits=logits)
                action = np.int64(action_dist.sample().item())
            
            return env.action_to_order(action, battle)
        except Exception as e:
            import traceback
            print(f"Exception random move chosen: {e}")
            traceback.print_exc()
            return self.choose_random_move(battle)
