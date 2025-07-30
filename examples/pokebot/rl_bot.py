import random
from typing import List

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player
from poke_env.battle import Battle
from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Space
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.core import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module import RLModule, MultiRLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.tune.registry import register_env

from poke_env.battle import AbstractBattle, Battle
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import RandomPlayer

class ExampleEnv(SinglesEnv[npt.NDArray[np.float32]]):
    LOW = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
    HIGH = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: Box(
                np.array(self.LOW, dtype=np.float32),
                np.array(self.HIGH, dtype=np.float32),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

    @classmethod
    def create_single_agent_env(cls, config: Dict[str, Any]) -> SingleAgentWrapper:
        env = cls(
            battle_format=config["battle_format"],
            log_level=25,
            open_timeout=None,
            strict=False,
        )
        opponent = RandomPlayer()
        return SingleAgentWrapper(env, opponent)

    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle):
        assert isinstance(battle, Battle)
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if battle.opponent_active_pokemon is not None:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=battle.opponent_active_pokemon._data.type_chart,
                )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)


class ActorCriticModule(TorchRLModule, ValueFunctionAPI):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        inference_only: bool,
        model_config: Dict[str, Any],
        catalog_class: Any,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            inference_only=inference_only,
            model_config=model_config,
            catalog_class=catalog_class,
        )
        self.model = nn.Linear(10, 100)
        self.actor = nn.Linear(100, 26)
        self.critic = nn.Linear(100, 1)

    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        obs = batch[Columns.OBS]
        embeddings = self.model(obs)
        logits = self.actor(embeddings)
        return {Columns.EMBEDDINGS: embeddings, Columns.ACTION_DIST_INPUTS: logits}

    def compute_values(
        self, batch: Dict[str, Any], embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if embeddings is None:
            embeddings = self.model(batch[Columns.OBS])
        return self.critic(embeddings).squeeze(-1)

class RLPlayer(Player):
    def __init__(self, checkpoint="", agent_id=None, **kwargs):
        super().__init__(**kwargs)
        register_env("showdown", ExampleEnv.create_single_agent_env)
        
        # Load the MultiRLModule
        self.multi_rl_module = RLModule.from_checkpoint(checkpoint)
        
        if not isinstance(self.multi_rl_module, MultiRLModule):
            raise ValueError("Expected MultiRLModule but got single agent module")
        
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
            env = ExampleEnv()
            obs = env.embed_battle(battle)
            
            # Convert to tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                # Method 2: Use the individual ActorCriticModule directly
                batch = {Columns.OBS: obs_tensor}
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
