from typing import Any, Dict, Optional
import traceback
import sys

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from tqdm import tqdm
from gymnasium.spaces import Dict as DictSpace, Box, Discrete, Space
from ray.rllib.algorithms import PPOConfig
from ray.rllib.core import Columns
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
import ray

# WandB for experiment tracking
import wandb

from poke_env import Player
from poke_env.battle import AbstractBattle, Battle
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import SimpleHeuristicsPlayer, MaxBasePowerPlayer
import os

class EnhancedPokemonEnv(SinglesEnv[npt.NDArray[np.float32]]):
    # Extended state space: 31 features - ALL NORMALIZED
    # Features breakdown:
    # [0-3]  moves_base_power(4):           [-1, 3] (-1=unavailable, 0.4-2.0=normalized power)
    # [4-7]  moves_dmg_multiplier(4):       [0, 4] (type effectiveness: 0=no effect, 4=4x damage)
    # [8-9]  fainted_team(1), fainted_opp(1): [0, 1] (fraction of team fainted)
    # [10-11] my_hp(1), opp_hp(1):          [0, 1] (HP fractions)
    # [12]   speed_advantage(1):            [-1, 1] (relative speed: -1=much slower, +1=much faster)
    # [13]   available_switches(1):         [0, 1] (fraction of max switches available)
    # [14]   opp_threat(1):                 [0, 1] (normalized threat level)
    # [15-19] switch_hp_ratios(5):          [0, 1] (HP ratios of available switches)
    # [20-24] switch_type_effectiveness(5): [-1, 1] (relative type advantage: negative=bad, positive=good)

    LOW = [-1, -1, -1, -1,    # moves_base_power: -1 for unavailable
           0, 0, 0, 0,        # moves_dmg_multiplier: 0 minimum
           0, 0,              # fainted ratios: 0 minimum
           0, 0,              # HP ratios: 0 minimum
           -1,                # speed_advantage: -1 for much slower
           0,                 # available_switches: 0 minimum
           0,                 # opp_threat: 0 minimum
           0, 0, 0, 0, 0,     # switch_hp_ratios: 0 minimum
           -1, -1, -1, -1, -1, # switch_type_effectiveness: -1 for very bad matchup
           0, 0, 0, 0, 0, 0
           ] 

    HIGH = [3, 3, 3, 3,       # moves_base_power: 3.0 for very high power moves (300 base power)
            4, 4, 4, 4,       # moves_dmg_multiplier: 4 for 4x effectiveness
            1, 1,             # fainted ratios: 1 maximum (all fainted)
            1, 1,             # HP ratios: 1 maximum (full HP)
            1,                # speed_advantage: 1 for much faster
            1,                # available_switches: 1 maximum (all 5 available)
            1,                # opp_threat: 1 maximum (4x threat normalized)
            1, 1, 1, 1, 1,    # switch_hp_ratios: 1 maximum (full HP)
            1, 1, 1, 1, 1,
            6,6,6,6,6,6]    # switch_type_effectiveness: 1 for very good matchup

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: DictSpace({
                "observations": Box(
                    low=np.array(self.LOW, dtype=np.float32),
                    high=np.array(self.HIGH, dtype=np.float32),
                    dtype=np.float32,
                ),
                "action_mask": Box(
                    low=np.zeros(10, dtype=np.float32),
                    high=np.ones(10, dtype=np.float32),
                    dtype=np.float32,
                )
            })
            for agent in self.possible_agents
        }

    @classmethod
    def create_single_agent_env(cls, config: Dict[str, Any]) -> SingleAgentWrapper:
        env = cls(
            battle_format=config["battle_format"],
            log_level=30,
            open_timeout=None,
            strict=False,
        )
        opponent = MaxBasePowerPlayer()
        return SingleAgentWrapper(env, opponent)

    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle):
        assert isinstance(battle, Battle)

        # Original features - NORMALIZED for consistent scale
        moves_base_power = -np.ones(4)  # -1 for unavailable moves
        moves_dmg_multiplier = np.ones(4)

        for i, move in enumerate(battle.available_moves):
            # Normalize base power: typical range 40-120, max ~200
            moves_base_power[i] = move.base_power / 100.0  # 40->0.4, 80->0.8, 120->1.2
            if battle.opponent_active_pokemon is not None:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=battle.opponent_active_pokemon._data.type_chart,
                )


        stat_keys = ["hp", "atk", "def", "spa", "spd", "spe"]
        active_stats = np.array([(battle.active_pokemon.stats.get(key, 0) / 100) for key in stat_keys] if battle.active_pokemon else [0] * 6)

        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6

        my_hp_ratio = 0.0
        opp_hp_ratio = 0.0
        if battle.active_pokemon:
            my_hp_ratio = battle.active_pokemon.current_hp_fraction
        if battle.opponent_active_pokemon:
            opp_hp_ratio = battle.opponent_active_pokemon.current_hp_fraction

        speed_advantage = 0.0
        if battle.active_pokemon and battle.opponent_active_pokemon:
            my_speed = battle.active_pokemon.base_stats.get('spe', 100)
            opp_speed = battle.opponent_active_pokemon.base_stats.get('spe', 100)
            speed_advantage = (my_speed - opp_speed) / max(my_speed + opp_speed, 1)
            speed_advantage = np.clip(speed_advantage, -1, 1)

        available_switches_count = len(battle.available_switches) / 5.0  # Max 5 switches

        opp_threat_level = 0.0
        if battle.active_pokemon and battle.opponent_active_pokemon:
            for opp_type in [battle.opponent_active_pokemon.type_1, battle.opponent_active_pokemon.type_2]:
                if opp_type:
                    multiplier = battle.active_pokemon.damage_multiplier(opp_type)
                    opp_threat_level = max(opp_threat_level, multiplier)
        opp_threat_level = min(opp_threat_level, 4.0) / 4.0  # Normalize 0-4 to 0-1

        switch_hp_ratios = np.zeros(5)
        available_switches_list = list(battle.available_switches)
        for i in range(min(5, len(available_switches_list))):
            switch_pokemon = available_switches_list[i]
            switch_hp_ratios[i] = switch_pokemon.current_hp_fraction

        switch_type_effectiveness = np.zeros(5)
        if battle.opponent_active_pokemon:
            for i in range(min(5, len(available_switches_list))):
                switch_pokemon = available_switches_list[i]
                # Use your suggested method for type effectiveness
                our_effectiveness = max([battle.opponent_active_pokemon.damage_multiplier(t) for t in switch_pokemon.types if t is not None], default=0.0)
                their_effectiveness = max([switch_pokemon.damage_multiplier(t) for t in battle.opponent_active_pokemon.types if t is not None], default=0.0)
                switch_type_effectiveness[i] = (min(our_effectiveness, 4.0) - min(their_effectiveness, 4.0)) / 4.0  # Normalize 0-4 to 0-1

        # Combine all features (ALL NORMALIZED) - Total: 31 features
        final_vector = np.concatenate([
            moves_base_power,           # 4 features: [-1, 3] (-1=unavailable, [0,300]=normalized power)
            moves_dmg_multiplier,       # 4 features: [0, 4] (type effectiveness multipliers [0x,.25x,0.5x,1,2x,4x])
            [fainted_mon_team, fainted_mon_opponent],  # 2 features: [0, 1] (fraction fainted)
            [my_hp_ratio, opp_hp_ratio],  # 2 features: [0, 1] (HP fractions)
            [speed_advantage, available_switches_count, opp_threat_level],  # 3 features: [-1,1], [0,1], [0,1]
            switch_hp_ratios,           # 5 features: [0,1] (HP ratios of available switches)
            switch_type_effectiveness,  # 5 features: [0,1] (type effectiveness vs opponent)
            active_stats,
        ])

        action_mask = np.zeros(10)
        for move in battle.available_moves:
            try:
                order = Player.create_order(move)
                action = self.order_to_action(order, battle)
                action_mask[action] = 1
            except Exception as e:
                battle.logger.warning(f"{str(e)} - in {str(order)}")
        if not battle.trapped or not battle.maybe_trapped:
          for switch in battle.available_switches:
              try:
                  order = Player.create_order(switch)
                  action = self.order_to_action(order, battle)
                  action_mask[action] = 1
              except Exception as e:
                  battle.logger.warning(f"{str(e)} - in {str(order)}")

        return {
            "observations": np.float32(final_vector),
            "action_mask": action_mask.astype(np.float32)
        }


class EnhancedActorCriticModule(TorchRLModule, ValueFunctionAPI):
    def __init__(
        self,
        observation_space: DictSpace,
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
        self.model = nn.Sequential(
            nn.Linear(31, 256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),

            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.LayerNorm(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.LayerNorm(16),
            nn.Linear(16, 10)
        )

        self.critic = nn.Sequential(
            nn.LayerNorm(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.LayerNorm(16),
            nn.Linear(16, 1)
        )

    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        obs = batch[Columns.OBS]["observations"]
        mask = batch[Columns.OBS]["action_mask"]

        embeddings = self.model(obs)
        logits = self.actor(embeddings)

        # Apply action masking with improved numerical stability
        masked_logits = torch.where(
            mask.bool(),
            logits,
            torch.full_like(logits, -1e8)
        )

        return {
            Columns.EMBEDDINGS: embeddings,
            Columns.ACTION_DIST_INPUTS: masked_logits
        }

    def compute_values(
        self, batch: Dict[str, Any], embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if embeddings is None:
            embeddings = self.model(batch[Columns.OBS]["observations"])
        return self.critic(embeddings).squeeze(-1)


def train_enhanced_pokemon_gpu(n_episodes, wandb_project="pokemon-rl", experiment_name=None, wandb_checkpoint=None):
    """Train enhanced Pokemon RL agent with GPU acceleration and WandB tracking"""

    # Initialize WandB
    if experiment_name is None:
        experiment_name = f"enhanced-pokemon-{n_episodes}ep"

    wandb.init(
        project=wandb_project,
        name=experiment_name,
        config={
            "n_episodes": n_episodes,
            "state_features": 31,
            "network_params": "~60k",
            "architecture": "31->256->128->64->32->16->10",
            "opponent": "MaxBasePowerPlayer",
            "battle_format": "gen5randombattle",
            "lr": 1e-4,
            "batch_size": 1024,
            "minibatch_size": 64,
            "gamma": 0.99
        }
    )

    register_env("enhanced_pokemon", EnhancedPokemonEnv.create_single_agent_env)

    config = PPOConfig()

    config = config.environment(
        "enhanced_pokemon",
        env_config={"battle_format": "gen5randombattle"},
        disable_env_checking=True,
    )

    config = config.env_runners(
        num_env_runners=0,
        create_env_on_local_worker=True,
    )

    config = config.framework(
        framework="torch",
        torch_compile_learner=False,
    )

    config = config.resources(num_gpus=1, num_cpus_for_local_worker=1)

    # Enhanced RL Module configuration
    config = config.rl_module(
        rl_module_spec=RLModuleSpec(
            module_class=EnhancedActorCriticModule,
            observation_space=DictSpace({
                "observations": Box(
                    low=np.array(EnhancedPokemonEnv.LOW, dtype=np.float32),
                    high=np.array(EnhancedPokemonEnv.HIGH, dtype=np.float32),
                    dtype=np.float32,
                ),
                "action_mask": Box(
                    low=np.zeros(10, dtype=np.float32),
                    high=np.ones(10, dtype=np.float32),
                    dtype=np.float32,
                )
            }),
            action_space=Discrete(10),
            model_config={},
        )
    )

    config = config.training(
        gamma=0.99,
        lr=1e-4,
        train_batch_size=1024,
        num_epochs=10,
        minibatch_size=64,
    )

    config = config.evaluation(evaluation_interval=None)

    algo = config.build_algo()
    cur_episode = 0
    if wandb_checkpoint:
      artifact = wandb.use_artifact(wandb_checkpoint)
      artifact_dir = artifact.download()
      cur_episode = artifact.metadata.get("cur_episode", 0)
      wandb_checkpoint = f"{artifact_dir}/model_checkpoint_{cur_episode}"
      algo = Algorithm.from_checkpoint(wandb_checkpoint)

    results = []
    pbar = tqdm(range(cur_episode, n_episodes), desc="Enhanced Training")

    # Track running averages for smoothed metrics
    reward_history = []
    episode_len_history = []

    for iteration in pbar:
        try:
            result = algo.train()

            # Extract key metrics
            episode_reward_mean = result.get("env_runners", {}).get("episode_return_mean", 0)
            episode_len_mean = result.get("env_runners", {}).get("episode_len_mean", 0)

            # Store metrics
            results.append({
                "iteration": iteration,
                "reward_mean": episode_reward_mean,
                "episode_len": episode_len_mean
            })

            # Track for smoothed metrics
            reward_history.append(episode_reward_mean)
            episode_len_history.append(episode_len_mean)

            # Calculate smoothed metrics (last 100 episodes)
            window_size = min(100, len(reward_history))
            reward_smoothed = np.mean(reward_history[-window_size:])
            episode_len_smoothed = np.mean(episode_len_history[-window_size:])

            # Log to WandB
            wandb.log({
                "iteration": iteration,
                "reward_mean": episode_reward_mean,
                "reward_smoothed": reward_smoothed,
                "episode_length_mean": episode_len_mean,
                "episode_length_smoothed": episode_len_smoothed,
                "total_episodes": iteration + 1,
            })

            # Update progress bar
            pbar.set_description(
                f"Iter {iteration}: Reward={episode_reward_mean:.2f} (smooth={reward_smoothed:.2f}), Len={episode_len_mean:.1f}"
            )

            # Save checkpoint every 100 iterations
            if iteration % 100 == 0 and iteration > 0:
                checkpoint_path = algo.save("/content/enhanced_pokemon_model")

                # Create artifact
                model_artifact = wandb.Artifact(
                    name=f"pokemon-rl-model-{experiment_name}",
                    type="model",
                    description=f"Trained Pokemon RL model after {n_episodes} episodes.",
                    metadata={
                        "reward": episode_reward_mean,
                        "reward_smoothed": reward_smoothed,
                        "n_episodes": n_episodes,
                        "cur_episode": iteration,
                        "state_features": 31,
                        "architecture": "31->256->128->64->32->10",
                        "opponent": "MaxBasePowerPlayer",
                        "battle_format": "gen5randombattle"
                    }
                )

                # Add the checkpoint directory to the artifact
                model_artifact.add_dir("/content/enhanced_pokemon_model", name=f"model_checkpoint_{iteration}")
                wandb.log_artifact(model_artifact)
                print("Model artifact successfully uploaded to WandB!")


        except Exception as e:
          print(f"Error at iteration {iteration}")
          print(f"Exception type: {type(e).__name__}")
          print(f"Exception message: {str(e)}")
          print(f"Exception args: {e.args}")
          print("Full stack trace:")

          # Print the full traceback
          exc_type, exc_value, exc_traceback = sys.exc_info()
          traceback.print_exception(exc_type, exc_value, exc_traceback)

          # Get the traceback as string for logging
          tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))

          wandb.log({
              "error_type": type(e).__name__,
              "error_message": str(e),
              "error_iteration": iteration,
              "full_traceback": tb_str
          })
          break

    # Final save
    final_checkpoint = algo.save("/content/enhanced_pokemon_model_final")
    print(f"training completed final model saved: {final_checkpoint}")

    # Final metrics
    if results:
        final_reward = results[-1]["reward_mean"]
        final_reward_smoothed = np.mean(reward_history[-100:]) if len(reward_history) >= 100 else np.mean(reward_history)

        print(f"Final average reward: {final_reward:.2f}")
        print(f"Final smoothed reward (last 100): {final_reward_smoothed:.2f}")

        # Log final summary to WandB
        wandb.log({
            "final_reward": final_reward,
            "final_reward_smoothed": final_reward_smoothed,
            "training_completed": True
        })

    # Save model as WandB artifact
    print("Saving model as WandB artifact...")
    try:
        # Create artifact
        model_artifact = wandb.Artifact(
            name=f"pokemon-rl-model-{experiment_name}",
            type="model",
            description=f"Trained Pokemon RL model after {n_episodes} episodes. "
                       f"Final reward: {final_reward:.2f}, Architecture: 31->256->128->64->32->10",
            metadata={
                "final_reward": final_reward,
                "final_reward_smoothed": final_reward_smoothed,
                "n_episodes": n_episodes,
                "cur_episode": n_episodes,
                "state_features": 31,
                "architecture": "31->256->128->64->32->10",
                "opponent": "MaxBasePowerPlayer",
                "battle_format": "gen5randombattle"
            }
        )

        # Add the checkpoint directory to the artifact
        model_artifact.add_dir("/content/enhanced_pokemon_model_final", name=f"model_checkpoint_{n_episodes}")
        wandb.log_artifact(model_artifact)
        print("Model artifact successfully uploaded to WandB")

    except Exception as e:
        print(f"Failed to save WandB artifact: {str(e)}")
        wandb.log({"artifact_save_error": str(e)})

    # Finish WandB run
    wandb.finish()

    return algo, results


if __name__ == "__main__":
    ray.init(
        num_gpus=0,
        num_cpus=1,
        ignore_reinit_error=True
    )
    algo, results = train_enhanced_pokemon_gpu(100, wandb_project="pokemon-rl", experiment_name="local-test")