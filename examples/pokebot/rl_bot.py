from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player

import numpy as np
import torch
import pickle
import json

from ray.rllib.core import Columns
from ray.rllib.core.rl_module import RLModule
from ray.tune.registry import register_env

from poke_env.battle import AbstractBattle
from examples.reinforcement_learning import EnhancedPokemonEnv


class RLPlayer(Player):
    def __init__(self, checkpoint_dir="", **kwargs):
        # Remove checkpoint_dir from kwargs before passing to parent
        super().__init__(**kwargs)
        register_env("showdown", EnhancedPokemonEnv.create_single_agent_env)
        
        # Load RLModule manually from checkpoint files
        self.rl_module = self._load_rl_module_from_files(checkpoint_dir)
        
        # Set to eval mode
        if hasattr(self.rl_module, 'eval'):
            self.rl_module.eval()
        
        # Create a single environment instance for embedding battles
        self.env = EnhancedPokemonEnv()

    def _load_rl_module_from_files(self, checkpoint_dir):
        """Load RLModule from individual checkpoint files"""
        import os
        
        # Load class and constructor arguments
        class_file = os.path.join(checkpoint_dir, "class_and_ctor_args.pkl")
        with open(class_file, 'rb') as f:
            class_and_ctor = pickle.load(f)
        
        # Load metadata
        metadata_file = os.path.join(checkpoint_dir, "metadata.json")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load module state
        state_file = os.path.join(checkpoint_dir, "module_state.pkl")
        with open(state_file, 'rb') as f:
            module_state = pickle.load(f)
        
        # Create the RLModule instance
        module_class = class_and_ctor["class"]
        ctor_args = class_and_ctor.get("ctor_args", {})
        ctor_kwargs = class_and_ctor.get("ctor_kwargs", {})
        
        # Instantiate the module
        if ctor_args and ctor_kwargs:
            rl_module = module_class(*ctor_args, **ctor_kwargs)
        elif ctor_args:
            rl_module = module_class(*ctor_args)
        elif ctor_kwargs:
            rl_module = module_class(**ctor_kwargs)
        else:
            rl_module = module_class()
        
        # Load the state dict
        rl_module.load_state_dict(module_state)
        
        return rl_module

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        try:
            # Get observation from the environment
            obs = self.env.embed_battle(battle)
            
            # Prepare batch in the format expected by the RLModule
            batch = {
                Columns.OBS: {
                    "observations": torch.tensor(
                        obs["observations"], 
                        dtype=torch.float32
                    ).unsqueeze(0),  # Add batch dimension
                    "action_mask": torch.tensor(
                        obs["action_mask"], 
                        dtype=torch.bool
                    ).unsqueeze(0)   # Add batch dimension
                }
            }
            
            with torch.no_grad():
                # Forward pass through the module
                output = self.rl_module._forward(batch)
                
                # Extract masked logits
                masked_logits = output[Columns.ACTION_DIST_INPUTS]
                
                # Sample action using categorical distribution
                action_dist = torch.distributions.Categorical(logits=masked_logits)
                action = action_dist.sample().item()
            
            # Convert action to battle order
            return self.env.action_to_order(action, battle)
            
        except Exception as e:
            print(f"Error in choose_move, falling back to random: {e}")
            import traceback
            traceback.print_exc()
            return self.choose_random_move(battle)


# Usage example:
# player = RLPlayer(
#     checkpoint_dir="/Users/kennethlee/Documents/poke-env/artifacts/pokemon-rl-model-prenorm:v61/model_checkpoint_1000/learner_group/learner/rl_module",
#     battle_format="gen8randombattle"
# )