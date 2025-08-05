from poke_env.player.player import Player
from poke_env.player.battle_order import BattleOrder
from poke_env.battle import AbstractBattle
from examples.reinforcement_learning import EnhancedActorCriticModule, EnhancedPokemonEnv

import numpy as np
import torch

from ray.rllib.core import Columns
from gymnasium.spaces import Dict as DictSpace, Box, Discrete

def load_model_from_checkpoint(pth_path, observation_space, action_space, model_config):
    """
    Load a saved .pth file into your EnhancedActorCriticModule
    """
    # Create model instance with same architecture
    model = EnhancedActorCriticModule(
        observation_space=observation_space,
        action_space=action_space,
        inference_only=True,  # Set to True if only doing inference
        model_config=model_config,
        catalog_class=None,  # Or whatever you used originally
    )
    
    # Load the state dict
    state_dict = torch.load(pth_path, map_location='cpu', weights_only=False)
    
    # Load the weights into the model
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Model loaded successfully with strict=True")
    except RuntimeError as e:
        print(f"Strict loading failed: {e}")
        print("Trying with strict=False...")
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded with strict=False (some weights may be missing/extra)")
    
    return model

class RLPlayer(Player):
  def __init__(self, pth_path, **kwargs):
    super().__init__(**kwargs)
    self.model = load_model_from_checkpoint(
      pth_path=pth_path,
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
    self.env = EnhancedPokemonEnv()

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
            output = self.model._forward(batch)
            
            # Extract masked logits
            masked_logits = output[Columns.ACTION_DIST_INPUTS]
            
            # Sample action using categorical distribution
            action_dist = torch.distributions.Categorical(logits=masked_logits)
            action = action_dist.sample().item()
        
        # Convert action to battle order
        return self.env.action_to_order(np.int64(action), battle)
        
    except Exception as e:
        print(f"Error in choose_move, falling back to random: {e}")
        import traceback
        traceback.print_exc()
        return self.choose_random_move(battle)