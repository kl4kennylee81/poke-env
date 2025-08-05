import asyncio
import time
from poke_env.player.battle_order import BattleOrder
from poke_env.battle.battle import Battle
from poke_env import AccountConfiguration
import json
from datetime import datetime
from enum import Enum
from collections import OrderedDict
from typing import Any, Set
from poke_env import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
from ray.rllib.algorithms.algorithm import Algorithm
from examples.pokebot.test_player import TestPlayer
import wandb
import os

from examples.pokebot.rl_bot import RLPlayer
    
async def main():
    wandb.init(project="pokemon-rl")
    artifact = wandb.use_artifact("pokemon-pytorch-model-prenorm:latest")
    artifact_dir = artifact.download()
    pth_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pth')]
    if len(pth_files) == 0:
        return
    pth_path = f"{artifact_dir}/{pth_files[0]}"
    test_player = RLPlayer(
    pth_path=pth_path,
    account_configuration=AccountConfiguration("RLBot2", None),
    battle_format="gen5randombattle"
    )
    second_player = MaxBasePowerPlayer(battle_format="gen5randombattle")
    await test_player.battle_against(second_player, n_battles=10)
    for battle_id, battle in test_player._battles.items():
        print(f"Battle ID: {battle_id} - {('L' if battle.lost else 'W')} ")
    print(f"W:{test_player.n_won_battles} L:{test_player.n_lost_battles} T:{test_player.n_tied_battles} Total: {test_player.n_finished_battles}")
    # game = await test_player.ladder(1)

if __name__ == "__main__":
    asyncio.run(main())
