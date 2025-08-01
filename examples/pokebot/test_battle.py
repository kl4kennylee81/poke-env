#!/usr/bin/env python3
"""
Simple Random Pokémon Bot Example using poke-env
Connects to localhost Showdown server and battles two random bots
You can spectate at http://localhost:8000
"""

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
from tabulate import tabulate
from pokebot.test_player import TestPlayer
from pokebot.rl_bot import RLPlayer
import wandb
    
async def main():
    """Main function to run random bot battles"""
    
    # Create two random bot instances - simple as shown in the official docs
    test_player = RLPlayer(
    "/Users/kennethlee/Documents/poke-env/artifacts/pokemon-rl-model-prenorm:v61/model_checkpoint_1000/learner_group/learner/rl_module",
    account_configuration=AccountConfiguration("RLBot", None),
    battle_format="gen5randombattle"
    )
    second_player = MaxBasePowerPlayer(battle_format="gen5randombattle")
    await test_player.battle_against(second_player, n_battles=1)
    for battle_id, battle in test_player._battles.items():
        print(f"Battle ID: {battle_id} - {('L' if battle.lost else 'W')} ")
    print(f"W:{test_player.n_won_battles} L:{test_player.n_lost_battles} T:{test_player.n_tied_battles} Total: {test_player.n_finished_battles}")

    # game = await test_player.ladder(1)

if __name__ == "__main__":
    asyncio.run(main())
