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
import wandb

from examples.pokebot.rl_bot import RLPlayer
    
async def main():
    """Main function to run random bot battles"""
    
    # Create two random bot instances - simple as shown in the official docs
    test_player = RLPlayer(
    account_configuration=AccountConfiguration("RLBot2", None),
    battle_format="gen5randombattle"
    )
    second_player = MaxBasePowerPlayer(battle_format="gen5randombattle")
    await test_player.battle_against(second_player, n_battles=50)
    for battle_id, battle in test_player._battles.items():
        print(f"Battle ID: {battle_id} - {('L' if battle.lost else 'W')} ")
    print(f"W:{test_player.n_won_battles} L:{test_player.n_lost_battles} T:{test_player.n_tied_battles} Total: {test_player.n_finished_battles}")

if __name__ == "__main__":
    asyncio.run(main())
