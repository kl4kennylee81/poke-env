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
import json
from datetime import datetime
from enum import Enum
from collections import OrderedDict
from typing import Any, Set
from poke_env import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
from tabulate import tabulate
from pokebot.test_player import TestPlayer
    
async def main():
    """Main function to run random bot battles"""
    
    # Create two random bot instances - simple as shown in the official docs
    test_player = TestPlayer()
    second_player = SimpleHeuristicsPlayer()
    await test_player.battle_against(second_player, n_battles=500)
    
    print(f"W:{test_player.n_won_battles} L:{test_player.n_lost_battles} T:{test_player.n_tied_battles} Total: {test_player.n_finished_battles}")

if __name__ == "__main__":
    print("Random Pokémon Bot Battle Spectator")
    print("=" * 60)
    print("Based on official poke-env documentation")
    print("=" * 60)
    print("Setup Instructions:")
    print("1. pip install poke-env")
    print("2. Clone and setup Pokémon Showdown:")
    print("   git clone https://github.com/smogon/pokemon-showdown.git")
    print("   cd pokemon-showdown")
    print("   npm install")
    print("   node pokemon-showdown start")
    print("3. Open http://localhost:8000 to watch battles")
    print("=" * 60)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")