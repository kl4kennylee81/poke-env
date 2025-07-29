import random
from typing import List

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.target import Target
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player
from poke_env.calc.damage_calc_gen9 import calculate_damage
from poke_env.battle import Battle, DoubleBattle, Move
from typing import Any, Dict, List, Optional, Union
from pokebot.calculate_damage import calculate_damage_simple

def estimate_stats_from_base(pokemon):
    """
    Estimate Pokemon stats with realistic EV distribution.
    Assumes typical competitive spreads rather than max EVs everywhere.
    """
    level = pokemon.level
    base_stats = pokemon.base_stats
    estimated_stats = {}
    
    # Determine likely EV distribution based on base stats
    # Most competitive Pokemon invest heavily in 2-3 stats, minimally in others
    ev_distribution = get_realistic_ev_spread(base_stats)
    
    for stat_name, base_value in base_stats.items():
        if stat_name == "hp":
            # Check if this looks like actual HP values vs percentage
            current_hp = getattr(pokemon, 'current_hp', 0)
            max_hp = getattr(pokemon, 'max_hp', 0)
            
            if max_hp > 100:
                # This looks like actual HP (player pokemon)
                estimated_stats[stat_name] = max_hp
            else:
                # Estimate with realistic EV investment
                ev_investment = ev_distribution.get(stat_name, 0)
                estimated_stats[stat_name] = ((2 * base_value + 31 + ev_investment // 4) * level) // 100 + level + 10
        else:
            # Use realistic EV distribution for other stats
            ev_investment = ev_distribution.get(stat_name, 0)
            estimated_stats[stat_name] = ((2 * base_value + 31 + ev_investment // 4) * level) // 100 + 5
    
    return estimated_stats


def get_realistic_ev_spread(base_stats):
    """
    Generate a realistic EV spread based on base stat distribution.
    Total EVs = 510, max per stat = 252.
    """
    # Create list of (stat_name, base_value) sorted by base value
    stats_by_value = sorted(base_stats.items(), key=lambda x: x[1], reverse=True)
    
    ev_spread = {stat: 0 for stat in base_stats.keys()}
    remaining_evs = 510
    
    # Strategy: Invest heavily in top 2 stats, moderately in 3rd, minimally in rest
    priorities = [
        (252, 1),  # Max invest in highest stat
        (252, 1),  # Max invest in second highest stat  
        (6, 1),    # Minimal invest in remaining stats for rounding
    ]
    
    stat_index = 0
    for ev_amount, num_stats in priorities:
        for _ in range(min(num_stats, len(stats_by_value) - stat_index)):
            if remaining_evs >= ev_amount and stat_index < len(stats_by_value):
                stat_name = stats_by_value[stat_index][0]
                actual_investment = min(ev_amount, remaining_evs)
                ev_spread[stat_name] = actual_investment
                remaining_evs -= actual_investment
                stat_index += 1
    
    # Distribute any remaining EVs to unfilled stats
    while remaining_evs > 0 and stat_index < len(stats_by_value):
        stat_name = stats_by_value[stat_index][0]
        investment = min(6, remaining_evs)  # Small investments
        ev_spread[stat_name] = investment
        remaining_evs -= investment
        stat_index += 1
    
    return ev_spread

def estimate_damage(
        attacker: Pokemon,
        defender: Pokemon,
        move: Move,
        battle: Union[Battle, DoubleBattle]
):
    # Check attacker stats
    attacker_has_stats = all(
        isinstance(x, (int, float)) and x is not None 
        for x in attacker.stats.values()
    )
    if not attacker_has_stats:
        raise ValueError("Attacker stats not available - cannot calculate damage")
    
    # Check defender stats
    defender_has_stats = all(
        isinstance(x, (int, float)) and x is not None 
        for x in defender.stats.values()
    )
    
    if defender_has_stats:
        # Use original calculator
        return calculate_damage_simple(
            attacker, defender, move, battle
        )
    else:
        estimated_stats = estimate_stats_from_base(defender)
            
        original_stats = defender._stats.copy()
        defender._stats = estimated_stats
        
        try:
            return calculate_damage_simple(
                attacker, defender, move, battle
            )
        finally:
            # Always restore original stats
            defender._stats = original_stats

def calculate_damage_with_base_stats(
    attacker_identifier: str,
    defender_identifier: str,
    move: Move,
    battle: Union[Battle, DoubleBattle],
    is_critical: bool = False,
):
    """
    Calculate damage using base stats for estimation when actual stats unavailable.
    
    :param estimation_method: "realistic" uses smart EV distribution, 
                             "conservative" uses lower EV assumptions
    """
    attacker = battle.get_pokemon(attacker_identifier)
    defender = battle.get_pokemon(defender_identifier)

    # Check attacker stats
    attacker_has_stats = all(
        isinstance(x, (int, float)) and x is not None 
        for x in attacker.stats.values()
    )
    if not attacker_has_stats:
        raise ValueError("Attacker stats not available - cannot calculate damage")
    
    # Check defender stats
    defender_has_stats = all(
        isinstance(x, (int, float)) and x is not None 
        for x in defender.stats.values()
    )
    
    if defender_has_stats:
        # Use original calculator
        return calculate_damage(
            attacker_identifier, defender_identifier, move, battle, is_critical
        )
    else:
        estimated_stats = estimate_stats_from_base(defender)
            
        original_stats = defender._stats.copy()
        defender._stats = estimated_stats
        
        try:
            return calculate_damage(
                attacker_identifier, defender_identifier, move, battle, is_critical
            )
        finally:
            # Always restore original stats
            defender._stats = original_stats


def estimate_matchup(mon: Pokemon, opponent: Pokemon):
    SPEED_TIER_COEFICIENT = 0.1
    HP_FRACTION_COEFICIENT = 0.4
    score = max([opponent.damage_multiplier(t) for t in mon.types if t is not None])
    score -= max(
        [mon.damage_multiplier(t) for t in opponent.types if t is not None]
    )
    if mon.base_stats["spe"] > opponent.base_stats["spe"]:
        score += SPEED_TIER_COEFICIENT
    elif opponent.base_stats["spe"] > mon.base_stats["spe"]:
        score -= SPEED_TIER_COEFICIENT

    score += mon.current_hp_fraction * HP_FRACTION_COEFICIENT
    score -= opponent.current_hp_fraction * HP_FRACTION_COEFICIENT

    return score

def find_best_revenge_killer(battle: AbstractBattle):
    """Find a switch that can 1HKO the opponent and outspeed them"""
    if not battle.available_switches:
        return None
    
    opponent = battle.opponent_active_pokemon
    opponent_current_hp = opponent.current_hp_fraction * opponent.max_hp
    
    revenge_killers = []
    
    for switch_pokemon in battle.available_switches:
        # Check if this Pokemon can outspeed the opponent
        if switch_pokemon.base_stats["spe"] <= opponent.base_stats["spe"]:
            continue  # Skip if can't outspeed
        
        # Check all moves this Pokemon can use for 1HKO potential
        can_1hko = False
        best_move_damage = 0
        
        # Get the Pokemon's moveset (you might need to adjust this based on your data structure)
        for move in switch_pokemon.moves.values():  # or however you access moves
            try:
                min_damage, max_damage = estimate_damage(
                    switch_pokemon, opponent, move, battle
                )
                
                # Check if minimum damage can 1HKO
                if min_damage >= opponent_current_hp:
                    can_1hko = True
                    best_move_damage = min_damage
                    break  # Found a guaranteed 1HKO move
                    
                best_move_damage = max(best_move_damage, max_damage)
                
            except:
                continue  # Skip moves that can't be calculated
        
        if can_1hko:
            revenge_killers.append((switch_pokemon, best_move_damage))
    
    if revenge_killers:
        # Return the switch with highest damage potential
        return max(revenge_killers, key=lambda x: x[1])[0]
    
    return None

def get_move_score(attacker_id, defender_id, battle, move):
    try:
        return calculate_damage_with_base_stats(attacker_id, defender_id, move, battle)[1]
    except:
        return 0

class TestPlayer(Player):
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        # Handle forced switches (when Pokemon faints)
        if (battle.force_switch or not battle.available_moves) and battle.available_switches:
            best_switch = max(battle.available_switches, 
                     key=lambda s: estimate_matchup(s, battle.opponent_active_pokemon))
                    # First check if we have a revenge killer
            revenge_killer = find_best_revenge_killer(battle)
            if revenge_killer:
                return self.create_order(revenge_killer)
            
            # Otherwise use the original matchup-based switching
            best_switch = max(battle.available_switches, 
                    key=lambda s: estimate_matchup(s, battle.opponent_active_pokemon))
            return self.create_order(best_switch)
        
        # Normal move selection logic
        if not battle.available_moves or not battle.active_pokemon:
            return self.choose_random_move(battle)
        
        attacker_id = battle.active_pokemon.identifier("p1")
        defender_id = battle.opponent_active_pokemon.identifier("p2")
        
        best_move = max(battle.available_moves, key=lambda m : get_move_score(attacker_id, defender_id, battle, m))
        min_damage, max_damage = calculate_damage_with_base_stats(attacker_id, defender_id, best_move, battle)
        opp_stats = estimate_stats_from_base(battle.opponent_active_pokemon)
        opp_hp = battle.opponent_active_pokemon.current_hp_fraction * opp_stats["hp"]
        if min_damage >= opp_hp and battle.active_pokemon.stats["spe"] > opp_stats["spe"]:
            # Guaranteed KO - use the move
            return self.create_order(best_move)
        opp_eff = max([battle.active_pokemon.damage_multiplier(t) 
        for t in battle.opponent_active_pokemon.types if t is not None])
        if min_damage >= opp_hp and opp_eff <= 1:
            return self.create_order(best_move)
        if min_damage * 2 >= opp_hp and battle.active_pokemon.stats["spe"] > opp_stats["spe"]:
            return self.create_order(best_move)
        if min_damage * 2 >= opp_hp and opp_eff <= 1:
            return self.create_order(best_move)
        if not battle.available_switches:
            return self.create_order(best_move)
        
        best_switch = max(battle.available_switches, 
            key=lambda s: estimate_matchup(s, battle.opponent_active_pokemon))
        if (estimate_matchup(battle.active_pokemon, battle.opponent_active_pokemon) <= -1.0 and
            estimate_matchup(best_switch,battle.opponent_active_pokemon) >= 1.0):
            return self.create_order(best_switch)
        if (min_damage * 4 < opp_hp and estimate_matchup(best_switch,battle.opponent_active_pokemon) >= 1.0):
            return self.create_order(best_switch)
        if (random.random() < 0.15 and 
            estimate_matchup(battle.active_pokemon, battle.opponent_active_pokemon) < 
            estimate_matchup(best_switch,battle.opponent_active_pokemon) and 
            estimate_matchup(best_switch,battle.opponent_active_pokemon) > 0
            ):
            return self.create_order(best_switch)
        return self.create_order(best_move)