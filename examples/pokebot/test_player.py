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
    HP_FRACTION_COEFFICIENT = 0.4
    
    # Type effectiveness calculations
    our_effectiveness = max([opponent.damage_multiplier(t) for t in mon.types if t is not None])
    their_effectiveness = max([mon.damage_multiplier(t) for t in opponent.types if t is not None])
    
    # Speed comparison
    our_speed = mon.base_stats["spe"]
    their_speed = opponent.base_stats["spe"]
    
    if our_speed > their_speed:
        # Faster = use our damage multiplier
        score = our_effectiveness
    else:
        # Slower = only good if we resist/neutral AND can hit back
        if their_effectiveness <= 0.5 and mon.current_hp_fraction > 0.33:  # Normal or resistant
            score = our_effectiveness
        elif their_effectiveness <= 1.0 and mon.current_hp_fraction > 0.5:
            score = our_effectiveness
        else:
            score = -their_effectiveness  # Bad matchup (slower + taking super effective)
    
    # HP difference
    score += (mon.current_hp_fraction - opponent.current_hp_fraction) * HP_FRACTION_COEFFICIENT
    
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
    
def find_best_lethal_move(battle, attacker_id, defender_id):
    """Find the most accurate move that guarantees a KO"""
    opp_stats = estimate_stats_from_base(battle.opponent_active_pokemon)
    opp_hp = battle.opponent_active_pokemon.current_hp_fraction * opp_stats["hp"]
    
    lethal_moves = []
    for move in battle.available_moves:
        try:
            min_damage, _ = calculate_damage_with_base_stats(attacker_id, defender_id, move, battle)
            if min_damage >= opp_hp:
                # Handle moves with True accuracy (always hit) vs numeric accuracy
                accuracy = 100 if move.accuracy is True else (move.accuracy or 100)
                lethal_moves.append((move, accuracy))
        except:
            min_damage, _ = calculate_damage_simple(battle.active_pokemon, battle.opponent_active_pokemon, move, battle)
            if min_damage >= opp_hp:
                # Handle moves with True accuracy (always hit) vs numeric accuracy
                accuracy = 100 if move.accuracy is True else (move.accuracy or 100)
                lethal_moves.append((move, accuracy))
    
    if lethal_moves:
        # Return most accurate lethal move
        return max(lethal_moves, key=lambda x: x[1])[0]
    
    return None

class TestPlayer(Player):
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        order = self._choose_move(battle)
        print(order)
        return order
    def _choose_move(self, battle: AbstractBattle) -> BattleOrder:
        try:
            # Handle forced switches (when Pokemon faints)
            if (battle.force_switch or not battle.available_moves) and battle.available_switches:
                # First check if we have a revenge killer
                revenge_killer = find_best_revenge_killer(battle)
                if revenge_killer:
                    return self.create_order(revenge_killer)
                
                # Otherwise use matchup-based switching
                best_switch = max(battle.available_switches, 
                        key=lambda s: estimate_matchup(s, battle.opponent_active_pokemon))
                return self.create_order(best_switch)

            if not battle.available_moves or not battle.active_pokemon or not battle.opponent_active_pokemon:
                return self.choose_random_move(battle)
            
            # Get identifiers for damage calculation
            attacker_id = battle.active_pokemon.identifier("p1")
            defender_id = battle.opponent_active_pokemon.identifier("p2")
            def get_move_score(move):
                try:
                    max_damage, min_damage = calculate_damage_with_base_stats(attacker_id, defender_id, move, battle)
                    return (max_damage + min_damage)*0.5 * move.accuracy
                except:
                    max_damage, min_damage = calculate_damage_simple(battle.active_pokemon, battle.opponent_active_pokemon, move, battle) 
                    return (max_damage + min_damage)*0.5 * move.accuracy
            
            # Find best move and calculate damage
            # Usage in your main logic:
            lethal_move = find_best_lethal_move(battle, attacker_id, defender_id)
            
            # Check for guaranteed KO scenarios
            our_speed = battle.active_pokemon.stats["spe"] or battle.active_pokemon.base_stats["spe"]
            # Get opponent's current HP
            opp_stats = estimate_stats_from_base(battle.opponent_active_pokemon)
            opp_speed = opp_stats["spe"]
            opp_hp = battle.opponent_active_pokemon.current_hp_fraction * opp_stats["hp"]
            # Get type effectiveness of opponent's moves against us
            opp_eff = max([battle.active_pokemon.damage_multiplier(t) 
                        for t in battle.opponent_active_pokemon.types if t is not None])
            
            if lethal_move and our_speed > opp_speed:
                return self.create_order(lethal_move)
            if lethal_move and opp_eff <= 1.0:
                return self.create_order(lethal_move)
            
            best_move = max(battle.available_moves, key=lambda m: get_move_score(m))
            dmg = get_move_score(best_move)
            
            # Priority 1: Guaranteed KO and we're faster
            if dmg >= opp_hp and our_speed > opp_speed:
                return self.create_order(best_move)
            
            # Priority 2: Guaranteed KO and opponent isn't super effective against us
            if dmg >= opp_hp and opp_eff <= 1.0:
                return self.create_order(best_move)
            
            # Priority 3: Guaranteed 2HKO and we're faster
            if dmg * 2.0 >= opp_hp and opp_eff <= 1.0:
                return self.create_order(best_move)
            
            opp_pokemons = [mon for mon in battle.opponent_team.values() if not mon.fainted]
            if len(opp_pokemons) == 1:
                self.create_order(best_move)

            # If no switches available, have to use the move
            if not battle.available_switches:
                return self.create_order(best_move)
            
            # Evaluate switching options
            best_switch = max(battle.available_switches,
                            key=lambda s: estimate_matchup(s, battle.opponent_active_pokemon))
            
            current_matchup = estimate_matchup(battle.active_pokemon, battle.opponent_active_pokemon)
            switch_matchup = estimate_matchup(best_switch, battle.opponent_active_pokemon)
            
            # Priority 5: Bad matchup and switch gives good matchup
            if opp_eff >= 2.0 and our_speed < opp_speed:
                return self.create_order(best_switch)
            # Priority 6: Very low damage output and switch gives good matchup
            if dmg * 3 < opp_hp and current_matchup < switch_matchup and switch_matchup > 1 and current_matchup <= -1:
                return self.create_order(best_switch)
            
            # Default: use the best move
            return self.create_order(best_move)
        except:
            return self.choose_random_move(battle)