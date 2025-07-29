"""Simplified damage calculator that takes Pokemon objects directly"""

import math
from typing import Dict, List, Optional, Union

from poke_env.battle import (
    Battle,
    DoubleBattle,
    Effect,
    Field,
    Move,
    MoveCategory,
    Pokemon,
    PokemonGender,
    PokemonType,
    SideCondition,
    Status,
    Target,
    Weather,
)
from poke_env.data import GenData


def calculate_damage_simple(
    attacker: Pokemon,
    defender: Pokemon,
    move: Move,
    battle: Union[Battle, DoubleBattle],
    is_critical: bool = False,
):
    """Return the possible damage range for a move using Pokemon objects directly.

    :param attacker: Attacking Pokemon object
    :type attacker: Pokemon
    :param defender: Defending Pokemon object  
    :type defender: Pokemon
    :param move: Move being used
    :type move: Move
    :param battle: Current battle object
    :type battle: Battle or DoubleBattle
    :param is_critical: Whether to compute damage for a critical hit
    :type is_critical: bool
    :return: Tuple of minimum and maximum damage rolls
    :rtype: Tuple[int, int]
    """
    
    # Basic validation
    if not all(isinstance(x, (int, float)) for x in attacker.stats.values()):
        return 0, 0
    if not all(isinstance(x, (int, float)) for x in defender.stats.values()):
        return 0, 0

    move_category = move.category
    flags = {x: 1 for x in move.entry.get("flags", {})}

    # Handle special moves that change category
    if (
        move.id in ["photongeyser", "terablast"]
        and attacker.stats["atk"]
        and attacker.stats["spa"]
    ):
        move_category = (
            MoveCategory.PHYSICAL
            if attacker.stats["atk"] * BOOST_MULTIPLIERS[attacker.boosts["atk"]]
            > attacker.stats["spa"] * BOOST_MULTIPLIERS[attacker.boosts["spa"]]
            else MoveCategory.SPECIAL
        )

    # Status moves don't deal damage
    if move_category == MoveCategory.STATUS and not move.id == "naturepower":
        return 0, 0

    # Handle protect effects
    if any(map(lambda x: x in PROTECT_EFFECTS, defender.effects)):
        if not (move.breaks_protect or (attacker.ability == "unseenfist" and flags.get("contact", 0) == 1)):
            return 0, 0

    # Handle fixed damage moves
    if move.id in ["seismictoss", "nightshade"]:
        return attacker.level, attacker.level
    elif move.id == "dragonrage":
        return 40, 40
    elif move.id == "sonicboom":
        return 20, 20

    # Handle abilities that block damage
    defender_ability = defender.ability
    if move.id in MOVE_IGNORES_ABILITY or (
        attacker.ability in ATTACKER_IGNORES_ABILITY
        and not defender.item == "abilityshield"
    ):
        defender_ability = ""

    # Type effectiveness calculation
    move_type = get_move_type(move, attacker, battle)
    type_effectiveness = calculate_type_effectiveness(move, move_type, defender, attacker, battle)
    
    if type_effectiveness == 0:
        return 0, 0

    # Handle immunity abilities
    if is_immune_to_move(move, move_type, attacker, defender, battle, flags):
        return 0, 0

    # Calculate base power
    base_power = calculate_base_power_simple(attacker, defender, move, battle)
    if base_power == 0:
        return 0, 0

    # Calculate attack and defense stats
    attack = calculate_attack_simple(attacker, defender, move, is_critical)
    defense = calculate_defense_simple(attacker, defender, move, battle, is_critical)

    # Calculate base damage
    base_damage = calculate_base_damage_simple(
        attacker, defender, base_power, attack, defense, move, move_type, battle, is_critical
    )

    # Apply modifiers
    stab_mod = get_stab_mod_simple(attacker, move, move_type)
    
    apply_burn = (
        attacker.status == Status.BRN
        and move_category == MoveCategory.PHYSICAL
        and not attacker.ability == "guts"
        and not move.id == "facade"
    )

    final_mods = calculate_final_mods_simple(attacker, defender, move, battle, type_effectiveness, flags, is_critical)
    final_mod = chain_mods(final_mods, 41, 131072)

    return (
        get_final_damage(base_damage, 0, type_effectiveness, apply_burn, stab_mod, final_mod),
        get_final_damage(base_damage, 15, type_effectiveness, apply_burn, stab_mod, final_mod),
    )


def get_move_type(move: Move, attacker: Pokemon, battle: Union[Battle, DoubleBattle]) -> Optional[PokemonType]:
    """Determine the actual type of the move after transformations"""
    move_type = move.type
    
    # Weather Ball
    if move.id == "weatherball":
        if Weather.SUNNYDAY in battle.weather and attacker.item != "utilityumbrella":
            move_type = PokemonType.FIRE
        elif Weather.RAINDANCE in battle.weather and attacker.item != "utilityumbrella":
            move_type = PokemonType.WATER
        elif Weather.SANDSTORM in battle.weather:
            move_type = PokemonType.ROCK
        elif Weather.HAIL in battle.weather or Weather.SNOW in battle.weather:
            move_type = PokemonType.ICE
        else:
            move_type = PokemonType.NORMAL
    
    # Tera Blast when terastallized
    elif move.id == "terablast" and attacker.is_terastallized:
        move_type = attacker.type_1
    
    # Handle -ate abilities
    no_type_change = move.id in {
        "revelationdance", "judgment", "naturepower", "technoblast", 
        "multiattack", "naturalgift", "weatherball", "terrainpulse", "struggle"
    } or (move.id == "terablast" and attacker.is_terastallized)

    if not no_type_change:
        if attacker.ability == "aerilate" and move_type == PokemonType.NORMAL:
            move_type = PokemonType.FLYING
        elif attacker.ability == "galvanize" and move_type == PokemonType.NORMAL:
            move_type = PokemonType.ELECTRIC
        elif attacker.ability == "pixilate" and move_type == PokemonType.NORMAL:
            move_type = PokemonType.FAIRY
        elif attacker.ability == "refrigerate" and move_type == PokemonType.NORMAL:
            move_type = PokemonType.ICE
    
    return move_type


def calculate_type_effectiveness(move: Move, move_type: Optional[PokemonType], defender: Pokemon, 
                               attacker: Pokemon, battle: Union[Battle, DoubleBattle]) -> float:
    """Calculate type effectiveness"""
    if not move_type:
        return 1.0
        
    first_effectiveness = get_move_effectiveness(
        move, move_type, defender.type_1,
        attacker.ability in ["scrappy", "mindseye"],
        Field.GRAVITY in battle.fields,
        defender.item == "ringtarget"
    )
    
    second_effectiveness = 1.0
    if len(defender.types) > 1:
        second_effectiveness = get_move_effectiveness(
            move, move_type, defender.type_2,
            attacker.ability in ["scrappy", "mindseye"], 
            Field.GRAVITY in battle.fields,
            defender.item == "ringtarget"
        )
    
    return first_effectiveness * second_effectiveness


def is_immune_to_move(move: Move, move_type: Optional[PokemonType], attacker: Pokemon, 
                     defender: Pokemon, battle: Union[Battle, DoubleBattle], flags: Dict[str, int]) -> bool:
    """Check if defender is immune to the move"""
    if not move_type:
        return False
        
    defender_ability = defender.ability
    
    # Type immunities via abilities
    immunity_checks = [
        (move_type == PokemonType.GRASS and defender_ability == "sapsipper"),
        (move_type == PokemonType.FIRE and defender_ability in ["flashfire", "wellbakedbody"]),
        (move_type == PokemonType.WATER and defender_ability in ["dryskin", "stormdrain", "waterabsorb"]),
        (move_type == PokemonType.ELECTRIC and defender_ability in ["lightningrod", "motordrive", "voltabsorb"]),
        (move_type == PokemonType.GROUND and defender_ability == "levitate" and 
         Field.GRAVITY not in battle.fields and move.id != "thousandarrows" and defender.item != "ironball"),
        (flags.get("bullet", 0) == 1 and defender_ability == "bulletproof"),
        (flags.get("sound", 0) == 1 and defender_ability == "soundproof" and move.id != "clangoroussoul"),
        (move.priority > 0 and defender_ability in ["queenlymajesty", "dazzling", "armortail"]),
    ]
    
    return any(immunity_checks)


def calculate_base_power_simple(attacker: Pokemon, defender: Pokemon, move: Move, 
                              battle: Union[Battle, DoubleBattle]) -> float:
    """Simplified base power calculation"""
    base_power = move.base_power * 1.0
    
    if base_power == 0:
        return 0
    
    # Handle some common base power modifications
    if move.id == "hex" and (defender.status is not None or defender.ability == "comatose"):
        base_power *= 2
    elif move.id == "facade" and attacker.status in {Status.BRN, Status.PAR, Status.PSN, Status.TOX}:
        base_power *= 2
    elif move.id == "brine" and defender.current_hp_fraction <= 0.5:
        base_power *= 2
    elif move.id == "acrobatics" and (attacker.item == "flyinggem" or attacker.item is None):
        base_power *= 2
    elif move.id == "weatherball" and len(battle.weather) > 0:
        base_power *= 2
        if attacker.item == "utilityumbrella" and (
            Weather.SUNNYDAY in battle.weather or Weather.RAINDANCE in battle.weather
        ):
            base_power = move.base_power
    
    # Apply basic modifiers
    bp_mods = []
    
    # Technician
    if attacker.ability == "technician" and base_power <= 60:
        bp_mods.append(6144)
    
    # STAB items (simplified)
    if attacker.item and move.type and (attacker.item.replace(" gem", "") == move.type.name.lower()):
        bp_mods.append(5325)
    
    if bp_mods:
        base_power = max(1, poke_round(base_power * chain_mods(bp_mods, 41, 2097152) / 4096))
    
    return base_power


def calculate_attack_simple(attacker: Pokemon, defender: Pokemon, move: Move, is_critical: bool = False) -> float:
    """Simplified attack calculation"""
    attack_stat = "atk" if move.category == MoveCategory.PHYSICAL else "spa"
    if move.id == "bodypress":
        attack_stat = "def"
    
    attack_source = attacker if move.id != "foulplay" else defender
    
    if (
        attack_source.boosts[attack_stat] == 0
        or (is_critical and attack_source.boosts[attack_stat] < 0)
        or defender.ability == "unaware"
    ):
        attack = attack_source.stats[attack_stat]
    else:
        attack = math.floor(
            attack_source.stats[attack_stat] * BOOST_MULTIPLIERS[attack_source.boosts[attack_stat]]
        )
    
    # Hustle
    if attacker.ability == "hustle" and move.category == MoveCategory.PHYSICAL:
        attack = poke_round((attack * 3) / 2)
    
    # Basic attack modifiers
    atk_mods = []
    
    # Guts
    if (attacker.ability == "guts" and attacker.status and move.category == MoveCategory.PHYSICAL):
        atk_mods.append(6144)
    
    # Choice items
    if ((attacker.item == "choiceband" and move.category == MoveCategory.PHYSICAL) or
        (attacker.item == "choicespecs" and move.category == MoveCategory.SPECIAL)):
        atk_mods.append(6144)
    
    if atk_mods:
        attack = max(1, poke_round((attack * chain_mods(atk_mods, 410, 131072)) / 4096))
    
    return attack


def calculate_defense_simple(attacker: Pokemon, defender: Pokemon, move: Move, 
                           battle: Union[Battle, DoubleBattle], is_critical: bool = False) -> float:
    """Simplified defense calculation"""
    hits_physical = move.category == MoveCategory.PHYSICAL
    defense_stat = "def" if hits_physical else "spd"
    
    defense = math.floor(defender.stats[defense_stat] * BOOST_MULTIPLIERS[defender.boosts[defense_stat]])
    if (
        defender.boosts[defense_stat] == 0
        or (is_critical and defender.boosts[defense_stat] > 0)
        or move.ignore_defensive
        or attacker.ability == "unaware"
    ):
        defense = defender.stats[defense_stat]
    
    # Sandstorm SpD boost for Rock types
    if (Weather.SANDSTORM in battle.weather and PokemonType.ROCK in defender.types and not hits_physical):
        defense = poke_round((defense * 3) / 2)
    
    # Snow Defense boost for Ice types  
    if (Weather.SNOW in battle.weather and PokemonType.ICE in defender.types and hits_physical):
        defense = poke_round((defense * 3) / 2)
    
    return max(1, defense)


def calculate_base_damage_simple(attacker: Pokemon, defender: Pokemon, base_power: float, 
                               attack: float, defense: float, move: Move, move_type: Optional[PokemonType],
                               battle: Union[Battle, DoubleBattle], is_critical: bool = False) -> int:
    """Calculate base damage"""
    base_damage = math.floor(
        math.floor((math.floor(2 * attacker.level / 5 + 2) * base_power) * attack / defense) / 50 + 2
    )
    
    # Weather effects
    if defender.item != "utilityumbrella":
        if (Weather.SUNNYDAY in battle.weather and move_type == PokemonType.FIRE) or (
            Weather.RAINDANCE in battle.weather and move_type == PokemonType.WATER
        ):
            base_damage = poke_round((base_damage * 6144) / 4096)
        elif (Weather.SUNNYDAY in battle.weather and move_type == PokemonType.WATER) or (
            Weather.RAINDANCE in battle.weather and move_type == PokemonType.FIRE
        ):
            base_damage = poke_round((base_damage * 2048) / 4096)
    
    # Critical hit
    if is_critical:
        base_damage = poke_round(base_damage * 1.5)
    
    return base_damage


def calculate_final_mods_simple(attacker: Pokemon, defender: Pokemon, move: Move,
                              battle: Union[Battle, DoubleBattle], type_effectiveness: float,
                              flags: Dict[str, int], is_critical: bool = False) -> List[int]:
    """Calculate final damage modifiers"""
    final_mods = []
    
    # Expert Belt
    if attacker.item == "expertbelt" and type_effectiveness > 1:
        final_mods.append(4915)
    
    # Life Orb
    if attacker.item == "lifeorb":
        final_mods.append(5324)
    
    # Solid Rock/Filter
    if (defender.ability in ["solidrock", "filter", "prismarmor"] and type_effectiveness > 1):
        final_mods.append(3072)
    
    # Multiscale/Shadow Shield
    if (defender.ability in ["multiscale", "shadowshield"] and defender.current_hp_fraction == 1):
        final_mods.append(2048)
    
    return final_mods


def get_stab_mod_simple(pokemon: Pokemon, move: Move, move_type: Optional[PokemonType]) -> int:
    """Calculate STAB modifier"""
    stab_mod = 4096
    
    if move.id == "struggle" or not move_type:
        return stab_mod
    
    if move_type in pokemon.original_types:
        stab_mod += 2048
    elif pokemon.ability in ["protean", "libero"] and not pokemon.is_terastallized:
        stab_mod += 2048
    
    # Tera STAB
    if (pokemon.tera_type == move_type and pokemon.tera_type != PokemonType.STELLAR and pokemon.is_terastallized):
        stab_mod += 2048
    
    # Adaptability
    if pokemon.ability == "adaptability" and move_type in pokemon.types:
        stab_mod += 2048
    
    return stab_mod


# Import the utility functions from the original file
def poke_round(num: float):
    return math.ceil(num) if num % 1 > 0.5 else math.floor(num)


def get_final_damage(base_amount: int, i: int, effectiveness: float, is_burned: bool, stab_mod: int, final_mod: int) -> float:
    damage_amount = math.floor(base_amount * (85 + i) / 100) * 1.0
    if stab_mod != 4096:
        damage_amount = (damage_amount * stab_mod) / 4096
    damage_amount = math.floor(poke_round(damage_amount) * effectiveness)
    if is_burned:
        damage_amount = math.floor(damage_amount / 2)
    return poke_round(max(1, (damage_amount * final_mod) / 4096))


def chain_mods(mods: List[int], lb: int, ub: int):
    m = 4096
    for mod in mods:
        if mod != 4096:
            m = (m * mod + 2048) >> 12
    return max(min(m, ub), lb)


def get_move_effectiveness(move: Move, move_type: PokemonType, type: PokemonType, 
                         is_ghost_revealed: Optional[bool] = False, is_gravity: Optional[bool] = False,
                         is_ring_target: Optional[bool] = False) -> float:
    if move.id == "struggle":
        return 1
    elif move.id == "freezedry" and type == PokemonType.WATER:
        return 2
    else:
        effectiveness = PokemonType.damage_multiplier(move_type, type, type_chart=GenData.from_gen(9).type_chart)
        if effectiveness == 0 and is_ring_target:
            effectiveness = 1
        return effectiveness


# Constants from original file
PROTECT_EFFECTS = {
    Effect.PROTECT, Effect.SPIKY_SHIELD, Effect.KINGS_SHIELD, Effect.BANEFUL_BUNKER,
    Effect.BURNING_BULWARK, Effect.OBSTRUCT, Effect.MAX_GUARD, Effect.SILK_TRAP,
}

ATTACKER_IGNORES_ABILITY = {"moldbreaker", "teravolt", "turboblaze"}
MOVE_IGNORES_ABILITY = {"moongeistbeam", "photongeyser", "sunsteelstrike"}

BOOST_MULTIPLIERS = {
    -6: 2.0 / 8, -5: 2.0 / 7, -4: 2.0 / 6, -3: 2.0 / 5, -2: 2.0 / 4, -1: 2.0 / 3,
    0: 1, 1: 1.5, 2: 2, 3: 2.5, 4: 3, 5: 3.5, 6: 4,
}