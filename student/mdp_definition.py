"""
student/mdp_definition.py — Define Your MDP Here
=================================================

Define the components of your Markov Decision Process:

    MDP = {S, A, P(s'|s,a), R(s,a,s'), γ}

You must implement:
    1. state_fn(raw)          — Convert observation → hashable state tuple
    2. reward_fn(s, a, s')    — Immediate reward for a transition
    3. terminal_fn(s, step)   — Whether the episode should end
    4. prior_transitions(s,a) — Starting transition matrix T(s,a,s')

The transition matrix is your model of the world. You start with a prior
that encodes your initial beliefs, then your agent updates it from real
observations using count-based estimation:

    T̂(s,a,s') = count(s,a,s') / count(s,a)

Your agents use BOTH Policy Iteration AND Value Iteration on this matrix to
compute an optimal policy via Bellman's equation:

    V^π(s) = R(s) + γ Σ_{s'} P(s'|s,π(s)) V^π(s')       (Bellman equation)
    V*(s)  = max_a [R(s) + γ Σ_{s'} P(s'|s,a) V*(s')]    (Bellman optimality)

Raw observation fields are documented in the README "Raw State Fields"
section. The action list is fetched at runtime — see `env.ACTION_NAMES`
and `env.num_actions`.
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# MDP PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

GAMMA = 0.99          # Discount factor
MAX_STEPS = 500       # Max steps per episode before truncation
NUM_EPISODES = 100    # Training episodes
BUCKET_SIZE = 10      # Grid bucketing (blocks per cell) for position discretization


LOG_ITEMS = {
    "oak_log", "spruce_log", "birch_log", "jungle_log", "acacia_log",
    "dark_oak_log", "mangrove_log", "cherry_log", "crimson_stem",
    "warped_stem",
}

WOODEN_TOOLS = (
    "wooden_pickaxe",
    "wooden_sword",
    "wooden_shovel",
    "wooden_hoe",
    "wooden_axe",
)


def _count_matching(inventory: dict, predicate) -> int:
    return sum(int(count) for item, count in inventory.items() if predicate(item))


def _bucket_count(value: int, cutoffs: tuple[int, ...]) -> int:
    for idx, cutoff in enumerate(cutoffs):
        if value <= cutoff:
            return idx
    return len(cutoffs)


def _wood_distance_bucket(raw: dict) -> int:
    """Nearest log in the 5x5 nearby grid: 0=center, 1, 2, or 3=not visible."""
    grid = raw.get("nearby_grid") or []
    best = None
    center = len(grid) // 2 if grid else 2
    for row_idx, row in enumerate(grid):
        for col_idx, block in enumerate(row or []):
            if block in LOG_ITEMS or str(block).endswith("_log") or str(block).endswith("_stem"):
                dist = max(abs(row_idx - center), abs(col_idx - center))
                best = dist if best is None else min(best, dist)
    if best is None:
        return 3
    return min(best, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. STATE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def state_fn(raw: dict) -> tuple:
    """
    Convert the raw observation into a hashable state tuple.

    Parameters
    ----------
    raw : dict
        Observation from the environment. See the README "Raw State Fields"
        section for the full list of keys. You can also call
        `env.get_raw_state()` to inspect a live dict.

    Returns
    -------
    tuple
        Hashable state representation. Must have the SAME shape every call.
    """
    inventory = raw.get("inventory") or {}

    log_count = _count_matching(
        inventory,
        lambda item: item in LOG_ITEMS or item.endswith("_log") or item.endswith("_stem"),
    )
    plank_count = _count_matching(inventory, lambda item: item.endswith("_planks"))
    stick_count = int(inventory.get("stick", 0))

    return (
        _bucket_count(log_count, (0, 5, 10)),       # 0, 1-5, 6-10, 11+
        _bucket_count(plank_count, (0, 20, 40)),    # 0, 1-20, 21-40, 41+
        _bucket_count(stick_count, (0, 5, 10)),     # 0, 1-5, 6-10, 11+
        bool(inventory.get("crafting_table", 0)),
        bool(raw.get("has_table_nearby", False)),
        *(bool(inventory.get(tool, 0)) for tool in WOODEN_TOOLS),
        _wood_distance_bucket(raw),
        bool(raw.get("water_adjacent", False)),
        int(raw.get("cliff_loc", 0) or 0),
        bool(raw.get("is_dead", False)),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. REWARD FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def reward_fn(old_state: tuple, action: int, new_state: tuple) -> float:
    """
    Compute immediate reward R(s, a, s') for a transition.

    Parameters
    ----------
    old_state : tuple — State before the action
    action    : int   — Action index (0 through env.num_actions-1)
    new_state : tuple — State after the action

    Returns
    -------
    float : Scalar reward.
    """
    if not old_state or not new_state:
        return 0.0

    reward = -0.05

    old_dead = old_state[13]
    new_dead = new_state[13]
    if new_dead and not old_dead:
        return -100.0

    old_log_bucket, old_plank_bucket, old_stick_bucket = old_state[:3]
    new_log_bucket, new_plank_bucket, new_stick_bucket = new_state[:3]

    if new_log_bucket > old_log_bucket:
        reward += 6.0 if new_log_bucket <= 2 else 1.0
    if new_plank_bucket > old_plank_bucket:
        reward += 5.0
    if new_stick_bucket > old_stick_bucket:
        reward += 5.0 if new_stick_bucket <= 2 else 1.0

    old_has_table_item, old_table_nearby = old_state[3], old_state[4]
    new_has_table_item, new_table_nearby = new_state[3], new_state[4]
    if new_has_table_item and not old_has_table_item:
        reward += 8.0
    if new_table_nearby and not old_table_nearby:
        reward += 15.0

    old_tools = old_state[5:10]
    new_tools = new_state[5:10]
    new_tool_count = sum(new_tools)
    old_tool_count = sum(old_tools)
    reward += 30.0 * max(0, new_tool_count - old_tool_count)
    if new_tool_count == len(WOODEN_TOOLS) and old_tool_count < len(WOODEN_TOOLS):
        reward += 120.0

    old_wood_dist, new_wood_dist = old_state[10], new_state[10]
    if new_wood_dist < old_wood_dist:
        reward += 1.0

    if new_state[11]:
        reward -= 0.25
    if new_state[12]:
        reward -= 0.25

    return reward


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TERMINAL FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def terminal_fn(state: tuple, step_count: int) -> bool:
    """
    Return True if the episode should end.

    Parameters
    ----------
    state      : tuple — Current state
    step_count : int   — Steps taken this episode
    """
    if step_count >= MAX_STEPS:
        return True
    if state and state[13]:
        return True
    return bool(state and all(state[5:10]))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PRIOR TRANSITION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def prior_transitions(state, action):
    """
    Prior transition model T(s, a) = [(probability, next_state), ...].

    Encodes your initial beliefs about the dynamics before any observation.
    The agent updates from real experience via T̂(s,a,s') = count(s,a,s') / count(s,a).

    Returns
    -------
    list of (float, tuple)
        Probabilities must sum to 1.0.
    """
    # Default: self-loop (nothing changes). Students override.
    return [(1.0, state)]
