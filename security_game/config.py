"""Game parameters. See DESIGN.md, section "Parameters"."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    """Immutable (and hashable) parameter set for one game.

    values          server values v_i
    defense_units   D, integer defense units the defender splits over servers
    traps           T, number of servers that receive a trap each round
    attack_units    A, integer attack units the attacker may spend (hold-back allowed)
    resilience      k, contest constant in a / (a + d + k)
    catch_prob      c, probability an attack on a trapped server is caught
    caught_penalty  C, attacker loss when caught
    catch_reward    R, defender gain when it catches the attacker
    block_cost      b, attacker loss when an attack is blocked
    block_reward    r_b, defender gain when it blocks an attack
    effort_cost     kappa, attacker cost per attack unit spent
    memory          W, rounds the attacker remembers
    """

    values: tuple[float, ...] = (10.0, 8.0, 6.0, 4.0)
    defense_units: int = 4
    traps: int = 1
    attack_units: int = 4
    resilience: float = 1.0
    catch_prob: float = 1.0
    caught_penalty: float = 6.0
    catch_reward: float = 6.0
    block_cost: float = 1.0
    block_reward: float = 1.0
    effort_cost: float = 0.25
    memory: int = 50

    def __post_init__(self) -> None:
        # Coerce to a tuple of floats so the config stays hashable for caching.
        object.__setattr__(self, "values", tuple(float(v) for v in self.values))

        if len(self.values) == 0:
            raise ValueError("values must contain at least one server")
        if any(v < 0 for v in self.values):
            raise ValueError("server values must be non-negative")
        for name in ("defense_units", "attack_units", "traps", "memory"):
            if not isinstance(getattr(self, name), int):
                raise TypeError(f"{name} must be an int")
        if self.defense_units <= 0:
            raise ValueError("defense_units must be positive")
        if self.attack_units <= 0:
            raise ValueError("attack_units must be positive")
        if not 0 <= self.traps <= self.n:
            raise ValueError("traps must satisfy 0 <= traps <= number of servers")
        if self.resilience < 0:
            raise ValueError("resilience must be non-negative")
        if not 0.0 <= self.catch_prob <= 1.0:
            raise ValueError("catch_prob must be in [0, 1]")
        for name in ("caught_penalty", "catch_reward", "block_cost",
                     "block_reward", "effort_cost"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.memory <= 0:
            raise ValueError("memory must be positive")

    @property
    def n(self) -> int:
        """Number of servers."""
        return len(self.values)
