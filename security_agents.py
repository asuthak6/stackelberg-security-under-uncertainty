"""SE 4457: Multiagent DDoS defense game. Pure simulation, no network traffic.

Two adaptive agents interact over abstract services. The attacker chooses a target
and traffic pressure level. The defender allocates a finite mitigation budget.
This code never opens sockets or sends packets.
"""
from __future__ import annotations
import argparse
import csv
import random
from dataclasses import dataclass, field
from pathlib import Path

VALUES = (10, 8, 6, 4)
BASE_CAPACITY = 5
INTENSITIES = (2, 4, 6)
BLOCK_PENALTY = 3

@dataclass
class DefenderAgent:
    budget: int = 4
    learning_rate: float = 0.25
    threat_weights: list[float] = field(default_factory=lambda: [1.0] * len(VALUES))

    def choose_action(self, rng: random.Random) -> list[int]:
        """Distribute mitigation units to high value, frequently attacked services."""
        allocations = [0] * len(VALUES)
        for _ in range(self.budget):
            # Diminishing return encourages spreading mitigation across services.
            scores = [VALUES[i] * self.threat_weights[i] / (1 + allocations[i]) for i in range(len(VALUES))]
            best = max(scores)
            candidates = [i for i, s in enumerate(scores) if abs(s - best) < 1e-10]
            allocations[rng.choice(candidates)] += 1
        return allocations

    def learn(self, target: int, intensity: int) -> None:
        """Update threat estimates using observed attack target and volume."""
        self.threat_weights = [(1 - self.learning_rate) * w for w in self.threat_weights]
        self.threat_weights[target] += self.learning_rate * (intensity / max(INTENSITIES))

@dataclass
class AttackerAgent:
    noise: float = 0.2
    epsilon: float = 0.1
    learning_rate: float = 0.3
    # Beliefs about mitigation units (not exact knowledge).
    beliefs: list[float] = field(default_factory=lambda: [1.0] * len(VALUES))

    def perceive(self, actual: list[int], rng: random.Random) -> list[float]:
        """Noisy abstract monitoring: observed mitigation may be +/- one unit."""
        seen = []
        for x in actual:
            if rng.random() < self.noise:
                seen.append(float(max(0, x + rng.choice((-1, 1)))))
            else:
                seen.append(float(x))
        return seen

    def choose_action(self, observed: list[float], rng: random.Random) -> tuple[int, int]:
        """Pick target and intensity to maximize estimated outage payoff minus cost."""
        if rng.random() < self.epsilon:
            return rng.randrange(len(VALUES)), rng.choice(INTENSITIES)
        estimates = [(1 - (1 - self.noise)) * prior + (1 - self.noise) * obs
                     for prior, obs in zip(self.beliefs, observed)]
        options = []
        for target, value in enumerate(VALUES):
            for intensity in INTENSITIES:
                # A service is estimated to fail only if load exceeds protected capacity.
                overloaded = intensity > BASE_CAPACITY + 2 * estimates[target]
                estimated_payoff = (value if overloaded else -BLOCK_PENALTY) - 0.35 * intensity
                options.append((estimated_payoff, target, intensity))
        best_score = max(x[0] for x in options)
        candidates = [(t, a) for score, t, a in options if abs(score - best_score) < 1e-10]
        return rng.choice(candidates)

    def learn(self, target: int, observed_mitigation: int) -> None:
        self.beliefs[target] = ((1 - self.learning_rate) * self.beliefs[target]
                                + self.learning_rate * observed_mitigation)


def simulate(rounds: int = 100, noise: float = 0.2, budget: int = 4,
             seed: int = 42, learning: bool = True) -> list[dict]:
    if rounds < 1 or budget < 0 or not 0 <= noise <= 1:
        raise ValueError("Use rounds >= 1, budget >= 0, and 0 <= noise <= 1")
    rng = random.Random(seed)
    defender = DefenderAgent(budget=budget)
    attacker = AttackerAgent(noise=noise)
    history = []
    for step in range(1, rounds + 1):
        allocation = defender.choose_action(rng)
        perception = attacker.perceive(allocation, rng)
        target, intensity = attacker.choose_action(perception, rng)
        capacity = BASE_CAPACITY + 2 * allocation[target]
        outage = intensity > capacity
        payoff = (VALUES[target] if outage else -BLOCK_PENALTY) - 0.35 * intensity
        history.append({
            "round": step,
            "defender_allocation": ",".join(map(str, allocation)),
            "perceived_allocation": ",".join(map(lambda x: f"{x:.0f}", perception)),
            "target": target + 1,
            "intensity": intensity,
            "service_capacity": capacity,
            "outage": int(outage),
            "attacker_payoff": round(payoff, 2),
            "defender_payoff": round(-payoff, 2),
        })
        if learning:
            # Abstract post-incident information: the attacked service's mitigation is revealed.
            attacker.learn(target, allocation[target])
            defender.learn(target, intensity)
    return history


def main():
    parser = argparse.ArgumentParser(description="Simulated multiagent DDoS game (no packets)")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-learning", action="store_true")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()
    history = simulate(args.rounds, args.noise, args.budget, args.seed, not args.no_learning)
    with Path(args.output).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    rate = sum(row["outage"] for row in history) / len(history)
    avg = sum(row["attacker_payoff"] for row in history) / len(history)
    print("Multiagent simulated DDoS defense game (no real traffic)")
    print(f"Rounds: {args.rounds}, noise: {args.noise}, defender budget: {args.budget}")
    print(f"Service outage rate: {rate:.1%}")
    print(f"Average attacker payoff: {avg:.2f}")
    print(f"Saved: {Path(args.output).resolve()}")
    print("First 5 rounds:")
    for row in history[:5]:
        print(f"  Round {row['round']}: mitigation [{row['defender_allocation']}], "
              f"target {row['target']}, load {row['intensity']}, capacity {row['service_capacity']}, "
              f"{'OUTAGE' if row['outage'] else 'DEFENDED'}")

if __name__ == "__main__":
    main()
