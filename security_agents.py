
"""
SE 4457: Foundations of Multiagent Systems
Project: Attacker versus Defender Cybersecurity Game

PROJECT OVERVIEW
================

This project simulates a cybersecurity game involving two autonomous agents:

1. DEFENDER AGENT
   Protects valuable network targets using a limited security budget.
   Learns which targets the attacker prefers and adjusts its defenses.

2. ATTACKER AGENT
   Attempts to attack valuable targets while avoiding defended ones.
   Does not have perfect information about the defender's decisions.
   Uses noisy observations and previous experiences to make decisions.

The goal is to investigate how imperfect perception affects the attacker's
decision making and overall performance.

The simulation runs for multiple rounds, allowing both agents to adapt.

Requirements: Python 3.10 or newer.
No external libraries required.
"""

from __future__ import annotations

# argparse lets us change simulation settings directly from the terminal.
# Example: python security_agents.py --rounds 1000 --noise 0.4
import argparse

# csv allows us to save experiment results in a spreadsheet format.
import csv

# random is used for observations, exploration, and tie breaking.
import random

# dataclass automatically generates constructors for our agent classes.
# For example, DefenderAgent(budget=2).
from dataclasses import dataclass

# Path helps us create and locate the results file.
from pathlib import Path


# ============================================================
# 1. ENVIRONMENT CONFIGURATION
# ============================================================

# Our cybersecurity environment contains four possible targets.
#
# Imagine these are four servers in a company's network.
#
# Target 1 has a value of 10.
# Target 2 has a value of 8.
# Target 3 has a value of 6.
# Target 4 has a value of 4.
#
# Higher values represent more valuable targets.
#
# If the attacker successfully attacks Target 1,
# they receive 10 points.
#
# If they successfully attack Target 4,
# they receive only 4 points.
#
# IMPORTANT:
# Python indexes start at 0.
#
# Internally:
# Index 0 = Target 1
# Index 1 = Target 2
# Index 2 = Target 3
# Index 3 = Target 4
#
# We convert these indexes to normal target numbers when
# displaying results.

TARGET_VALUES = (10, 8, 6, 4)


# ============================================================
# 2. DEFENDER AGENT
# ============================================================

@dataclass
class DefenderAgent:
    """
    Represents the agent responsible for protecting the network.

    OBJECTIVE:
    Prevent the attacker from successfully attacking valuable targets.

    CONSTRAINT:
    The defender cannot protect every target because its
    security budget is limited.

    INTELLIGENCE:
    The defender remembers previous attacks and uses that
    information to decide which targets deserve protection.

    This is an adaptive agent because its future decisions
    depend on previous interactions.
    """

    # Number of targets the defender can protect per round.
    #
    # With budget=2, the defender protects 2 of the 4 targets.
    budget: int = 2

    # Controls how quickly the defender adapts to new attacks.
    #
    # A larger learning rate gives recent attacks more influence.
    #
    # A smaller learning rate makes the defender change
    # its behaviour more gradually.
    learning_rate: float = 0.25

    def __post_init__(self):
        """
        Runs automatically after the DefenderAgent is created.

        Initializes the defender's knowledge of attack patterns.
        """

        # Check that the budget is valid.
        #
        # We cannot protect a negative number of targets,
        # or more targets than actually exist.
        if not 0 <= self.budget <= len(TARGET_VALUES):
            raise ValueError(
                "Budget must be between 0 and the number of targets"
            )

        # Initially, the defender has no attack history.
        #
        # We assign every target a starting score of 1.0.
        #
        # These are initial weights, NOT actual attack counts
        # or probabilities.
        #
        # Starting equally means no target has an initial
        # advantage based on previous attack observations.
        #
        # Example:
        # [1.0, 1.0, 1.0, 1.0]
        self.attack_counts = [1.0] * len(TARGET_VALUES)

    def choose_action(self, rng: random.Random) -> set[int]:
        """
        Decide which targets to defend during the current round.

        The defender considers two factors:

        1. How valuable each target is.
        2. How frequently or recently that target has been attacked.

        A valuable target with a high attack tendency
        should receive greater protection priority.

        Returns:
            A set containing the indexes of protected targets.
        """

        # Calculate a protection priority score for each target.
        #
        # Formula:
        #
        # Priority = Target Value * Attack Weight
        #
        # Suppose the attack weights are:
        # [1.0, 2.0, 0.5, 1.0]
        #
        # Target values:
        # [10, 8, 6, 4]
        #
        # Priority scores:
        #
        # Target 1: 10 * 1.0 = 10
        # Target 2: 8 * 2.0 = 16
        # Target 3: 6 * 0.5 = 3
        # Target 4: 4 * 1.0 = 4
        #
        # The defender would prioritize Target 2 and Target 1.
        #
        # The scores rank expected danger.
        # They are not normalized probabilities.

        scores = [
            value * count
            for value, count in zip(
                TARGET_VALUES,
                self.attack_counts
            )
        ]

        # Rank targets from highest priority to lowest.
        #
        # range(len(scores)) produces target indexes:
        # [0, 1, 2, 3]
        #
        # sorted() orders them according to their scores.
        #
        # reverse=True means highest scores come first.
        #
        # rng.random() adds a random tie breaker so equally
        # scored targets do not always follow index order.
        ranked = sorted(
            range(len(scores)),
            key=lambda i: (scores[i], rng.random()),
            reverse=True
        )

        # Select the highest ranked targets until the
        # defender's security budget is exhausted.
        #
        # Example:
        # ranked = [1, 0, 3, 2]
        # budget = 2
        #
        # Result:
        # {1, 0}
        #
        # This means Targets 2 and 1 are protected.
        return set(ranked[:self.budget])

    def learn(self, attacked_target: int) -> None:
        """
        Update the defender's attack weights after a round.

        The defender learns which target was attacked,
        regardless of whether the attack succeeded.

        Recently attacked targets receive more attention.
        """

        # Gradually reduce all previously stored weights.
        #
        # Formula:
        #
        # New Weight = Old Weight * (1 - Learning Rate)
        #
        # With learning_rate = 0.25:
        #
        # Old weight: 1.0
        # New weight: 1.0 * 0.75 = 0.75
        #
        # This is called memory decay.
        #
        # It prevents very old attacks from permanently
        # dominating the defender's decisions.

        self.attack_counts = [
            (1 - self.learning_rate) * x
            for x in self.attack_counts
        ]

        # Increase the weight of the target that was
        # attacked in the current round.
        #
        # For example, if Target 3 was attacked,
        # its corresponding weight receives +0.25.
        #
        # Over time, the defender adjusts protection
        # priorities based on the attacker's behaviour.
        self.attack_counts[attacked_target] += self.learning_rate


# ============================================================
# 3. ATTACKER AGENT
# ============================================================

@dataclass
class AttackerAgent:
    """
    Represents an autonomous attacker trying to compromise
    valuable targets.

    OBJECTIVE:
    Maximize the reward earned from successful attacks.

    CHALLENGE:
    The attacker does not know exactly which targets
    are currently protected.

    PERCEPTION:
    The attacker receives observations that may be incorrect.

    REASONING:
    The attacker combines observations with its previous
    beliefs to estimate which targets are vulnerable.

    LEARNING:
    After attacking, it updates its belief about whether
    that target is typically protected.
    """

    # Probability that an observation is incorrect.
    #
    # noise = 0.0:
    # All observations are correct.
    #
    # noise = 0.2:
    # Each target's observed protection status has a
    # 20% probability of being flipped.
    #
    # noise = 0.5:
    # Observations are equally likely to be correct or wrong.
    #
    # noise = 1.0:
    # Every observation is inverted.
    #
    # The agent's decision rule accounts for known
    # noise levels above 0.5 by reversing the signal.
    noise: float = 0.2

    # Probability the attacker chooses a random target
    # instead of its currently preferred target.
    #
    # exploration = 0.10 means:
    #
    # Approximately 10% of rounds involve random exploration.
    #
    # The remaining rounds use calculated expected payoffs.
    #
    # Exploration allows the attacker to occasionally test
    # targets it might otherwise avoid.
    exploration: float = 0.10

    # Controls how quickly the attacker updates its beliefs.
    learning_rate: float = 0.30

    def __post_init__(self):
        """
        Initialize the attacker's knowledge and validate settings.
        """

        # Noise and exploration must both be valid probabilities.
        if not 0 <= self.noise <= 1 or not 0 <= self.exploration <= 1:
            raise ValueError(
                "Noise and exploration must be between 0 and 1"
            )

        # The attacker maintains an estimated probability
        # that each target is defended.
        #
        # Initially:
        #
        # Target 1: 50%
        # Target 2: 50%
        # Target 3: 50%
        # Target 4: 50%
        #
        # 0.5 represents uncertainty.
        #
        # The attacker has not yet collected evidence about
        # how the defender allocates its resources.
        self.defense_beliefs = [0.5] * len(TARGET_VALUES)

    def perceive(
        self,
        actual_defenses: set[int],
        rng: random.Random
    ) -> list[bool]:
        """
        Observe the defender's security configuration.

        However, these observations are imperfect.

        The attacker might believe:
        A protected target is unprotected, or vice versa.

        This implements the perception component of our project.
        """

        observations = []

        # Examine each target independently.
        for i in range(len(TARGET_VALUES)):

            # Determine whether this target is actually protected.
            #
            # True means protected.
            # False means unprotected.
            observed = i in actual_defenses

            # Generate a random number between 0 and 1.
            #
            # If the number is less than the noise level,
            # flip the observation.
            #
            # Example:
            # Actual = Protected (True)
            # Noise event occurs
            # Perceived = Unprotected (False)
            if rng.random() < self.noise:
                observed = not observed

            # Save the attacker's perception of this target.
            observations.append(observed)

        # Example output:
        #
        # [True, False, True, False]
        #
        # According to the attacker:
        #
        # Target 1 is protected.
        # Target 2 is unprotected.
        # Target 3 is protected.
        # Target 4 is unprotected.
        #
        # This information may not match reality.
        return observations

    def choose_action(
        self,
        observations: list[bool],
        rng: random.Random
    ) -> int:
        """
        Choose which target to attack.

        The attacker tries to maximize expected reward.

        It considers:
        1. The value of each target.
        2. Its estimated probability of being defended.
        3. The penalty for a blocked attack.

        It can also randomly explore.
        """

        # STEP 1: EXPLORATION
        #
        # With 10% probability, choose a random target.
        #
        # This allows the attacker to explore potential
        # opportunities rather than always choosing
        # its current best option.
        if rng.random() < self.exploration:
            return rng.randrange(len(TARGET_VALUES))

        # STEP 2: ESTIMATE OBSERVATION RELIABILITY
        #
        # Formula:
        #
        # reliability = abs(1 - 2 * noise)
        #
        # Examples:
        #
        # noise=0.0 -> reliability=1.0
        # noise=0.2 -> reliability=0.6
        # noise=0.5 -> reliability=0.0
        # noise=1.0 -> reliability=1.0
        #
        # At 50% noise, the observation provides no useful
        # information about protection.
        #
        # At 100% noise, every observation is inverted,
        # so reversing it reveals the true status.
        reliability = abs(1 - 2 * self.noise)

        estimates = []

        # STEP 3: COMBINE PREVIOUS BELIEFS WITH NEW OBSERVATIONS
        #
        # For each target, calculate an estimated probability
        # that it is protected.
        for prior, seen in zip(
            self.defense_beliefs,
            observations
        ):

            # Convert the observation into a number.
            #
            # When noise <= 0.5:
            #
            # Protected = 1.0
            # Unprotected = 0.0
            #
            # When noise > 0.5:
            #
            # We invert the observation because it is more
            # likely to be wrong than correct.
            sensed = (
                float(seen)
                if self.noise <= 0.5
                else float(not seen)
            )

            # Combine historical belief with current perception.
            #
            # Formula:
            #
            # Estimate =
            # (1 - reliability) * prior
            # + reliability * sensed
            #
            # Example:
            #
            # Previous belief = 0.5
            # Noise = 0.2
            # Reliability = 0.6
            # Observation = protected (1.0)
            #
            # Estimate:
            # 0.4 * 0.5 + 0.6 * 1.0
            # = 0.2 + 0.6
            # = 0.8
            #
            # The attacker estimates an 80% probability
            # that this target is protected.
            #
            # NOTE:
            # This is a weighted heuristic, not a formal
            # Bayesian probability update.
            estimates.append(
                (1 - reliability) * prior
                + reliability * sensed
            )

        # STEP 4: CALCULATE EXPECTED PAYOFF
        #
        # Formula:
        #
        # Expected Payoff =
        # Probability of Success * Target Value
        # - Probability of Being Blocked * Penalty
        #
        # Since p is the estimated probability of defense:
        #
        # Probability of Success = 1 - p
        #
        # Therefore:
        #
        # Expected Payoff = (1 - p) * value - p * 5
        #
        # Example:
        #
        # Target value = 10
        # Estimated protection probability = 0.8
        #
        # Expected Payoff:
        # (1 - 0.8) * 10 - 0.8 * 5
        # = 2 - 4
        # = -2
        #
        # The attacker should probably avoid this target.
        #
        # Another example:
        #
        # Target value = 8
        # Estimated protection probability = 0.1
        #
        # Expected Payoff:
        # 0.9 * 8 - 0.1 * 5
        # = 7.2 - 0.5
        # = 6.7
        #
        # Target 2 is much more attractive.

        scores = [
            (1 - p) * value - p * 5
            for p, value in zip(
                estimates,
                TARGET_VALUES
            )
        ]

        # STEP 5: PICK THE BEST TARGET
        #
        # Select the target with the highest
        # calculated expected payoff.
        #
        # rng.random() randomly resolves equal scores.
        #
        # The attacker does not directly know whether
        # this choice will succeed.
        return max(
            range(len(scores)),
            key=lambda i: (scores[i], rng.random())
        )

    def learn(
        self,
        attacked_target: int,
        was_defended: bool
    ) -> None:
        """
        Update the attacker's belief after observing
        the result of its attack.

        The attacker learns the true protection status
        of the target it attacked.

        It does not automatically learn the protection
        status of the other three targets.
        """

        # Retrieve the previous belief for this target.
        old = self.defense_beliefs[attacked_target]

        # Update the belief using an exponential moving average.
        #
        # Formula:
        #
        # New Belief =
        # (1 - Learning Rate) * Old Belief
        # + Learning Rate * Actual Observation
        #
        # Example:
        #
        # Previous belief = 0.5
        # Learning rate = 0.3
        # Target was protected = True (1)
        #
        # New belief:
        # 0.7 * 0.5 + 0.3 * 1
        # = 0.35 + 0.3
        # = 0.65
        #
        # The attacker now estimates a 65% probability
        # that this target is typically protected.
        #
        # This belief will influence its future decisions.
        self.defense_beliefs[attacked_target] = (
            (1 - self.learning_rate) * old
            + self.learning_rate * int(was_defended)
        )


# ============================================================
# 4. MAIN SIMULATION
# ============================================================

def simulate(
    rounds: int,
    noise: float,
    budget: int,
    seed: int,
    learning: bool = True
):
    """
    Run the attacker versus defender game.

    Each round follows this sequence:

    1. Defender chooses targets to protect.
    2. Attacker receives noisy observations.
    3. Attacker decides which target to attack.
    4. The environment determines success or failure.
    5. Both agents receive feedback.
    6. Both agents update their strategies if learning is enabled.

    This repeated interaction is the core multiagent simulation.
    """

    # Make sure at least one round is requested.
    if rounds < 1:
        raise ValueError("Rounds must be positive")

    # Create a random number generator with a fixed seed.
    #
    # This makes experiments reproducible.
    #
    # Running the same settings with seed=42 produces
    # the same random sequence and simulation results.
    rng = random.Random(seed)

    # Create the two agents.
    #
    # Each agent maintains its own internal information
    # and uses its own decision making process.
    defender = DefenderAgent(budget=budget)
    attacker = AttackerAgent(noise=noise)

    # Store the results of every round.
    history = []

    # Run the game for the requested number of rounds.
    for round_number in range(1, rounds + 1):

        # STEP 1: DEFENDER CHOOSES PROTECTION
        #
        # Example:
        # defended = {0, 1}
        #
        # Targets 1 and 2 are protected.
        defended = defender.choose_action(rng)

        # STEP 2: ATTACKER PERCEIVES THE DEFENSES
        #
        # This information might be incorrect
        # because of observation noise.
        observed = attacker.perceive(defended, rng)

        # STEP 3: ATTACKER SELECTS A TARGET
        #
        # Uses noisy observations, learned beliefs,
        # expected payoff, and occasional exploration.
        choice = attacker.choose_action(observed, rng)

        # STEP 4: CHECK ATTACK OUTCOME
        #
        # If the attacked target is protected,
        # the attack is blocked.
        #
        # Otherwise, the attack succeeds.
        blocked = choice in defended

        # STEP 5: CALCULATE ATTACKER PAYOFF
        #
        # Blocked attack:
        # Attacker loses 5 points.
        #
        # Successful attack:
        # Attacker receives the target's value.
        attacker_payoff = (
            -5 if blocked else TARGET_VALUES[choice]
        )

        # STEP 6: CALCULATE DEFENDER PAYOFF
        #
        # We use a zero sum game.
        #
        # Whatever the attacker gains,
        # the defender loses.
        #
        # Example:
        #
        # Attacker payoff = +10
        # Defender payoff = -10
        #
        # Attacker payoff = -5
        # Defender payoff = +5
        #
        # This represents directly conflicting objectives.
        defender_payoff = -attacker_payoff

        # STEP 7: RECORD THE ROUND
        #
        # The dictionary stores information needed
        # for analyzing and comparing agent performance.
        history.append({
            "round": round_number,

            # Display protected targets using normal
            # numbering rather than Python indexes.
            "defended_targets": ",".join(
                str(i + 1) for i in sorted(defended)
            ) or "none",

            # Store the attacker's perceived protection map.
            #
            # 1 means perceived protected.
            # 0 means perceived unprotected.
            #
            # Example: 1010
            "attacker_observation": "".join(
                "1" if x else "0"
                for x in observed
            ),

            # Which target the attacker selected.
            "attacked_target": choice + 1,

            # 1 means blocked.
            # 0 means successful.
            "blocked": int(blocked),

            # Rewards for both agents.
            "attacker_payoff": attacker_payoff,
            "defender_payoff": defender_payoff,

            # Attacker's belief BEFORE learning from this round.
            #
            # round(..., 3) displays three decimal places.
            "attacker_belief_target": round(
                attacker.defense_beliefs[choice], 3
            ),
        })

        # STEP 8: LEARNING
        #
        # If learning is enabled, both agents
        # update their internal information.
        #
        # The attacker learns whether the target
        # it selected was actually defended.
        #
        # The defender learns which target was attacked.
        #
        # These updates influence future rounds.
        if learning:
            attacker.learn(choice, blocked)
            defender.learn(choice)

    # Return all recorded results for analysis.
    return history


# ============================================================
# 5. RESULTS ANALYSIS
# ============================================================

def summarize(history):
    """
    Calculate overall performance statistics.

    We measure:
    1. Total number of rounds.
    2. Percentage of successful attacks.
    3. Average attacker payoff.

    These metrics allow us to compare experiments.
    """

    # Count the number of rounds.
    n = len(history)

    # Count the successful attacks.
    #
    # blocked=0 means the attack succeeded.
    #
    # Divide by the total number of rounds to
    # calculate the attack success rate.
    #
    # Example:
    # 70 successes / 100 rounds = 0.70 = 70%
    success_rate = (
        sum(not row["blocked"] for row in history) / n
    )

    # Calculate average attacker payoff.
    #
    # Example:
    # Total attacker payoff = 338
    # Number of rounds = 100
    #
    # Average payoff = 338 / 100 = 3.38
    average_payoff = (
        sum(row["attacker_payoff"] for row in history) / n
    )

    return {
        "rounds": n,
        "attack_success_rate": success_rate,
        "average_attacker_payoff": average_payoff,
    }


# ============================================================
# 6. PROGRAM ENTRY POINT
# ============================================================

def main():
    """
    Read simulation settings from the terminal,
    run the game, save results, and display a summary.
    """

    # Create a command line argument parser.
    #
    # This allows us to experiment with different settings
    # without modifying the source code.
    p = argparse.ArgumentParser(
        description="Adaptive attacker and defender multiagent game"
    )

    # Number of simulated attack rounds.
    # Default = 100.
    p.add_argument(
        "--rounds",
        type=int,
        default=100
    )

    # Probability of incorrect perception.
    # Default = 20%.
    p.add_argument(
        "--noise",
        type=float,
        default=0.2
    )

    # Number of targets defender can protect.
    # Default = 2.
    p.add_argument(
        "--budget",
        type=int,
        default=2
    )

    # Fixed random seed for reproducibility.
    p.add_argument(
        "--seed",
        type=int,
        default=42
    )

    # Option to disable both agents' learning updates.
    #
    # Useful for comparing adaptive agents against
    # agents with unchanged initial beliefs.
    #
    # Note: Even without learning, agents still make
    # decisions using current observations.
    p.add_argument(
        "--no-learning",
        action="store_true"
    )

    # Filename for the experiment results.
    p.add_argument(
        "--output",
        default="results.csv"
    )

    # Read command line arguments.
    args = p.parse_args()

    # Run the simulation.
    #
    # If --no-learning was provided, learning becomes False.
    history = simulate(
        args.rounds,
        args.noise,
        args.budget,
        args.seed,
        not args.no_learning
    )

    # Determine where the results will be saved.
    output = Path(args.output)

    # Open the output CSV file.
    #
    # The file will contain one row per simulation round.
    with output.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        # Use dictionary keys as column names.
        writer = csv.DictWriter(
            f,
            fieldnames=list(history[0])
        )

        # Write column headers.
        writer.writeheader()

        # Write the results of every round.
        writer.writerows(history)

    # Calculate overall experiment statistics.
    result = summarize(history)

    # Print the results to the terminal.
    print("Attacker vs Defender: Multiagent Security Game")

    print(
        f"Rounds: {result['rounds']} "
        f"| noise: {args.noise} "
        f"| defender budget: {args.budget}"
    )

    print(
        f"Attack success rate: "
        f"{result['attack_success_rate']:.1%}"
    )

    print(
        f"Average attacker payoff: "
        f"{result['average_attacker_payoff']:.2f}"
    )

    print(f"Results saved to: {output.resolve()}")

    # Show a preview of the first five rounds.
    #
    # This makes it easier to see how agents interact.
    print("First 5 rounds:")

    for row in history[:5]:
        print(
            f"  Round {row['round']}: "
            f"defender protects [{row['defended_targets']}], "
            f"attacker chooses {row['attacked_target']}, "
            f"{'BLOCKED' if row['blocked'] else 'SUCCESS'}, "
            f"payoff {row['attacker_payoff']}"
        )


# ============================================================
# 7. START THE PROGRAM
# ============================================================

# This condition ensures main() executes only when this
# Python file is run directly.
#
# Example:
# python security_agents.py
#
# It will not automatically execute if another Python
# file imports this module.

if __name__ == "__main__":
    main()
