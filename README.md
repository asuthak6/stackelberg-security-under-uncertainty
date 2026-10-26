# SE 4457: Attacker vs. Defender Multiagent Security Game

## Project Overview

This project is a Python based multiagent cybersecurity simulation developed for SE 4457: Foundations of Multiagent Systems at Western University.

The simulation involves two autonomous agents, an attacker and a defender, competing in a network environment containing four servers.

The attacker attempts to disrupt server availability through simulated Distributed Denial of Service (DDoS) attacks, while the defender allocates limited mitigation resources to prevent outages.

The primary objective is to investigate how imperfect perception, autonomous decision making, and adaptive behaviour influence the performance of both agents.

## Current Implementation

### 1. Attacker Agent

The attacker is responsible for selecting a target server and determining the intensity of a simulated DDoS attack.

Its decision making incorporates:

* Noisy observations of the defender's security configuration
* Estimated protection levels based on previous interactions
* Expected payoff calculations
* Exploration of alternative actions
* Belief updates following attack outcomes

The attacker attempts to maximize its rewards by identifying valuable, potentially vulnerable servers.

### 2. Defender Agent

The defender is responsible for distributing limited mitigation resources across the network.

Its decision making incorporates:

* Server importance and protection priorities
* Previous attack patterns
* Available security resources
* Adaptive resource allocation

The defender attempts to minimize successful attacks and maintain server availability.

### 3. Simulated DDoS Attack

The current implementation models a simplified volumetric traffic flooding attack.

Each server has a base traffic capacity of 5 units.

The attacker selects an intensity of 2, 4, or 6 simulated traffic units, while the defender allocates resources to increase server capacity.

An outage occurs when the attack intensity exceeds the server's effective capacity.

**Note:** This project does not generate actual network traffic or perform real cyberattacks. All attacks are represented through mathematical simulations.

### 4. Multiagent Interaction

Each simulation round follows this sequence:

1. The defender allocates mitigation resources across the servers.
2. The attacker receives potentially inaccurate observations.
3. The attacker evaluates possible targets and attack intensities.
4. The attacker selects an action.
5. The environment determines whether the attack causes an outage.
6. Both agents receive feedback and update their internal information.
7. The process repeats for the specified number of rounds.

This repeated interaction allows the agents to adapt their decisions based on previous outcomes.

## Multiagent Concepts

The project currently demonstrates several concepts from multiagent systems:

| Concept | Implementation |
|---|---|
| Autonomy | Both agents independently select actions |
| Perception | The attacker receives noisy information about defenses |
| Reasoning | Agents evaluate potential actions using heuristic strategies |
| Adaptation | Agents update internal information following interactions |
| Competition | The agents pursue conflicting objectives |
| Environment | Four simulated servers with resource constraints |

The current agents use adaptive heuristics rather than formal reinforcement learning.

## Technologies

* Python 3.10+
* Python standard library
* Object oriented programming
* CSV for experiment results
* Git and GitHub for version control

No external Python packages are required.

## Running the Project

### Requirements

Install Python 3.10 or newer.

### Run the default simulation

```bash
python security_agents.py
```

### Run 1,000 rounds

```bash
python security_agents.py --rounds 1000
```

### Change observation noise

```bash
python security_agents.py --rounds 1000 --noise 0.5
```

### Change defender budget

```bash
python security_agents.py --rounds 1000 --budget 2
```

### Disable learning

```bash
python security_agents.py --no-learning
```

### Save results to a custom file

```bash
python security_agents.py --output experiment.csv
```

## Initial Experimental Results

An initial simulation was executed with the following configuration:

| Parameter | Value |
|---|---|
| Simulation rounds | 100 |
| Observation noise | 0.20 |
| Defender budget | 4 |
| Server count | 4 |
| Attack intensities | 2, 4, 6 |

### Results

| Metric | Result |
|---|---|
| Service outage rate | 55% |
| Average attacker payoff | 0.51 |

These results are preliminary and represent one simulation configuration. Additional controlled experiments are required to evaluate the agents' effectiveness.

The simulation saves detailed results to `results.csv` for further analysis.

## Current Limitations

The current prototype has several limitations:

* The network environment is simplified and does not simulate actual network protocols.
* Only generic volumetric flooding is currently represented.
* Agent decisions use heuristic strategies rather than advanced reinforcement learning.
* Additional experiments are needed to assess performance across different configurations.
* The system does not currently compute a Stackelberg equilibrium.

## Future Development

The next planned improvements include:

1. **Multiple attack types:** Introduce abstract UDP flooding, SYN flooding, and HTTP flooding models.
2. **Improved attacker reasoning:** Explore Bayesian belief updating to better handle uncertain observations.
3. **Reinforcement learning:** Investigate Q learning for adaptive attack selection.
4. **Baseline comparisons:** Compare random, heuristic, and learning based strategies.
5. **Experimental evaluation:** Test different noise levels, defender budgets, and simulation lengths.
6. **Visualization:** Generate graphs showing attack success, defender effectiveness, and agent adaptation over time.

## Research Objective

The central research question is:

**How does imperfect perception influence attacker decision making and adaptive defense strategies in a multiagent cybersecurity environment?**

By comparing different reasoning methods, attack strategies, and observation noise levels, the project aims to evaluate how autonomous agents perform when operating under uncertainty.

## Project Status

**Current Stage:** Functional prototype with simulated DDoS attacks and adaptive attacker and defender agents.

**Next Milestone:** Implement additional simulated attack types and conduct controlled comparisons of agent strategies.