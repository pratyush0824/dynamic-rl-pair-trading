# Reinforcement Learning Pair Trading for Cryptocurrency

This project implements a Reinforcement Learning based pair trading system inspired by the paper:

**Yang, Hongshen & Malik, Avinash (2024)**  
*Reinforcement Learning Pair Trading: A Dynamic Scaling Approach*  
Journal of Risk and Financial Management.

The system builds a trading environment where an RL agent learns when and how much to trade between statistically related crypto assets.

The implementation focuses on:

• spread-based statistical arbitrage  
• reinforcement learning trading agents  
• dynamic position sizing  
• portfolio performance evaluation

---

# Project Structure

The repository contains the following Python modules:
```
agent_rl_advanced.py
data_loading.py
env_trading_advanced.py
pair_spread_utils.py
portfolio_manager.py
results_analysis.py
```

---

# File Descriptions

## data_loading.py
Handles loading and preprocessing of cryptocurrency market data.

Responsibilities:
- Load price data for multiple assets
- Merge datasets into a single dataframe
- Format columns for the trading environment

Expected column format:

```
close_BTC
close_ETH
close_LTC
...
```

---

## pair_spread_utils.py
Utility functions used for pair trading calculations.

Includes:

**compute_regression_spread()**

Computes the spread between two assets using regression residuals:

```
p_i = β0 + β1 * p_j + s
```

where `s` is the spread used for trading signals.

**compute_zone()**

Classifies spread into trading zones:

| Zone | Meaning |
|----|----|
Short Zone | Spread > open threshold |
Neutral Short | Close threshold < Spread < Open threshold |
Close Zone | Spread near mean |
Neutral Long | Negative spread deviation |
Long Zone | Spread < -open threshold |

---

## env_trading_advanced.py
Defines the **custom reinforcement learning trading environment**.

Built using **OpenAI Gym API**.

Key features:

- regression-based spread
- rolling z-score normalization
- zone-based observations
- continuous action space
- dynamic position sizing
- reward shaping

### Observation Space

```
[z-score, zone, position, portfolio_value]
```

for each trading pair.

### Action Space

```
A ∈ [-1, 1]
```

Where:

| Action | Meaning |
|----|----|
-1 | full short-leg |
0 | no position |
1 | full long-leg |

Values between represent partial portfolio allocation.

---

## agent_rl_advanced.py
Handles RL agent training and evaluation.

Supported algorithms:

- PPO
- A2C
- DQN

Uses **Stable-Baselines3**.

Responsibilities:

- environment wrapping
- agent training
- policy evaluation
- backtesting loop

---

## portfolio_manager.py
Coordinates the trading workflow.

Responsibilities:

- environment initialization
- RL agent training
- evaluation
- portfolio tracking

Outputs:

- equity curve
- trading history
- portfolio statistics

---

## results_analysis.py
Computes trading performance metrics.

Metrics include:

- cumulative return
- Sharpe ratio
- volatility
- drawdown
- win/loss ratio
- trade statistics

These metrics are used to evaluate the effectiveness of the trading strategy.

---

# Methodology

The system follows a statistical arbitrage workflow:

```
Market Data
    ↓
Spread Calculation
    ↓
Rolling Z-Score
    ↓
Trading Zones
    ↓
Reinforcement Learning Agent
    ↓
Dynamic Position Allocation
    ↓
Portfolio Evaluation
```

Spread is defined as the regression residual between two assets.

```
spread = price_i - (β0 + β1 * price_j)
```

The spread is normalized using rolling z-score:

```
z = (spread - mean) / std
```

The RL agent observes spread signals and learns an optimal trading policy.

---

# Reinforcement Learning Setup

Environment type:

```
Custom Gym Environment
```

Reward function components:

1. Portfolio profit/loss
2. Action reward for correct zone behavior
3. Transaction penalty to discourage excessive trading

This encourages the agent to:

- open positions when spreads deviate
- close positions when spreads revert
- trade efficiently with minimal cost

---

# Installation

Install dependencies:

```
pip install numpy pandas gym stable-baselines3 statsmodels
```

Optional:

```
pip install matplotlib
```

---
