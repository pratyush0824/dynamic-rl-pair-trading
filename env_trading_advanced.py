import gym
import numpy as np
import pandas as pd
from gym import spaces

from pair_spread_utils import compute_regression_spread, compute_zone


class AdvancedPairTradingEnv(gym.Env):

    def __init__(
        self,
        df_merged: pd.DataFrame,
        pair_list: list,
        window_size: int = 60,
        step_size: int = 60,
        initial_capital: float = 1e5,
        max_leverage: float = 1.0,
        transaction_cost: float = 0.0,
        funding_spread: float = 0.0,
        reward_scaling: float = 1e-4,
        max_episode_steps: int = 5000,
        risk_stop: float = 0.3,
        open_threshold: float = 1.8,
        close_threshold: float = 0.4
    ):

        super().__init__()

        self.df = df_merged.copy()
        self.pair_list = pair_list
        self.window_size = window_size
        self.step_size = step_size
        self.initial_capital = initial_capital
        self.max_leverage = max_leverage
        self.transaction_cost = transaction_cost
        self.funding_spread = funding_spread
        self.reward_scaling = reward_scaling
        self.max_episode_steps = max_episode_steps
        self.risk_stop = risk_stop

        self.open_threshold = open_threshold
        self.close_threshold = close_threshold

        self.spread_cols = []
        self.zscore_cols = []

        # --- compute spreads using regression (paper definition)

        for pair in self.pair_list:

            base, quote = pair.split('-')

            col_base = f"close_{base}"
            col_quote = f"close_{quote}"

            spread_col = f"spread_{base}_{quote}"

            window_base = self.df[col_base].iloc[:self.window_size]
            window_quote = self.df[col_quote].iloc[:self.window_size]

            _, beta0, beta1 = compute_regression_spread(
                window_base,
                window_quote
            )

            spread = self.df[col_base] - (beta0 + beta1 * self.df[col_quote])

            self.df[spread_col] = spread
            self.spread_cols.append(spread_col)

        # --- rolling zscore

        for spread_col in self.spread_cols:

            roll_mean = self.df[spread_col].rolling(self.window_size).mean()
            roll_std = self.df[spread_col].rolling(self.window_size).std()

            z_col = spread_col.replace("spread", "zscore")

            self.df[z_col] = (self.df[spread_col] - roll_mean) / (roll_std + 1e-8)

            self.zscore_cols.append(z_col)

        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.num_pairs = len(self.pair_list)

        obs_dim = self.num_pairs * 3 + 1

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Action space must be [-1,1] (paper)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_pairs,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):

        self.current_idx = self.window_size
        self.step_counter = 0

        self.positions = np.zeros(self.num_pairs)
        self.portfolio_value = self.initial_capital

        self.trades_count = 0
        self.time_in_market_steps = 0

        self.equity_curve = []
        self.dates = []

        self.trade_history = []

        return self._get_obs(), {}

    def _get_current_row(self):

        return self.df.iloc[self.current_idx]

    def _get_obs(self):

        row = self._get_current_row()

        zscores = np.array([row[c] for c in self.zscore_cols])

        zones = np.array([
            compute_zone(z, self.open_threshold, self.close_threshold)
            for z in zscores
        ])

        obs = []

        obs.extend(zscores)
        obs.extend(zones)
        obs.extend(self.positions)

        obs.append(self.portfolio_value / self.initial_capital)

        return np.array(obs, dtype=np.float32)

    def step(self, action):

        action = np.clip(action, -1, 1)

        row = self._get_current_row()

        zscores = np.array([row[c] for c in self.zscore_cols])

        zones = np.array([
            compute_zone(z, self.open_threshold, self.close_threshold)
            for z in zscores
        ])

        prev_positions = self.positions.copy()

        # --- PnL from previous positions

        if self.current_idx + 1 < len(self.df):

            next_row = self.df.iloc[self.current_idx + 1]

            pnl = 0

            for i, pair in enumerate(self.pair_list):

                spread_col = self.spread_cols[i]

                spread_now = row[spread_col]
                spread_next = next_row[spread_col]

                spread_return = spread_next - spread_now

                if np.isfinite(spread_return):
                    pnl += prev_positions[i] * spread_return

        else:

            pnl = 0

        # --- adjust positions (paper: adjust not replace)

        position_change = action - prev_positions

        self.positions += position_change

        self.positions = np.clip(self.positions, -1.0, 1.0)

        transaction_cost = np.sum(np.abs(position_change)) * self.transaction_cost

        pnl -= transaction_cost

        self.portfolio_value += pnl

        # --- reward components

        portfolio_reward = pnl

        transaction_penalty = np.sum(np.abs(position_change))

        action_reward = 0

        for z, a in zip(zones, action):

            if z == 2 and a < 0:
                action_reward += 1

            if z == -2 and a > 0:
                action_reward += 1

            if z == 0 and abs(a) < 0.05:
                action_reward += 1

        reward = (
                portfolio_reward
                + 0.1 * action_reward
                - 0.01 * transaction_penalty
        )

        reward *= self.reward_scaling

        # --- bookkeeping

        self.equity_curve.append(self.portfolio_value)

        if "time" in row:
            self.dates.append(row["time"])
        else:
            self.dates.append(self.current_idx)

        self.trade_history.append(
            {
                "step": self.step_counter,
                "time": self.dates[-1],
                "old_pos": prev_positions.tolist(),
                "new_pos": self.positions.tolist(),
                "step_pnl": pnl,
                "portfolio_value": self.portfolio_value
            }
        )

        if np.any(self.positions != 0):
            self.time_in_market_steps += 1

        self.current_idx += self.step_size
        self.step_counter += 1

        done = False

        if self.current_idx >= len(self.df) - 1:
            done = True

        if self.portfolio_value < self.initial_capital * (1 - self.risk_stop):
            done = True

        if self.step_counter >= self.max_episode_steps:
            done = True

        obs = self._get_obs()

        return obs, reward, done, False, {}
