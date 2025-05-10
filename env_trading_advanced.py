import gym
import numpy as np
import pandas as pd
from gym import spaces

class AdvancedPairTradingEnv(gym.Env):
    """
    Example environment with:
      - trade_history stored in self.trade_history
      - each step, we append the old_pos, new_pos, step_pnl, portfolio_value, etc.
    """

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
        risk_stop: float = 0.3
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

        # 1) Build 'spread' columns => log(p_base) - log(p_quote)
        self.spread_cols = []
        for pair in self.pair_list:
            base, quote = pair.split('-')
            col_base = f"close_{base}"
            col_quote = f"close_{quote}"
            spread_col = f"spread_{base}_{quote}"
            self.df[spread_col] = np.log(self.df[col_base]) - np.log(self.df[col_quote])
            self.spread_cols.append(spread_col)

        # 2) rolling mean & std => zscore
        for spread_col in self.spread_cols:
            roll_mean = self.df[spread_col].rolling(self.window_size).mean()
            roll_std = self.df[spread_col].rolling(self.window_size).std()
            z_col = spread_col.replace('spread', 'zscore')
            self.df[z_col] = (self.df[spread_col] - roll_mean) / (roll_std + 1e-8)

        # 3) Drop NaN
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        if len(self.df) < self.window_size + 1:
            raise ValueError(
                f"Not enough data after rolling. Need > {self.window_size}, got {len(self.df)}."
            )

        # 4) Adjust max_episode_steps if needed
        max_possible = (len(self.df) - self.window_size - 1) // self.step_size
        if self.max_episode_steps is not None:
            self.max_episode_steps = min(self.max_episode_steps, max_possible)

        # 5) Observations: zscore_i + position_i + portfolio_value_ratio
        self.num_pairs = len(self.pair_list)
        obs_dim = self.num_pairs * 2 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 6) Action space: continuous in [-0.5, 0.5]
        self.action_space = spaces.Box(
            low=-0.5, high=0.5, shape=(self.num_pairs,), dtype=np.float32
        )

        # Internals
        self.current_idx = 0
        self.step_counter = 0
        self.done_flag = False

        self.positions = np.zeros(self.num_pairs, dtype=np.float32)
        self.last_positions = np.zeros_like(self.positions)
        self.portfolio_value = self.initial_capital
        self.trades_count = 0
        self.time_in_market_steps = 0

        # Logs
        self.equity_curve = []
        self.dates = []

        # Pair-level equity tracking (optional)
        self.equity_curve_per_pair = { pair: [] for pair in self.pair_list }
        self.dates_per_pair = { pair: [] for pair in self.pair_list }

        # **IMPORTANT**: We'll store trade records here
        self.trade_history = []

    def _get_current_row(self):
        if self.current_idx >= len(self.df):
            return self.df.iloc[-1]
        return self.df.iloc[self.current_idx]

    def _get_obs(self):
        row = self._get_current_row()
        zscores = []
        for pair in self.pair_list:
            base, quote = pair.split('-')
            z_col = f"zscore_{base}_{quote}"
            zscores.append(row[z_col])

        obs = []
        obs.extend(zscores)
        obs.extend(self.positions)
        obs.append(self.portfolio_value / self.initial_capital)
        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.done_flag = False
        self.current_idx = self.window_size
        if self.current_idx >= len(self.df):
            self.current_idx = len(self.df) - 1

        self.step_counter = 0
        self.positions = np.zeros(self.num_pairs, dtype=np.float32)
        self.last_positions = np.zeros_like(self.positions)
        self.portfolio_value = self.initial_capital
        self.trades_count = 0
        self.time_in_market_steps = 0

        self.equity_curve = []
        self.dates = []
        self.equity_curve_per_pair = { pair: [] for pair in self.pair_list }
        self.dates_per_pair = { pair: [] for pair in self.pair_list }

        # Re-initialize the trade_history on each reset
        self.trade_history = []

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 1) PnL from old positions
        step_pnl, pairwise_pnls = self._compute_pnl()

        # 2) Transaction cost
        pos_change = np.abs(action - self.positions).sum()
        cost = pos_change * self.initial_capital * self.transaction_cost
        step_pnl -= cost

        # 3) Funding cost
        funding_cost = np.sum(np.abs(self.positions)) * self.initial_capital * self.funding_spread
        step_pnl -= funding_cost

        # 4) Update portfolio
        self.portfolio_value += step_pnl

        done = False
        truncated = False
        if self.portfolio_value <= (self.risk_stop * self.initial_capital):
            # risk stop
            step_pnl = self.risk_stop * self.initial_capital - self.portfolio_value
            self.portfolio_value = self.risk_stop * self.initial_capital
            done = True
            truncated = True

        # 5) Count a trade if sum of position changes > 0.01
        position_diff = np.sum(np.abs(action - self.positions))
        if position_diff > 0.01:
            self.trades_count += 1

        row_now = self._get_current_row()
        self.equity_curve.append(self.portfolio_value)
        self.dates.append(row_now["time"])

        # pair-level partial PnL
        if not hasattr(self, 'pair_values'):
            self.pair_values = {p: self.initial_capital / len(self.pair_list) for p in self.pair_list}

        for i, pair in enumerate(self.pair_list):
            self.pair_values[pair] += pairwise_pnls[i]
            if self.pair_values[pair] < 0:
                self.pair_values[pair] = 0
            self.equity_curve_per_pair[pair].append(self.pair_values[pair])
            self.dates_per_pair[pair].append(row_now["time"])

        # **HERE** we append the trade info
        self.trade_history.append({
            "step": self.step_counter,
            "time": row_now["time"],
            "old_pos": self.positions.copy(),
            "new_pos": action.copy(),
            "step_pnl": step_pnl,
            "portfolio_value": self.portfolio_value
        })

        self.last_positions = self.positions.copy()
        self.positions = action

        scaled_reward = step_pnl * self.reward_scaling

        self.current_idx += self.step_size
        self.step_counter += 1

        if not done:
            if self.current_idx >= len(self.df) - 1:
                done = True
            elif self.step_counter >= self.max_episode_steps:
                done = True
                truncated = True

        obs = self._get_obs()
        info = {
            "portfolio_value": self.portfolio_value,
            "trades_count": self.trades_count
        }
        return obs, scaled_reward, done, truncated, info

    def _compute_pnl(self):
        if self.step_counter == 0:
            return 0.0, [0.0]*self.num_pairs

        row_now = self._get_current_row()
        row_prev_idx = self.current_idx - self.step_size
        if row_prev_idx < 0:
            row_prev_idx = 0
        row_prev = self.df.iloc[row_prev_idx]

        total_pnl = 0.0
        pairwise_pnls = []
        for i, pair in enumerate(self.pair_list):
            base, quote = pair.split('-')
            spread_col = f"spread_{base}_{quote}"
            spread_now = row_now[spread_col]
            spread_prev = row_prev[spread_col]
            spread_diff = spread_now - spread_prev

            pos_frac = self.positions[i]
            notional = self.initial_capital * abs(pos_frac)
            direction = np.sign(pos_frac)
            pair_pnl = notional * direction * spread_diff
            total_pnl += pair_pnl
            pairwise_pnls.append(pair_pnl)

        return total_pnl, pairwise_pnls

    def render(self, mode='human'):
        print(f"Step: {self.step_counter}, "
              f"Index: {self.current_idx}, "
              f"Value: {self.portfolio_value:.2f}, "
              f"Trades: {self.trades_count}")
