import os
import pandas as pd
import numpy as np
from datetime import timedelta

from cmds.env_trading_advanced import AdvancedPairTradingEnv
from cmds.agent_rl_advanced import train_agent, evaluate_agent

def run_portfolio_scenarios(
    df_12_months,
    results_dir="./results",
    timesteps=20_000,
    algo='PPO',
    weekly_retrain=True
):
    """
    1) We have 12 months data in df_12_months (already preprocessed).
    2) We'll split the first ~10 months for training, the last ~2 months for testing.
    3) If weekly_retrain=True, we do a walk-forward approach:
       - Start from train_start to train_end for initial training
       - Then each week in the test window, retrain model with the data up to that point
       - Evaluate for next week, etc.
    4) We'll do 3 portfolios, 2 cost scenarios.
    """

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Identify time range
    df_12_months = df_12_months.sort_values(by="time").reset_index(drop=True)
    start_time = df_12_months["time"].iloc[0]
    end_time = df_12_months["time"].iloc[-1]

    # We'll define train_end as start_time + ~10 months
    # For simplicity, say 30 days * 10 = 300 days
    # or you can do actual date offsets
    train_days = 300
    train_end_time = start_time + pd.Timedelta(days=train_days)

    # We'll define the test_end_time as the last row
    # so test range is [train_end_time, end_time]
    df_train_all = df_12_months[df_12_months["time"] < train_end_time].copy()
    df_test_all = df_12_months[df_12_months["time"] >= train_end_time].copy()

    # Define portfolios
    portfolio_1 = ["btc-eth","btc-ltc","btc-xrp","eth-ltc","eth-ftm"]
    portfolio_2 = ["btc-eth"]
    portfolio_3 = ["btc-ltc","eth-ftm"]

    cost_scenarios = [
        {
            'name': 'ZeroCost',
            'transaction_cost': 0.0,
            'funding_spread': 0.0
        },
        {
            'name': 'NonTrivialCosts',
            'transaction_cost': 0.001,   # 0.1% cost
            'funding_spread': 0.0002    # 0.02% funding
        }
    ]

    portfolios = [
        ("portfolio_1", portfolio_1),
        ("portfolio_2", portfolio_2),
        ("portfolio_3", portfolio_3),
    ]

    results = {}

    # We'll define a function to get sub-dataframe for a time range
    def filter_df_by_time(df, start_t, end_t):
        return df[(df["time"] >= start_t) & (df["time"] < end_t)].copy()

    # We'll define how we do weekly segments
    # start_of_test = train_end_time
    # end_of_test = end_time
    # we will do: [start_of_test, start_of_test+7days), next 7 days, etc.

    test_results = {}
    # We only do walk-forward if weekly_retrain is True
    # Otherwise, we just train once on df_train_all, test once on df_test_all

    if not weekly_retrain:
        # Single training, single test
        for pf_name, pf_pairs in portfolios:
            for scenario in cost_scenarios:
                sc_name = scenario['name']
                run_label = f"{pf_name}_{sc_name}"

                # Train once on df_train_all
                def make_train_env():
                    return AdvancedPairTradingEnv(
                        df_merged=df_train_all,
                        pair_list=pf_pairs,
                        window_size=60,
                        step_size=60,
                        initial_capital=100_000,
                        max_leverage=1.0,
                        transaction_cost=scenario['transaction_cost'],
                        funding_spread=scenario['funding_spread'],
                        reward_scaling=1e-4,
                        max_episode_steps=999999,
                        risk_stop=0.3
                    )

                model_save_path = f"{results_dir}/{run_label}_{algo}.zip"
                model = train_agent(
                    make_env_fn=make_train_env,
                    algo=algo,
                    total_timesteps=timesteps,
                    model_save_path=model_save_path
                )

                # Evaluate on df_test_all
                def make_test_env():
                    return AdvancedPairTradingEnv(
                        df_merged=df_test_all,
                        pair_list=pf_pairs,
                        window_size=60,
                        step_size=60,
                        initial_capital=100_000,
                        max_leverage=1.0,
                        transaction_cost=scenario['transaction_cost'],
                        funding_spread=scenario['funding_spread'],
                        reward_scaling=1e-4,
                        max_episode_steps=999999,
                        risk_stop=0.3
                    )

                eval_res = evaluate_agent(make_test_env, model, max_steps=999999)
                test_results[run_label] = {
                    'portfolio': pf_pairs,
                    'scenario': sc_name,
                    'backtest': eval_res
                }

    else:
        # Weekly retraining approach
        # We'll define each 7-day segment in test range
        current_start = train_end_time
        # We'll store partial results in test_results
        for pf_name, pf_pairs in portfolios:
            for scenario in cost_scenarios:
                sc_name = scenario['name']
                run_label_full = f"{pf_name}_{sc_name}"

                # We'll accumulate the final backtest in a combined equity/time
                combined_equity = []
                combined_dates = []
                combined_trade_history = []
                final_value = 100000
                trades_count = 0
                time_in_market_steps = 0
                total_steps = 0

                # We'll define an iterative approach
                tmp_start = current_start
                # We'll define model = None initially
                model = None
                last_portfolio_val = 100000

                while tmp_start < end_time:
                    tmp_end = tmp_start + pd.Timedelta(days=7)
                    if tmp_end > end_time:
                        tmp_end = end_time

                    # 1) Expand training data up to tmp_start
                    df_train_slice = df_12_months[df_12_months["time"] < tmp_start].copy()
                    if len(df_train_slice) < 100:
                        # not enough data
                        break

                    # Train model on that slice
                    def make_train_env():
                        return AdvancedPairTradingEnv(
                            df_merged=df_train_slice,
                            pair_list=pf_pairs,
                            window_size=60,
                            step_size=60,
                            initial_capital=100_000,
                            max_leverage=1.0,
                            transaction_cost=scenario['transaction_cost'],
                            funding_spread=scenario['funding_spread'],
                            reward_scaling=1e-4,
                            max_episode_steps=999999,
                            risk_stop=0.3
                        )
                    model_save_path = f"{results_dir}/{run_label_full}_{tmp_start.date()}.zip"
                    model = train_agent(
                        make_env_fn=make_train_env,
                        algo=algo,
                        total_timesteps=timesteps,
                        model_save_path=model_save_path
                    )

                    # 2) Evaluate on [tmp_start, tmp_end)
                    df_test_slice = filter_df_by_time(df_12_months, tmp_start, tmp_end)
                    if len(df_test_slice) < 2:
                        # no real test
                        tmp_start = tmp_end
                        continue

                    def make_test_env():
                        return AdvancedPairTradingEnv(
                            df_merged=df_test_slice,
                            pair_list=pf_pairs,
                            window_size=60,
                            step_size=60,
                            initial_capital=last_portfolio_val,
                            max_leverage=1.0,
                            transaction_cost=scenario['transaction_cost'],
                            funding_spread=scenario['funding_spread'],
                            reward_scaling=1e-4,
                            max_episode_steps=999999,
                            risk_stop=0.3
                        )
                    eval_res = evaluate_agent(make_test_env, model, max_steps=999999)

                    # We combine the equity curves
                    combined_equity.extend(eval_res['equity_curve'])
                    combined_dates.extend(eval_res['dates'])
                    combined_trade_history.extend(eval_res['trade_history'])
                    final_value = eval_res['final_value']
                    trades_count += eval_res['trades_count']
                    time_in_market_steps += eval_res['time_in_market_steps']
                    total_steps += eval_res['total_steps']
                    last_portfolio_val = final_value  # start next step with final

                    tmp_start = tmp_end
                    if tmp_start >= end_time:
                        break

                # End while
                # Build a pseudo-backtest result
                final_backtest = {
                    'final_value': final_value,
                    'trades_count': trades_count,
                    'equity_curve': combined_equity,
                    'dates': combined_dates,
                    'trade_history': combined_trade_history,
                    'time_in_market_steps': time_in_market_steps,
                    'total_steps': total_steps,
                    'equity_curve_per_pair': {},  # omitted for brevity in walk-forward
                    'dates_per_pair': {}
                }

                test_results[run_label_full] = {
                    'portfolio': pf_pairs,
                    'scenario': sc_name,
                    'backtest': final_backtest
                }

    return test_results
