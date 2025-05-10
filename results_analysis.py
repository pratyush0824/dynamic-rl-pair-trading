import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import skew, kurtosis

def save_trade_history(trade_history, run_label, output_dir="./results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, f"trade_history_{run_label}.txt")
    with open(filepath, "w") as f:
        f.write("step,time,old_pos,new_pos,step_pnl,portfolio_value\n")
        for t in trade_history:
            f.write(f"{t['step']},{t['time']},{t['old_pos']},{t['new_pos']},{t['step_pnl']},{t['portfolio_value']}\n")

def daily_analysis(equity_times, equity_values):
    """
    Build a daily-resampled DataFrame of equity and compute daily returns.
    Returns a DataFrame with columns: equity, norm_equity, daily_ret
    """
    df_curve = pd.DataFrame({'time': equity_times, 'equity': equity_values})
    df_curve.set_index('time', inplace=True)

    if isinstance(df_curve.index, pd.DatetimeIndex):
        # forward fill to each minute
        df_curve = df_curve.asfreq('T', method='ffill')
        # now resample daily
        df_daily = df_curve.resample('1D').last().dropna()
    else:
        # fallback if index is not datetime
        df_curve['day_idx'] = np.arange(len(df_curve))
        df_daily = df_curve.copy()
        df_daily.set_index('day_idx', inplace=True)

    if len(df_daily) == 0:
        return df_daily  # empty

    # build norm_equity, daily_ret
    start_val = df_daily['equity'].iloc[0]
    df_daily['norm_equity'] = df_daily['equity'] / start_val
    df_daily['daily_ret'] = df_daily['norm_equity'].pct_change().fillna(0)
    return df_daily

def compute_metrics(df_daily):
    """
    Given daily data with 'equity', 'norm_equity', 'daily_ret',
    compute final metrics: final_cumret, cagr, ann_vol, sharpe, skew, kurt
    """
    if len(df_daily) < 2:
        return {}

    final_cumret = df_daily['norm_equity'].iloc[-1] - 1.0
    num_days = len(df_daily)
    cagr = 0.0
    if num_days > 1:
        final_equity = df_daily['norm_equity'].iloc[-1]
        cagr = (final_equity ** (365.0 / num_days)) - 1.0

    daily_std = df_daily['daily_ret'].std()
    ann_vol = daily_std * np.sqrt(365) if daily_std > 0 else 0.0
    sharpe = 0.0
    if daily_std and daily_std > 1e-12:
        sharpe = (df_daily['daily_ret'].mean() / daily_std) * np.sqrt(365)

    daily_skew = skew(df_daily['daily_ret'])
    daily_kurt = kurtosis(df_daily['daily_ret'])

    return {
        "Cumulative Return": final_cumret,
        "CAGR": cagr,
        "Annual Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Skewness": daily_skew,
        "Kurtosis": daily_kurt
    }

def analyze_run(run_label, backtest_res, output_dir="./results"):
    """
    Analyze a single run: entire portfolio plus pair-level lines if multiple pairs.
    Also produce a time-series Sharpe ratio plot.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    trade_history = backtest_res['trade_history']
    equity_curve = backtest_res['equity_curve']
    dates = backtest_res['dates']
    eq_per_pair = backtest_res['equity_curve_per_pair']
    dates_per_pair = backtest_res['dates_per_pair']

    # Save trade history
    save_trade_history(trade_history, run_label, output_dir=output_dir)

    # 1) Entire portfolio daily analysis
    df_daily_port = daily_analysis(dates, equity_curve)
    metrics_port = {}
    if len(df_daily_port) > 0:
        metrics_port = compute_metrics(df_daily_port)

    # 2) Plot entire portfolio daily returns & cumulative
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    if len(df_daily_port) > 0:
        axes[0].bar(df_daily_port.index, df_daily_port['daily_ret'], color='steelblue')
        axes[0].set_title(f"{run_label} - Daily Returns (Portfolio)")
        axes[0].set_ylabel("Daily Return")

        axes[1].plot(df_daily_port.index, df_daily_port['norm_equity'], label='Portfolio')
        axes[1].set_title(f"{run_label} - Cumulative (start=1.0) (Portfolio)")
        axes[1].legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{run_label}_portfolio_returns.png")
    plt.savefig(plot_path)
    plt.close()

    # 3) If multiple pairs, produce additional lines for each pair
    #    We'll create a single figure with them overlaid
    if len(eq_per_pair) > 1:
        plt.figure(figsize=(10, 6))
        # first do the portfolio line
        if len(df_daily_port) > 0:
            plt.plot(df_daily_port.index, df_daily_port['norm_equity'], label='Portfolio', lw=2)

        # now each pair
        for pair, eq_list in eq_per_pair.items():
            pair_dates = dates_per_pair[pair]
            df_daily_pair = daily_analysis(pair_dates, eq_list)
            if len(df_daily_pair) > 0:
                plt.plot(df_daily_pair.index, df_daily_pair['norm_equity'], label=pair)

        plt.title(f"{run_label} - Cumulative (Portfolio vs. Each Pair)")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return (start=1.0)")
        plt.legend()
        plot_path2 = os.path.join(output_dir, f"{run_label}_pairs_vs_portfolio.png")
        plt.savefig(plot_path2)
        plt.close()

    # 4) Time-series Sharpe ratio: we can compute rolling Sharpe over a window of daily returns
    #    We'll do a 20-day rolling Sharpe for the portfolio
    if len(df_daily_port) > 20:
        window = 20
        rolling_sharpe = []
        idx_list = df_daily_port.index[window:]
        for i in range(window, len(df_daily_port)):
            window_rets = df_daily_port['daily_ret'].iloc[i-window:i]
            wstd = window_rets.std()
            wmean = window_rets.mean()
            if wstd > 1e-12:
                sh = (wmean / wstd) * np.sqrt(365)
            else:
                sh = 0.0
            rolling_sharpe.append(sh)

        plt.figure(figsize=(10, 4))
        plt.plot(idx_list, rolling_sharpe, label='Rolling Sharpe (20-day window)')
        plt.title(f"{run_label} - Rolling Sharpe Ratio")
        plt.legend()
        plt.tight_layout()
        sh_path = os.path.join(output_dir, f"{run_label}_rolling_sharpe.png")
        plt.savefig(sh_path)
        plt.close()

    # Return final metrics
    final_dict = {
        'run_label': run_label,
        'portfolio_metrics': metrics_port,
        'trades_count': backtest_res['trades_count'],
        'final_value': backtest_res['final_value'],
        'time_in_market_steps': backtest_res['time_in_market_steps'],
        'total_steps': backtest_res['total_steps']
    }
    return final_dict

def plot_all_cumulative(all_results, output_path="./results/all_cumulative.png"):
    """
    Overlays each run's backtest cumulative returns in one figure (only portfolio line).
    """
    plt.figure(figsize=(10, 6))
    for run_label, res_data in all_results.items():
        eq_curve = res_data['backtest']['equity_curve']
        dates = res_data['backtest']['dates']
        df_daily = daily_analysis(dates, eq_curve)
        if len(df_daily) < 2:
            continue
        plt.plot(df_daily.index, df_daily['norm_equity'], label=run_label)

    plt.title("All Runs - Backtest Cumulative (start=1.0)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Function to convert analysis_dict to DataFrame
def convert_analysis_dict_to_df(analysis_dict):
    data_list = []
    for portfolio, details in analysis_dict.items():
        row = {
            "Portfolio": portfolio,
            "Scenario": details["scenario"],
            "Pairs": ", ".join(details["portfolio"]),
            **details["metrics"]["portfolio_metrics"],  # Unpack portfolio metrics
            "Trades Count": details["metrics"]["trades_count"],
            "Final Value": details["metrics"]["final_value"],
            "Total Steps": details["metrics"]["total_steps"],
        }
        data_list.append(row)

    df_analysis = pd.DataFrame(data_list)
    return df_analysis