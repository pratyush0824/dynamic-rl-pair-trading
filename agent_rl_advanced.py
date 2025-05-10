import gym
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv

def train_agent(
    make_env_fn,
    algo='PPO',
    total_timesteps=100_000,
    model_save_path='rl_model.zip',
    policy_kwargs=None,
    sb3_verbose=1
):
    """
    Train a Stable-Baselines3 RL agent on the environment returned by make_env_fn.
    The environment must return (obs, reward, done, truncated, info) from step().
    """
    vec_env = DummyVecEnv([make_env_fn])

    if algo.upper() == 'PPO':
        model = PPO("MlpPolicy", vec_env, verbose=sb3_verbose, policy_kwargs=policy_kwargs)
    elif algo.upper() == 'A2C':
        model = A2C("MlpPolicy", vec_env, verbose=sb3_verbose, policy_kwargs=policy_kwargs)
    elif algo.upper() == 'DQN':
        model = DQN("MlpPolicy", vec_env, verbose=sb3_verbose, policy_kwargs=policy_kwargs)
    else:
        raise ValueError("Unsupported algo: choose from 'PPO', 'A2C', or 'DQN'")

    model.learn(total_timesteps=total_timesteps)
    model.save(model_save_path)
    return model

def evaluate_agent(make_env_fn, model, max_steps=999999, deterministic=True):
    """
    Evaluate a trained model in a fresh environment.
    We'll unify done=done_flag or truncated for the loop condition.
    Returns a dict with final portfolio, trades_count, equity_curve, etc.
    """
    env = make_env_fn()
    obs, info = env.reset()
    done = False
    step = 0

    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done_flag, truncated, info = env.step(action)
        done = done_flag or truncated
        step += 1

    return {
        'final_value': env.portfolio_value,
        'trades_count': env.trades_count,
        'equity_curve': env.equity_curve,
        'dates': env.dates,
        'trade_history': env.trade_history,
        'time_in_market_steps': env.time_in_market_steps,
        'total_steps': env.step_counter,
        'equity_curve_per_pair': env.equity_curve_per_pair,  # for pair-level lines
        'dates_per_pair': env.dates_per_pair                 # same length/time dimension
    }
