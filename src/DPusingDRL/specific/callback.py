#!/usr/bin/env python3

# customCallback.py

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
import numpy as np
import os

class CustomEvalCallback(EvalCallback):

    def __init__(self, eval_env, best_model_save_path='', log_path='', eval_freq=10000, deterministic=True, render=False, verbose=1, **kwargs):
        super(CustomEvalCallback, self).__init__(
            eval_env,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            eval_freq=eval_freq,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            **kwargs
        )
        self.rewards = []

    def _on_step(self) -> bool:
        super()._on_step()  # Call the parent method
        if self.n_calls % self.eval_freq == 0:
            # Retrieve and log evaluation results
            episode_rewards, episode_lengths = self.eval_env.get_attr('get_episode_rewards_and_lengths')()
            mean_reward = np.mean(episode_rewards)
            self.rewards.append(mean_reward)
            self.logger.record('eval/mean_reward', mean_reward)
            if self.verbose > 0:
                print(f"Mean reward: {mean_reward} after {self.n_calls} steps")
        return True

