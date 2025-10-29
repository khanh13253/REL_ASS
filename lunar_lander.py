#!/usr/bin/env python

"""LunarLander environment class for RL-Glue-py."""

from environment import BaseEnvironment
import numpy as np
import gymnasium as gym

class LunarLanderEnvironment(BaseEnvironment):
    def env_init(self, env_info={}):
        """
        Setup for the environment called when the experiment first starts.
        """
        # Lấy seed từ env_info (nếu có)
        self.seed = env_info.get("seed", 0)

        # Tạo môi trường — dùng render_mode='human' nếu bạn muốn hiển thị mô phỏng
        self.env = gym.make("LunarLander-v3", render_mode="human")

        # Reset với seed (API mới)
        obs, info = self.env.reset(seed=self.seed)

        # Lưu trạng thái đầu tiên
        self.reward_obs_term = (0.0, obs, False)

    def env_start(self):
        """Called before the agent starts."""
        obs, info = self.env.reset(seed=self.seed)
        self.reward_obs_term = (0.0, obs, False)
        return obs

    def env_step(self, action):
        """A step taken by the environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.reward_obs_term = (reward, obs, done)
        return self.reward_obs_term

    def env_cleanup(self):
        """Close environment."""
        self.env.close()

    def env_message(self, message):
        if message == "get_seed":
            return self.seed
        return None
