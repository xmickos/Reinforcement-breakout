import gym
import numpy as np
import torch
from PIL import Image


class DQNBreakout(gym.Wrapper):
    def __init__(self, render_mode='rgb_array', repeat=4, device='cpu'):
        env = gym.make("Breakout-v4", render_mode=render_mode)

        super(DQNBreakout, self).__init__(env)

        self.img_shape = (84, 84)
        self.repeat = repeat
        self.lives = env.ale.lives()
        self.device = device
        self.frame_buffer = []

    def step(self, action):
        total_reward = 0
        done = False

        for i in range(self.repeat):
            observation, reward, done, truncated, info = self.env.step(action)

            total_reward += reward

            # print(info, total_reward)

            current_lives = info['lives']

            if current_lives < self.lives:
                total_reward -= 1
                self.lives = current_lives

            # print(f"lives: {self.lives}, total_reward: {total_reward}")

            self.frame_buffer.append(observation)

            if done:
                break

        max_frame = np.max(self.frame_buffer[-2:], axis=0)
        # max_frame = max_frame.to(self.device)
        max_frame = self.process_observation(observation)
        max_frame = max_frame.to(self.device)

        total_reward = torch.tensor(total_reward).view(1, -1).float()
        total_reward.to(self.device)

        done = torch.tensor(done).view(1, -1).float()
        done.to(self.device)

        return max_frame, total_reward, done, info

    def reset(self):
        self.frame_buffer = []

        observation, _ = self.env.reset()

        self.lives = self.env.ale.lives()

        observation = self.process_observation(observation)

        return observation

    def process_observation(self, observation):
        img = Image.fromarray(observation)
        img = img.resize(self.img_shape)
        img = img.convert("L")
        img = np.array(img)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.unsqueeze(0)
        img = img / 255.0

        img = img.to(self.device)

        # pupupu
        return img
