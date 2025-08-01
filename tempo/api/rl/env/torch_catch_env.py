# type: ignore

import gymnasium as gym
import numpy as np
import torch


def draw_ball(boards, ball_x, ball_y, r, color, dev):
    batch, channels, height, width = boards.shape

    # Create a meshgrid with the same shape as your boards tensor
    y_range = (
        torch.arange(height, device=dev).view(1, 1, -1, 1).expand(batch, channels, height, width)
    )
    x_range = (
        torch.arange(width, device=dev).view(1, 1, 1, -1).expand(batch, channels, height, width)
    )

    # Calculate the distance from each grid point to the ball's position
    ball_y = ball_y.view(-1, 1, 1, 1)
    dist_x = x_range - ball_x[:, None, None, None]
    dist_y = y_range - ball_y
    dist = torch.sqrt(dist_x * dist_x + dist_y * dist_y)

    # Create a mask by checking if the distance is less than or equal to the
    # circle's radius
    mask = dist <= r

    # Use the mask to set the circle area to 1 in the boards tensor
    boards[mask] = color

    return boards


def draw_paddle(boards, paddle_x, paddle_y, thickness, length, color, dev):
    batch, channels, height, width = boards.shape

    # Create a meshgrid with the same shape as boards
    y_range = (
        torch.arange(height, device=dev).view(1, 1, -1, 1).expand(batch, channels, height, width)
    )
    x_range = (
        torch.arange(width, device=dev).view(1, 1, 1, -1).expand(batch, channels, height, width)
    )

    # Calculate the paddle bounds
    paddle_y = torch.tensor(paddle_y, device=dev).expand(batch, 1, 1, 1)
    top_bound = paddle_y - thickness // 2
    bottom_bound = paddle_y + thickness // 2
    left_bound = paddle_x[:, None, None, None] - length // 2
    right_bound = paddle_x[:, None, None, None] + length // 2

    # Create a mask by checking if the meshgrid points are within the paddle bounds
    mask = (
        (y_range >= top_bound)
        & (y_range <= bottom_bound)
        & (x_range >= left_bound)
        & (x_range <= right_bound)
    )

    # Use the mask to set the paddle area to color in the boards tensor
    boards[mask] = color

    return boards


cuda_dev = torch.device("cuda")


class TorchCatchEnv(gym.vector.VectorEnv):
    def __init__(
        self,
        num_envs: int = 1000,
        rows: int = 256,
        columns: int = 256,
        channels: int = 1,
        seed: int = 42,
        dev: torch.device = cuda_dev,
        homogeneous_termination: bool = True,
        paddle_length: int = 5,
        paddle_thickness: int = 2,
        ball_radius: int = 1,
        paddle_y_offset: int = 4,
        ball_color: float = 255.0,
        paddle_color: float = 128.0,
    ):
        self.num_envs = num_envs
        self._rows = rows
        self._columns = columns
        self._channels = channels
        self._seed = seed
        self._homogeneous_termination = homogeneous_termination
        self.num_actions = 3
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=255.0,
            shape=(self._channels, self._columns, self._rows),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(3)
        self.dev = dev
        self.batch_indices = torch.arange(num_envs, device=dev)

        self._paddle_y = torch.tensor(self._rows - paddle_y_offset).to(self.dev)
        self._ball_radius = torch.tensor(ball_radius).to(self.dev)
        self._paddle_length = torch.tensor(paddle_length).to(self.dev)
        self._paddle_thickness = torch.tensor(paddle_thickness).to(self.dev)
        torch.manual_seed(seed)

        self.zero_reward = torch.tensor(0.0, device=self.dev)
        self.one_reward = torch.tensor(1.0, device=self.dev)
        self.neg_one_reward = torch.tensor(-1.0, device=self.dev)

        self.done = False
        self.homogeneous_dones = torch.ones((num_envs,), device=self.dev).to(torch.bool)
        self.homogeneous_not_dones = torch.zeros((num_envs,), device=self.dev).to(torch.bool)
        self.homogeneous_ep_len = rows - paddle_y_offset
        self.steps_since_reset = 0

        self._ball_color = torch.tensor(ball_color, device=self.dev)
        self._paddle_color = torch.tensor(paddle_color, device=self.dev)

    def _compute_reward(self, dones):
        paddle_left = self._paddle_x - (self._paddle_length // 2)
        paddle_right = self._paddle_x + (self._paddle_length // 2)
        ball_in_paddle = torch.logical_and(
            self._ball_x <= paddle_right, self._ball_x >= paddle_left
        )
        successful_catch_on_this_step = torch.logical_and(dones, ball_in_paddle)
        failed_catch_on_this_step = torch.logical_and(dones, ~ball_in_paddle)
        rewards = torch.where(successful_catch_on_this_step, self.one_reward, self.zero_reward)
        rewards = torch.where(failed_catch_on_this_step, self.neg_one_reward, rewards)

        return rewards

    def _is_terminal(self):
        if self._homogeneous_termination:
            if self.steps_since_reset >= self.homogeneous_ep_len:
                return self.homogeneous_dones
            else:
                return self.homogeneous_not_dones
        else:
            dones = self._ball_y >= self._paddle_y
            self.done = torch.all(dones)
            return dones

    def _render_observations(self):
        boards = torch.zeros(
            (self.num_envs, self._channels, self._rows, self._columns),
            device=self.dev,
            dtype=torch.float32,
        )
        draw_ball(
            boards,
            self._ball_x,
            self._ball_y,
            r=self._ball_radius,
            color=self._ball_color,
            dev=self.dev,
        )
        draw_paddle(
            boards,
            self._paddle_x,
            self._paddle_y,
            thickness=self._paddle_thickness,
            length=self._paddle_length,
            color=self._paddle_color,
            dev=self.dev,
        )
        self.obs = boards
        return boards

    def reset(self):
        self.steps_since_reset = 0
        self.done = False
        if self._homogeneous_termination:
            self._ball_y = torch.zeros((self.num_envs,), device=self.dev)
        else:
            self._ball_y = torch.randint(
                high=int(self._rows * (5 / 6)), size=(self.num_envs,), device=self.dev
            )

        self._ball_x = torch.randint(high=self._columns, size=(self.num_envs,), device=self.dev)
        self._paddle_x = torch.randint(high=self._columns, size=(self.num_envs,), device=self.dev)
        return self._render_observations()

    def step(self, action):
        if self.done:
            raise Exception("Step called on done env")
        a = action - 1
        self._paddle_x = torch.clip(self._paddle_x + a, 0, self._columns - 1)
        self._ball_y += 1
        self._render_observations()
        dones = self._is_terminal()
        rewards = self._compute_reward(dones)
        return self.obs, rewards, dones, {}

    def render(self, mode="human"):
        obs = self.obs[0]
        if mode == "rgb":
            return obs
        elif mode == "human":
            from matplotlib import pyplot as plt

            img = (obs / 255.0).permute(1, 2, 0).cpu().numpy()
            plt.imshow(img)
            plt.axis("off")
            plt.show(block=False)
            plt.pause(0.00001)
            plt.clf()
        else:
            raise Exception("Unknown render type")
