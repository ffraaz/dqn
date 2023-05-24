"""
Deep Q-Network (DQN)
Papers: https://www.nature.com/articles/nature14236, https://arxiv.org/abs/1312.5602
Most hyperparameters are taken from here: https://docs.cleanrl.dev/rl-algorithms/dqn/
"""
import argparse
import os
import platform
import random
import time
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import functional as T  # type: ignore

FRAME_SIZE = 84


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", type=int, default=int(1e7))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_frequency", type=int, default=4)
    parser.add_argument("--memory_size", type=int, default=int(500_000))
    parser.add_argument("--target_network_update_frequency", type=int, default=1000)
    parser.add_argument("--learning_start", type=int, default=80_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon_start", type=float, default=1)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--exploration_fraction", type=float, default=0.1)
    parser.add_argument("--save_interval", type=int, default=100_000)
    parser.add_argument("--log_interval", type=int, default=1000)
    args = parser.parse_args()
    train(args)


def train(args: argparse.Namespace) -> None:  # pylint: disable=R0914
    env = gym.make("ALE/Pong-v5", repeat_action_probability=0)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_network = DQN(n_actions=env.action_space.n).to(device)  # type: ignore
    target_network = deepcopy(q_network)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=args.lr)
    memory = Memory(args.memory_size, state_shape=(4, FRAME_SIZE, FRAME_SIZE), device=device)
    epsilon_schedule = EpsilonSchedule(
        args.epsilon_start, args.epsilon_end, args.exploration_fraction, args.n_iter
    )
    validation_set = _validation_set(env, device)
    save_dir = _save_dir()
    writer = SummaryWriter(log_dir=save_dir)
    i = 0
    _log_validation_Q(q_network, validation_set, writer, i)

    while True:
        frame, _ = env.reset()
        next_state = _initial_state(frame, device)
        terminal = False
        while not terminal:
            state = next_state
            action = _epsilon_greedy(q_network, state, env, epsilon_schedule, i)
            frame, reward, terminal, _, info = env.step(action)
            next_state = _next_state(state, frame)
            memory.add(state, action, reward, next_state, terminal)  # type: ignore

            if i >= args.learning_start and i % args.train_frequency == 0:
                loss = _loss(q_network, target_network, memory, args.batch_size, args.gamma)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            i += 1
            if i % args.target_network_update_frequency == 0:
                target_network = deepcopy(q_network)
            if i % args.log_interval == 0:
                _log_validation_Q(q_network, validation_set, writer, i)
            if terminal:
                writer.add_scalar("reward of last episode", info["episode"]["r"], i)
            if i % args.save_interval == 0 or i == args.n_iter:
                _save_checkpoint(q_network, optimizer, save_dir, i)
            if i == args.n_iter:
                return


class DQN(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=-3),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def max_Q(self, state: torch.Tensor) -> torch.Tensor:
        return self(state).max(dim=-1, keepdim=True)[0]


class Memory:  # pylint: disable=R0902
    def __init__(self, max_size: int, state_shape: tuple[int, int, int], device: torch.device):
        self.max_size = max_size
        self.device = device
        self.n = 0
        self.states = torch.zeros(max_size, *state_shape)
        self.actions = torch.zeros(max_size, 1, dtype=torch.long)
        self.rewards = torch.zeros(max_size, 1)
        self.next_states = torch.zeros(max_size, *state_shape)
        self.terminals = torch.zeros(max_size, 1)

    def add(  # pylint: disable=R0913
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        terminal: bool,
    ) -> None:
        i = self.n % self.max_size
        self.n += 1
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.terminals[i] = terminal

    def sample(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        i = torch.randint(min(self.n, self.max_size), (batch_size,))
        return (
            self.states[i].to(self.device),
            self.actions[i].to(self.device),
            self.rewards[i].to(self.device),
            self.next_states[i].to(self.device),
            self.terminals[i].to(self.device),
        )


def _loss(
    q_network: DQN,
    target_network: DQN,
    memory: Memory,
    batch_size: int,
    gamma: float,
) -> torch.Tensor:
    state, action, reward, next_state, terminal = memory.sample(batch_size)
    target = reward + (1 - terminal) * gamma * target_network.max_Q(next_state)
    Q = q_network(state).gather(dim=-1, index=action)
    assert Q.shape == target.shape == (batch_size, 1)
    assert Q.requires_grad
    assert not target.requires_grad
    return F.mse_loss(Q, target)


def _epsilon_greedy(
    q_network: DQN,
    state: torch.Tensor,
    env: gym.Env,
    epsilon_schedule: "EpsilonSchedule",
    i: int,
) -> torch.Tensor:
    if random.random() < epsilon_schedule(i):
        return env.action_space.sample()
    with torch.no_grad():
        return q_network(state).argmax()


class EpsilonSchedule:  # pylint: disable=R0903
    def __init__(
        self,
        epsilon_start: float,
        epsilon_end: float,
        exploration_fraction: float,
        n_iter: int,
    ):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.slope = (epsilon_end - epsilon_start) / (exploration_fraction * n_iter)

    def __call__(self, i: int) -> float:
        return max(self.slope * i + self.epsilon_start, self.epsilon_end)


def _initial_state(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    return preprocess(frame, device).broadcast_to((4, -1, -1))


def _next_state(state: torch.Tensor, frame: np.ndarray) -> torch.Tensor:
    return torch.cat((state[1:], preprocess(frame, state.device)))


def preprocess(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(frame).to(device)
    x = x.permute(2, 0, 1)
    x = T.rgb_to_grayscale(x)
    x = F.interpolate(x.unsqueeze(0), size=(110, FRAME_SIZE)).squeeze(0)
    x = T.crop(x, top=18, left=0, height=FRAME_SIZE, width=FRAME_SIZE)
    x = x - x.min()
    x = x / x.max()
    return x


def _validation_set(env: gym.Env, device: torch.device) -> torch.Tensor:
    frame, _ = env.reset()
    next_state = _initial_state(frame, device)
    states = [next_state]
    terminal = False
    while not terminal:
        state = next_state
        action = env.action_space.sample()
        frame, _, terminal, _, _ = env.step(action)
        next_state = _next_state(state, frame)
        states.append(next_state)
    assert len(states) >= 500
    return torch.stack(states)


def _log_validation_Q(
    q_network: DQN,
    validation_set: torch.Tensor,
    writer: SummaryWriter,
    i: int,
) -> None:
    mean_validation_Q = q_network.max_Q(validation_set).mean()
    print(f"{i}: mean validation Q: {mean_validation_Q}")
    writer.add_scalar("mean validation Q", mean_validation_Q, i)


def _save_checkpoint(
    q_network: DQN,
    optimizer: torch.optim.Optimizer,
    save_dir: str,
    i: int,
) -> None:
    torch.save(
        {
            "i": i,
            "state_dict": q_network.state_dict(),
        },
        os.path.join(save_dir, "checkpoint.pt"),
    )
    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer_checkpoint.pt"))


def _save_dir() -> str:
    home = os.path.expanduser("~")
    path = "Desktop/experiments/rl/runs" if platform.system() == "Darwin" else "rl/runs"
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    return os.path.join(home, path, timestamp)


if __name__ == "__main__":
    main()
