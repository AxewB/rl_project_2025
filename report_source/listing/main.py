import csv
import os
import random
from collections import deque
from datetime import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.core import Env
from torch import nn, optim
from tqdm import tqdm

seed = 310
np.random.seed(seed)
np.random.default_rng(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_reward_and_loss(sim_results, name, width=6, height=6, show_fig=False):
    fig, ax = plt.subplots(1, 2, figsize=(width, height))
    fig.suptitle(f"Loss and reward history | [{name}]")

    for sim_name, sim_results in sim_results.items():
        print("plotting: ", sim_name)

        ax[0].plot(sim_results["reward"], label=f"{sim_name}")
        ax[0].set_xlabel("Episode")
        ax[0].set_ylabel("Reward")

        ax[1].plot(sim_results["loss"], label=f"{sim_name}")
        ax[1].set_xlabel("Episode")
        ax[1].set_ylabel("Loss")

    ax[0].legend()
    ax[0].grid()
    ax[1].legend()
    ax[1].grid()

    fig.tight_layout()
    os.makedirs("fig", exist_ok=True)
    fig.savefig(f"fig/{name}_loss_and_reward_history.png")

    if show_fig:
        plt.show()


def plot_fig_single(sim_results, name, width=6, height=6, show_fig=False):
    fig1, ax1 = plt.subplots(figsize=(width, height))
    fig2, ax2 = plt.subplots(figsize=(width, height))
    fig1.suptitle("История наград")
    fig2.suptitle("История потерь")

    for sim_name, sim_results in sim_results.items():
        print("plotting: ", sim_name)

        ax1.plot(sim_results["reward"], label=f"{sim_name}")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")

        ax2.plot(sim_results["loss"], label=f"{sim_name}")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Loss")

    ax1.legend()
    ax1.grid()
    ax2.legend()
    ax2.grid()

    fig1.tight_layout()
    fig2.tight_layout()
    os.makedirs("fig", exist_ok=True)
    fig1.savefig(f"fig/{name}_reward_history.png")
    fig2.savefig(f"fig/{name}_loss_history.png")

    if show_fig:
        plt.show()


def dueling_experiment(
    env, obs_size, n_actions, params, env_name: str, agent_algorithm: str = "default"
):
    simulation_results = {}

    print("Dueling experiment")
    agent = DQNAgent(
        obs_size=obs_size,
        n_actions=n_actions,
        layers=params["layers"](obs_size, n_actions),
        gamma=params["gamma"],
        epsilon=params["epsilon"],
        epsilon_decay=params["epsilon_decay"],
        epsilon_min=params["epsilon_min"],
        batch_size=params["batch_size"],
        learning_rate=params["lr"],
        algorithm=agent_algorithm,
        dueling=True,
    )

    simulation = Simulation(
        env=env,
        agent=agent,
        sim_name="dueling_experiment",
        time=datetime.now(),
        num_steps=params["num_steps"],
        num_episodes=params["num_episodes"],
    )

    simulation.run(env_name)
    simulation.save_to_csv(
        f"dueling_default_experiment___env={env_name}___agent={agent_algorithm}.csv"
    )

    simulation_results["dueling"] = {
        "reward": simulation.reward_history,
        "loss": simulation.loss_history,
    }

    plot_reward_and_loss(
        simulation_results,
        name=f"dueling_experiment___env={env_name}___agent={agent_algorithm}",
        width=15,
        height=5,
    )


def default_experiment(
    env, obs_size, n_actions, params, env_name: str, agent_algorithm: str = "dqn"
):
    simulation_results = {}

    print("Default experiment")
    agent = DQNAgent(
        obs_size=obs_size,
        n_actions=n_actions,
        layers=params["layers"](obs_size, n_actions),
        gamma=params["gamma"],
        epsilon=params["epsilon"],
        epsilon_decay=params["epsilon_decay"],
        epsilon_min=params["epsilon_min"],
        batch_size=params["batch_size"],
        learning_rate=params["lr"],
        algorithm=agent_algorithm,
    )

    simulation = Simulation(
        env=env,
        agent=agent,
        sim_name="epsilon_experiment",
        time=datetime.now(),
        num_steps=params["num_steps"],
        num_episodes=params["num_episodes"],
        modify_reward=params["modify_rewards"],
    )

    simulation.run(env_name)
    simulation.save_to_csv(
        f"default_experiment___env={env_name}___agent={agent_algorithm}.csv"
    )

    simulation_results["default"] = {
        "reward": simulation.reward_history,
        "loss": simulation.loss_history,
    }

    plot_reward_and_loss(
        simulation_results,
        name=f"default_experiment___env={env_name}___agent={agent_algorithm}",
        width=15,
        height=5,
    )


class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity=1000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.next_idx = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities, default=1.0)

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.next_idx] = (state, action, reward, next_state, done)
            self.priorities[self.next_idx] = max_priority
            self.next_idx = (self.next_idx + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")

        priorities = np.array(self.priorities)[: len(self.buffer)]
        probs = priorities**self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
            torch.tensor(indices),
            weights,
        )

    def update_priorities(self, indices, td_errors, epsilon=1e-6):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error.item()) + epsilon

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, obs_size, n_actions, layers: list[nn.Module]):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DuelingQNetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DQNAgent:
    def __init__(
        self,
        obs_size: int,
        n_actions: int,
        layers: list[nn.Module],
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.955,
        epsilon_min: float = 0.01,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        algorithm: str = "dqn",  # or double_q or prioritized
        dueling: bool = False,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dueling = dueling

        if self.dueling:
            self.q_net = DuelingQNetwork(obs_size, n_actions).to(self.device)
            self.net_target = DuelingQNetwork(obs_size, n_actions).to(self.device)
        else:
            self.q_net = QNetwork(obs_size, n_actions, layers).to(self.device)
            self.net_target = QNetwork(obs_size, n_actions, layers).to(self.device)

        self.net_target.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)

        self.obs_size = obs_size
        self.n_actions = n_actions
        self.layers = layers

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.algorithm = algorithm

        if self.algorithm == "prioritized":
            self.replay_buffer = PrioritizedReplayBuffer(10000)
        else:
            self.replay_buffer = ReplayBuffer(10000)

        self.loss: float = 0.0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values = self.q_net(state_tensor)

            return torch.argmax(q_values).item()

    def _train_prioritized_replay_buffer(self):
        if not (isinstance(self.replay_buffer, PrioritizedReplayBuffer)):
            raise ValueError(
                f"Prioritized algorithm requires PrioritizedReplayBuffer. Current is: {type(PrioritizedReplayBuffer)}"
            )

        states, actions, rewards, next_states, dones, indices, weights = (
            self.replay_buffer.sample(self.batch_size)
        )
        states = states.to(self.device)
        actions = actions.to(self.device).long()
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        indices = indices.to(self.device)
        weights = weights.to(self.device)

        loss = self._alg_prioritized(
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            weights,
        )

        return loss

    def _train_default_replay_buffer(self):
        if not (isinstance(self.replay_buffer, ReplayBuffer)):
            raise ValueError(
                f"DQN or Double DQN algorithm requires standard ReplayBuffer. Current is: {type(PrioritizedReplayBuffer)}"
            )

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = states.to(self.device)
        actions = actions.to(self.device).long()
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        if self.algorithm == "dqn":
            loss = self._alg_dqn(
                states,
                actions,
                rewards,
                next_states,
                dones,
            )
        elif self.algorithm == "double_q":
            loss = self._alg_double_q(
                states,
                actions,
                rewards,
                next_states,
                dones,
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return loss

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        if self.algorithm == "prioritized":
            loss = self._train_prioritized_replay_buffer()
        else:
            loss = self._train_default_replay_buffer()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss = loss.item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def _alg_dqn(
        self,
        states,
        actions,
        rewards,
        next_states,
        dones,
    ):
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        next_q_values = self.net_target(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values)
        return loss

    def _alg_double_q(
        self,
        states,
        actions,
        rewards,
        next_states,
        dones,
    ):
        next_q_values_online = self.q_net(next_states)
        next_actions = torch.argmax(next_q_values_online, dim=1)

        next_q_values_target = self.net_target(next_states)
        next_q_values = next_q_values_target.gather(
            1, next_actions.unsqueeze(1)
        ).squeeze(1)

        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = nn.MSELoss()(current_q_values, target_q_values.detach())
        return loss

    def _alg_prioritized(
        self,
        states,
        actions,
        rewards,
        next_states,
        dones,
        indices,
        weights,
    ):
        if not (isinstance(self.replay_buffer, PrioritizedReplayBuffer)):
            raise ValueError(
                f"Prioritized algorithm requires PrioritizedReplayBuffer. Current is: {type(PrioritizedReplayBuffer)}"
            )

        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.net_target(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        td_errors = target_q_values - current_q_values

        loss = (weights * td_errors.pow(2)).mean()
        self.replay_buffer.update_priorities(indices, td_errors.detach())
        return loss

    def update_target(self):
        self.net_target.load_state_dict(self.q_net.state_dict())


class Simulation:
    def __init__(
        self,
        env: Env,
        agent: DQNAgent,
        time: datetime,
        sim_name: str = "simulation",
        num_episodes: int = 1000,
        num_steps: int = 200,
        update_target_frequency: int = 1,
        modify_reward: bool = False,
        seed: int = 310,
    ):
        self.env = env

        self.agent = agent
        self.sim_name = sim_name
        self.time = time
        self.seed = seed

        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.update_target_frequency = update_target_frequency
        modify_reward = modify_reward

        self.reward_history = []
        self.loss_history = []

    def save_to_csv(self, filename: str):
        os.makedirs("csv", exist_ok=True)
        filepath = os.path.join("csv", filename)

        with open(filepath, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Episode",
                    "Reward",
                    "Loss",
                ]
            )

            for episode, (reward, loss) in enumerate(
                zip(self.reward_history, self.loss_history), start=1
            ):
                writer.writerow([episode, reward, loss])

        print(f"=== Saving csv to {filepath}")

    def normalize_state(self, state, env):
        low = env.observation_space.low
        high = env.observation_space.high
        return (state - low) / (high - low)

    def modify_reward(self, state):
        pos, vel = state

        vel = np.interp(vel, np.array([0, 1]), np.array([-0.5, 0.5]))

        degree = pos * 360
        degree2radian = np.deg2rad(degree)
        reward = 0.2 * (np.cos(degree2radian) + 2 * np.abs(vel))

        reward -= 0.5

        if pos > 0.98:
            reward += 20
        elif pos > 0.92:
            reward += 0
        elif pos > 0.82:
            reward += 6
        elif pos > 0.65:
            reward += 1 - np.exp(-2 * pos)

        initial_position = 0.40842572

        if vel > 0.3 and pos > initial_position + 0.1:
            reward += 1 + 2 * pos

        return reward

    def train_mountain_car(self):
        if self.env is None:
            raise ValueError("Env is none")

        self.reward_history = []

        with tqdm(range(self.num_episodes), desc="Episode") as prog_bar:
            for episode in prog_bar:
                state, _ = self.env.reset(seed=self.seed)
                episode_reward: float = 0.0
                episode_loss: float = 0.0

                for step in range(self.num_steps):
                    norm_state = self.normalize_state(state, self.env)
                    action = self.agent.select_action(state)
                    next_state, reward, done, truncated, _ = self.env.step(action)

                    # modify reward if it's mountain car
                    if self.modify_reward:
                        mod_reward = self.modify_reward(next_state)
                    else:
                        mod_reward = reward

                    self.agent.replay_buffer.push(
                        norm_state,
                        action,
                        mod_reward,
                        self.normalize_state(next_state, self.env),
                        done,
                    )

                    self.agent.train()

                    episode_reward += float(mod_reward)
                    state = next_state

                    episode_loss += float(self.agent.loss)

                    if step % self.update_target_frequency == 0:
                        self.agent.update_target()

                    if done:
                        break

                self.agent.update_epsilon()

                self.reward_history.append(episode_reward)
                self.loss_history.append(episode_loss)

                # show mean reward for last 10 episodes
                avg_reward = np.mean(self.reward_history[-10:])
                prog_bar.set_postfix(
                    {
                        "avg_reward": f"{avg_reward:.2f}",
                        "epsilon": f"{self.agent.epsilon:.3f}",
                    }
                )

    def train_lunar_lander(self):
        for episode in tqdm(range(self.num_episodes), desc="Episode"):
            state, _ = self.env.reset(seed=self.seed)
            episode_reward: float = 0.0
            episode_loss: float = 0.0

            for step in range(self.num_steps):
                action = self.agent.select_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                self.agent.replay_buffer.push(state, action, reward, next_state, done)

                state = next_state
                episode_reward += float(reward)

                self.agent.train()
                episode_loss += float(self.agent.loss)

                if done:
                    break

            self.reward_history.append(episode_reward)
            self.loss_history.append(episode_loss)

        self.agent.update_target()

    def run(self, env_name):
        if env_name == "MountainCar-v0":
            self.train_mountain_car()
        elif env_name == "LunarLander-v3":
            self.train_lunar_lander()
        else:
            raise ValueError("Sumulation.run: Wrong env name.")


def get_env(env_name):
    """
    Returns env, obs_size, n_actions, params (agent and some simulation parameters)
    """
    env = None
    n_actions = None
    obs_size = None
    params = None

    if env_name == "LunarLander-v3":
        env = gym.make(
            "LunarLander-v3",
            continuous=False,
            gravity=-10.0,
            enable_wind=False,
            wind_power=15.0,
            turbulence_power=1.5,
        )
        n_actions = 4
        obs_size = 8
        params = {
            "layers": lambda obs_size, n_actions: [
                nn.Linear(obs_size, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, n_actions),
            ],
            "gamma": 0.99,
            "epsilon": 0.9,
            "epsilon_decay": 0.955,
            "epsilon_min": 0.05,
            "batch_size": 64,
            "num_steps": 300,
            "num_episodes": 1000,
            "lr": 3e-2,
            "target_update_frequency": 10,
            "modify_rewards": False,
        }
    elif env_name == "MountainCar-v0":
        env = gym.make(
            "MountainCar-v0",
            max_episode_steps=200,
            render_mode="rgb_array",
            goal_velocity=0.1,
        )
        n_actions = 3
        obs_size = 2
        params = {
            "layers": lambda obs_size, n_actions: [
                nn.Linear(obs_size, 12),
                nn.ReLU(),
                nn.Linear(12, 8),
                nn.ReLU(),
                nn.Linear(8, n_actions),
            ],
            "gamma": 1.0,
            "epsilon": 0.999,
            "epsilon_decay": 0.997,
            "epsilon_min": 0.01,
            "batch_size": 64,
            "num_steps": 200,
            "num_episodes": 1000,
            "lr": 75e-5,
            "target_update_frequency": 10,
            "modify_rewards": True,
        }
    else:
        raise ValueError("Wrong env name.")

    return env, obs_size, n_actions, params


if __name__ == "__main__":
    env = None
    actions = None
    obs = None

    env_names = [
        "MountainCar-v0",
        "LunarLander-v3",
    ]

    experiments = [
        ("dqn", default_experiment),
        ("double_q", default_experiment),
        ("prioritized", default_experiment),
        ("dqn", dueling_experiment),
    ]

    for env_name in env_names:
        for agent_algorithm, experiment in experiments:
            env, obs, actions, params = get_env(env_name)
            experiment(
                env, obs, actions, params, env_name, agent_algorithm=agent_algorithm
            )
