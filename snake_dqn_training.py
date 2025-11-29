import os
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback


class SnakeEnv(gym.Env):
    def __init__(self, grid_size=10):
        super(SnakeEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0, high=1, shape=(grid_size, grid_size, 3), dtype=np.float32
                ),
                "features": spaces.Box(low=-10, high=10, shape=(16,), dtype=np.float32),
            }
        )

        self.opposite_actions = {0: 2, 1: 3, 2: 0, 3: 1}
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = random.choice([0, 1, 2, 3])
        self.food = self._new_food()
        self.done = False
        self.steps_without_food = 0
        self.previous_distance = self._manhattan_distance(self.snake[0], self.food)
        self.total_reward = 0
        obs = self._get_obs()
        return obs, {}

    def _new_food(self):
        free = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in self.snake
        ]
        return random.choice(free) if free else self.snake[0]

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_danger_in_direction(self, head, direction):
        x, y = head
        if direction == 0:  # haut
            x -= 1
        elif direction == 1:  # droite
            y += 1
        elif direction == 2:  # bas
            x += 1
        elif direction == 3:  # gauche
            y -= 1

        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return 1.0
        if (x, y) in self.snake[:-1]:
            return 1.0
        return 0.0

    def _get_obs(self):
        grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        for x, y in self.snake:
            grid[x, y, 0] = 1.0
        fx, fy = self.food
        grid[fx, fy, 1] = 1.0
        headx, heady = self.snake[0]
        grid[headx, heady, 2] = 1.0

        features = np.zeros(16, dtype=np.float32)

        for i in range(4):
            features[i] = self._get_danger_in_direction(self.snake[0], i)

        features[4] = 1.0 if fx < headx else 0.0
        features[5] = 1.0 if fy > heady else 0.0
        features[6] = 1.0 if fx > headx else 0.0
        features[7] = 1.0 if fy < heady else 0.0

        features[8] = self._manhattan_distance(self.snake[0], self.food) / (
            self.grid_size * 2
        )

        features[9] = len(self.snake) / (self.grid_size * self.grid_size)

        features[10] = headx / self.grid_size
        features[11] = (
            self.grid_size - heady - 1
        ) / self.grid_size
        features[12] = (
            self.grid_size - headx - 1
        ) / self.grid_size
        features[13] = heady / self.grid_size

        features[14] = self.direction / 3.0

        features[15] = min(
            self.steps_without_food / (self.grid_size * self.grid_size), 1.0
        )

        return {"grid": grid, "features": features}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        if hasattr(action, "item"):
            action = int(action.item())
        else:
            action = int(action)

        if len(self.snake) > 1 and action == self.opposite_actions.get(self.direction):
            action = self.direction

        headx, heady = self.snake[0]
        new_headx, new_heady = headx, heady

        if action == 0:
            new_headx -= 1
        elif action == 1:
            new_heady += 1
        elif action == 2:
            new_headx += 1
        elif action == 3:
            new_heady -= 1

        if (
            new_headx < 0
            or new_headx >= self.grid_size
            or new_heady < 0
            or new_heady >= self.grid_size
        ):
            action = self.direction
            new_headx, new_heady = headx, heady
            if action == 0:
                new_headx -= 1
            elif action == 1:
                new_heady += 1
            elif action == 2:
                new_headx += 1
            elif action == 3:
                new_heady -= 1

        self.direction = action
        new_head = (new_headx, new_heady)

        if (
            new_headx < 0
            or new_headx >= self.grid_size
            or heady < 0
            or new_heady >= self.grid_size
            or new_head in self.snake
        ):
            self.done = True
            penalty = -15 - (
                len(self.snake) * 0.5
            )
            return self._get_obs(), penalty, True, False, {}

        self.snake.insert(0, new_head)
        self.steps_without_food += 1

        current_distance = self._manhattan_distance(new_head, self.food)

        distance_reward = (
            self.previous_distance - current_distance
        ) * 1.0
        self.previous_distance = current_distance

        if new_head == self.food:
            base_food_reward = 20
            time_bonus = max(0, 5 - self.steps_without_food * 0.02)
            length_bonus = len(self.snake) ** 1.2 * 0.5

            reward = base_food_reward + time_bonus + length_bonus
            self.food = self._new_food()
            self.steps_without_food = 0
            self.previous_distance = self._manhattan_distance(self.snake[0], self.food)
        else:
            self.snake.pop()
            survival_reward = 0.1
            reward = survival_reward + distance_reward

            if self.steps_without_food > self.grid_size * self.grid_size:
                timeout_penalty = -0.1 * (
                    self.steps_without_food / (self.grid_size * self.grid_size)
                )
                reward += timeout_penalty

        self.total_reward += reward
        return self._get_obs(), reward, self.done, False, {}


class ProgressCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(ProgressCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_mean_reward = -np.inf

    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            self.episode_rewards.append(self.model.ep_info_buffer[-1]["r"])
            self.episode_lengths.append(self.model.ep_info_buffer[-1]["l"])

            if len(self.episode_rewards) % 5000 == 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                mean_length = np.mean(self.episode_lengths[-100:])
                print(
                    f"\n Episodes: {len(self.episode_rewards)} | "
                    f"Mean Reward: {mean_reward:.2f} | "
                    f"Mean Length: {mean_length:.2f}"
                )

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save("./snake_dqn_best")
                    print(
                        f" Nouveau meilleur modèle sauvegardé! (reward: {mean_reward:.2f})"
                    )

        return True


if __name__ == "__main__":
    env = SnakeEnv(grid_size=10)
    model_path = "./IA_snake_1010_SB.zip"

    device = "cuda"

    if os.path.exists(model_path):
        print(" Modèle existant trouvé, reprise de l'entraînement...")
        model = DQN.load(model_path, env=env, device=device, verbose=0)
    else:
        print(" Aucun modèle trouvé, création d'un nouveau...")
        model = DQN(
            "MultiInputPolicy",
            env,
            device=device,
            verbose=0,
            learning_rate=5e-4,
            buffer_size=500_000,
            learning_starts=1000,
            batch_size=128,
            tau=0.005,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.4,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
        )

    callback = ProgressCallback()

    print("\n Début de l'entraînement...")
    print(f" Taille de la grille: {env.grid_size}x{env.grid_size}")
    print(
        f" Features: {env.observation_space['features'].shape[0]} features vectorielles"
    )

    model.learn(total_timesteps=1, callback=callback)