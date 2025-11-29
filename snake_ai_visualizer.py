import pygame
import numpy as np
import random
from stable_baselines3 import DQN
import gymnasium as gym
from gymnasium import spaces


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
        return self._get_obs(), {}

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
        if direction == 0:
            x -= 1
        elif direction == 1:
            y += 1
        elif direction == 2:
            x += 1
        elif direction == 3:
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
        features[11] = (self.grid_size - heady - 1) / self.grid_size
        features[12] = (self.grid_size - headx - 1) / self.grid_size
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
            or new_heady < 0
            or new_heady >= self.grid_size
            or new_head in self.snake
        ):
            self.done = True
            penalty = -15 - (len(self.snake) * 0.5)
            return self._get_obs(), penalty, True, False, {}

        self.snake.insert(0, new_head)
        self.steps_without_food += 1

        current_distance = self._manhattan_distance(new_head, self.food)
        distance_reward = (self.previous_distance - current_distance) * 1.0
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


class SnakeGameViewer:
    def __init__(self, env, model, cell_size=50):
        pygame.init()
        self.env = env
        self.model = model
        self.cell_size = cell_size
        self.size = env.grid_size * cell_size
        self.screen = pygame.display.set_mode(
            (self.size, self.size + 100)
        )
        pygame.display.set_caption("Snake AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 22)
        self.score = 0
        self.high_score = 0
        self.games_played = 0

    def draw_grid(self):
        for x in range(0, self.size, self.cell_size):
            pygame.draw.line(self.screen, (40, 40, 40), (x, 0), (x, self.size), 1)
        for y in range(0, self.size, self.cell_size):
            pygame.draw.line(self.screen, (40, 40, 40), (0, y), (self.size, y), 1)

    def draw(self, obs):
        self.screen.fill((20, 20, 20))

        grid = obs["grid"]

        for x in range(self.env.grid_size):
            for y in range(self.env.grid_size):
                if grid[x, y, 0] == 1:
                    color = (
                        (0, 180, 0) if grid[x, y, 2] == 0 else (0, 255, 0)
                    )
                    pygame.draw.rect(
                        self.screen,
                        color,
                        (
                            y * self.cell_size + 2,
                            x * self.cell_size + 2,
                            self.cell_size - 4,
                            self.cell_size - 4,
                        ),
                        border_radius=8,
                    )
                if grid[x, y, 1] == 1:
                    pygame.draw.circle(
                        self.screen,
                        (255, 50, 50),
                        (
                            y * self.cell_size + self.cell_size // 2,
                            x * self.cell_size + self.cell_size // 2,
                        ),
                        self.cell_size // 3,
                    )

        self.draw_grid()

        stats_y = self.size + 10

        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        high_score_text = self.font.render(
            f"Best: {self.high_score}", True, (255, 215, 0)
        )
        self.screen.blit(score_text, (10, stats_y))
        self.screen.blit(high_score_text, (self.size - 150, stats_y))

        stats_y += 35
        games_text = self.font_small.render(
            f"Games: {self.games_played}", True, (180, 180, 180)
        )
        length_text = self.font_small.render(
            f"Length: {len(self.env.snake)}", True, (180, 180, 180)
        )
        self.screen.blit(games_text, (10, stats_y))
        self.screen.blit(length_text, (self.size - 150, stats_y))

        stats_y += 30
        steps_text = self.font_small.render(
            f"Steps w/o food: {self.env.steps_without_food}",
            True,
            (255, 100, 100) if self.env.steps_without_food > 50 else (180, 180, 180),
        )
        self.screen.blit(steps_text, (10, stats_y))

        pygame.display.flip()

    def run(self, speed=10):
        obs, _ = self.env.reset()
        self.score = 0
        running = True

        print("\n Visualisation de l'IA Snake")
        print(f" Grille: {self.env.grid_size}x{self.env.grid_size}")
        print(f" Vitesse: {speed} FPS")
        print(" Ferme la fenêtre pour arrêter\n")

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = True
                        while paused:
                            for pause_event in pygame.event.get():
                                if pause_event.type == pygame.QUIT:
                                    running = False
                                    paused = False
                                elif pause_event.type == pygame.KEYDOWN:
                                    if pause_event.key == pygame.K_SPACE:
                                        paused = False
                    elif event.key == pygame.K_UP:
                        speed = min(speed + 2, 30)
                        print(f" Vitesse: {speed} FPS")
                    elif event.key == pygame.K_DOWN:
                        speed = max(speed - 2, 1)
                        print(f" Vitesse: {speed} FPS")

            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = self.env.step(action)

            if reward > 10:
                self.score = len(self.env.snake) - 1

            self.draw(obs)
            self.clock.tick(speed)

            if done or truncated:
                self.games_played += 1
                if self.score > self.high_score:
                    self.high_score = self.score
                    print(f" Nouveau record! Score: {self.high_score}")
                else:
                    print(
                        f" Game Over - Score: {self.score} | Best: {self.high_score}"
                    )

                pygame.time.wait(800)
                obs, _ = self.env.reset()
                self.score = 0

        pygame.quit()
        print(f"\n Session terminée")
        print(f"   Games joués: {self.games_played}")
        print(f"   Meilleur score: {self.high_score}")


if __name__ == "__main__":
    env = SnakeEnv(grid_size=10)

    try:
        model = DQN.load("./IA_snake_1010_SB")
        print(" Modèle chargé: snake_dqn_agent")
    except:
        print("❌ Erreur: Impossible de charger le modèle 'snake_dqn_agent'")
        print("   Assure-toi d'avoir entraîné le modèle d'abord!")
        exit(1)

    viewer = SnakeGameViewer(env, model, cell_size=50)
    print("\n Contrôles:")
    print("   SPACE: Pause/Resume")
    print("   ↑: Augmenter la vitesse")
    print("   ↓: Diminuer la vitesse")
    viewer.run(speed=8)
