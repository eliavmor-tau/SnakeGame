import os

from snakeMonteCarlo.scrips.SnakeGame import SnakeGame
import numpy as np
import matplotlib.pyplot as plt
from hashlib import md5
import pickle

state_numbers = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
n_states = np.sum(state_numbers)
n_actions = 4
actions_map = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}


def eps_greedy(Q, s, eps):
    if s in Q or isinstance(Q, np.ndarray):
        if np.random.uniform(0, 1) <= eps:
            return np.random.randint(n_actions)
        else:
            return np.argmax(Q[s])
    else:
        Q[s] = np.random.uniform(0, 0.1, n_actions)
        return np.random.randint(n_actions)


def convert_state_to_number(state, d=3):
    apple_y, apple_x = np.argwhere(state == 128)[0]
    snake_head_y, snake_head_x = np.argwhere(state == 255)[0]
    snake_body = np.argwhere(state == 200)
    snake_area = []
    if snake_head_y - 1 < 0:
        snake_area[0] = 0
    if snake_head_y + 1 >= state.shape[0]:
        snake_area[2] = 0
    if snake_head_x + 1 >= state.shape[1]:
        snake_area[1] = 0
    if snake_head_x - 1 < 0:
        snake_area[3] = 0
    for pos in snake_body:
        pos_y, pos_x = pos
        if pos_y == snake_head_y - 1 and pos_x == snake_head_x:
            snake_area[0] = 0
        if pos_y == snake_head_y + 1 and pos_x == snake_head_x:
            snake_area[2] = 0
        if pos_x == snake_head_x - 1 and pos_y == snake_head_y:
            snake_area[3] = 0
        if pos_x == snake_head_x + 1 and pos_y == snake_head_y:
            snake_area[1] = 0

    y, x = [snake_head_y - apple_y], [apple_x - snake_head_x]
    angle = np.arctan2(y, x) * 180 / np.pi
    angle %= 360

    angle_vec = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    angle_idx = np.argmin(np.abs(angle_vec - angle) % 338)
    apple_angle_vec = (np.arange(8) == angle_idx).astype(np.int)
    state_vec = np.hstack([snake_area, apple_angle_vec])
    return int(np.sum(state_vec * state_numbers))


def convert_state_to_hash(state, d=3):
    apple_y, apple_x = np.argwhere(state == 128)[0]
    snake_head_y, snake_head_x = np.argwhere(state == 255)[0]
    snake_area = []
    for i in range(snake_head_y - d, snake_head_y + d + 1):
        for j in range(snake_head_x - d, snake_head_x + d + 1):
            if i == snake_head_y and j == snake_head_x:
                continue
            if i < 0 or i >= state.shape[0] or j < 0 or j >= state.shape[1]:
                snake_area.append(0)
            else:
                if state[i, j] == 200:
                    snake_area.append(0)
                else:
                    snake_area.append(1)

    y, x = [snake_head_y - apple_y], [apple_x - snake_head_x]
    angle = np.arctan2(y, x) * 180 / np.pi
    angle %= 360

    angle_vec = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    angle_idx = np.argmin(np.abs(angle_vec - angle) % 338)
    apple_angle_vec = (np.arange(8) == angle_idx).astype(np.int)
    state_vec = np.hstack([snake_area, apple_angle_vec])
    md5_state = md5(str(state_vec).encode())
    return md5_state.hexdigest()


def monte_carlo(Q, trajectory):
    G = dict()
    N = dict()
    for state, action, reward in trajectory:
        if (state, action) in G:
            G[(state, action)] += reward
            N[(state, action)] += 1
        else:
            G[(state, action)] = reward
            N[(state, action)] = 1

    for k in G:
        G[k] /= N[k]

    for state, action in G.keys():
        lr = np.power(1 / N[(state, action)], 0.8)
        Q[state][action] = Q[state][action] + lr * (G[(state, action)] - Q[state][action])
    return Q


if __name__ == "__main__":
    train = False
    os.makedirs("../pickle", exist_ok=True)
    os.makedirs("../game_output", exist_ok=True)

    if train:
        Q = dict()
        n_episodes = 100001
        eps = 0.4
        d = 3
        env = SnakeGame(height=8, width=8)
        total_rewards = []
        for e in range(n_episodes):
            screen = env.reset()
            state = convert_state_to_hash(screen, d)
            trajectory = []
            episode_reward = 0

            while True:
                action = eps_greedy(Q=Q, s=state, eps=eps)
                screen, reward, is_done, _ = env.step(action=action)
                trajectory.append((state, action, reward))
                episode_reward += reward

                if is_done:
                    episode_reward /= len(trajectory)
                    total_rewards.append(episode_reward)
                    new_trajectory = []
                    last_reward = 0
                    for state, action, reward in reversed(trajectory):
                        if reward != 0:
                            last_reward = reward
                        else:
                            last_reward = last_reward * 0.8
                        new_trajectory.insert(0, (state, action, last_reward))
                    Q = monte_carlo(Q, new_trajectory)
                    break
                state = convert_state_to_hash(screen, d)

            if not e % 500 and e > 0:
                eps *= 0.99
                print(f"Epoch {e} average reward {episode_reward}")
                print(f"Eps {eps}")
                print("-"*20)

            if not e % 2000:
                with open(f"Q_episode_{e}_d_{d}.p", "wb") as f:
                    pickle.dump(Q, f)
    else:
        with open("../pickle/Q_episode_100000_d_3.p", "rb") as f:
            Q = pickle.load(f)
            d = 3
            n_episodes = 1
            env = SnakeGame(height=8, width=8)
            for e in range(n_episodes):
                screen = env.reset()
                state = convert_state_to_hash(screen, d)
                episode_reward = 0
                idx = 1
                while True:
                    print(f"step {idx}")
                    action = eps_greedy(Q=Q, s=state, eps=0)
                    plt.title(f"{actions_map[action]}")
                    plt.imshow(screen)
                    plt.savefig(f"../game_output/{idx}.png")
                    screen, reward, is_done, _ = env.step(action=action)
                    episode_reward += reward
                    idx += 1
                    if is_done:
                        break
                    state = convert_state_to_hash(screen, d)