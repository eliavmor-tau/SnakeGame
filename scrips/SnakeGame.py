import gym
import numpy as np
import time
import matplotlib.pyplot as plt


class Snake:
    def __init__(self, x, y, action):
        self.coordinates = [(x, y)]
        self.last_action = action

    def update(self, x, y, action):
        self.coordinates = [(x, y)] + self.coordinates[:-1]
        self.last_action = action

    def expand(self, x, y, action):
        tail = self.coordinates[-1]
        self.update(x, y, action)
        self.coordinates.append(tail)

    def get_coordinates(self):
        return self.coordinates

    def get_head(self):
        return self.coordinates[0]

    def get_tail(self):
        return self.coordinates[-1]

    def get_last_action(self):
        return self.last_action

    def check_crash(self, x, y):
        for i in range(len(self) - 1):
            snake_x, snake_y = self.__getitem__(i)
            if x == snake_x and y == snake_y:
                return True
        return False

    def __getitem__(self, item):
        assert item < len(self.coordinates)
        return self.coordinates[item]

    def __len__(self):
        return len(self.coordinates)


class SnakeGame:

    def __init__(self, height=10, width=10):
        self.height = height
        self.width = width
        self.n_apples = 1
        self.apple = 128
        self.board = np.zeros((self.height, self.width), dtype=np.int)
        self.action_space = gym.spaces.Discrete(n=4)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width), dtype=np.int)
        snake_x = np.random.randint(0, self.width)
        snake_y = np.random.randint(0, self.height)
        self.snake = Snake(x=snake_x, y=snake_y, action=self.action_space.sample())
        self.apples = []
        for i in range(self.n_apples):
            self.set_new_apple()
        self.update_board(x=snake_x, y=snake_y, is_snake=True)
        self.action_map = {0: 'Up', 1: 'Right', 2: 'Down', 3: 'Left'}
        self.last_snake_apple_distance = self._get_snake_apple_distance()

    def _get_snake_apple_distance(self):
        sx, sy = self.snake.get_head()
        ax, ay = self.apples[0]
        return np.sqrt(np.power(ax - sx, 2) + np.power(ay - sy, 2))

    def update_board(self, x, y, is_apple=False, is_snake=False):
        assert is_apple and not is_snake or is_snake and not is_apple
        assert 0 <= x < self.width and 0 <= y < self.height
        if is_apple:
            self.board[y, x] = self.apple
        elif is_snake:
            coordinates = self.snake.get_coordinates()
            for idx, coordinate in enumerate(coordinates):
                x, y = coordinate
                if idx != 0:
                    color = 200
                else:
                    color = 255
                self.board[y, x] = color

    def clean_snake_tail(self):
        x, y = self.snake.get_tail()
        self.board[y, x] = 0

    def get_random_available_pos(self):
        snake_coordinates = set(self.snake.get_coordinates())
        apple_coordinates = set(self.apples)
        occupied_positions = snake_coordinates.union(apple_coordinates)
        y, x = np.random.randint(self.height), np.random.randint(self.width)
        while (x, y) in occupied_positions:
            y, x = np.random.randint(self.height), np.random.randint(self.width)
        return [x, y]

    def set_new_apple(self):
        x, y = self.get_random_available_pos()
        self.apples.append((x, y))
        self.update_board(x=x, y=y, is_apple=True)

    def render(self):
        print(f"Last action: {self.action_map[self.snake.get_last_action()]}")
        print(self.board)

    def next_coordinate(self, action):
        assert action >=0 and action <= self.action_space.n
        head_x, head_y = self.snake.get_head()
        sec_x, sec_y = head_x, head_y
        if len(self.snake) > 1:
            sec_x, sec_y = self.snake[1]

        if self.action_map[action] == 'Up' and (sec_y != (head_y - 1)):
            return head_x, head_y - 1, action

        elif self.action_map[action] == 'Right' and (sec_x != (head_x + 1)):
            return head_x + 1, head_y, action

        elif self.action_map[action] == 'Down' and (sec_y != (head_y + 1)):
            return head_x, head_y + 1, action

        elif self.action_map[action] == 'Left' and (sec_x != (head_x - 1)):
            return head_x - 1, head_y, action
        # illegal move, continue with last action
        else:
            return self.next_coordinate(self.snake.get_last_action())

    def step(self, action):
        assert action >=0 and action <= self.action_space.n
        next_x, next_y, action = self.next_coordinate(action=action)
        # cross board bounds
        if next_x < 0 or next_x >= self.width or next_y < 0 or next_y >= self.height:
            self.snake.last_action = action
            return self.board, -100, True, {'apples': self.apples}

        # snake crash itself
        if self.snake.check_crash(next_x, next_y):
            self.snake.last_action = action
            return self.board, -100, True, {'apples': self.apples}

        self.clean_snake_tail()
        reward = 0
        if (next_x, next_y) in self.apples:
            self.apples.remove((next_x, next_y))
            self.snake.expand(x=next_x, y=next_y, action=action)
            self.set_new_apple()
            self.last_snake_apple_distance = self._get_snake_apple_distance()
            reward = 30
        else:
            self.snake.update(x=next_x, y=next_y, action=action)
            current_snake_apple_distance = self._get_snake_apple_distance()
            reward = -0.5 if current_snake_apple_distance - self.last_snake_apple_distance > 0 else 0
            self.last_snake_apple_distance = current_snake_apple_distance

        self.update_board(x=next_x, y=next_y, is_snake=True)
        return self.board, reward, False, {'apples': self.apples}

    def reset(self):
        self.board = np.zeros((self.height, self.width), dtype=np.int)
        snake_x = np.random.randint(0, self.width)
        snake_y = np.random.randint(0, self.height)
        self.snake = Snake(x=snake_x, y=snake_y, action=self.action_space.sample())
        self.update_board(x=snake_x, y=snake_y, is_snake=True)
        self.apples = []
        for i in range(self.n_apples):
            self.set_new_apple()
        self.last_snake_apple_distance = self._get_snake_apple_distance()
        return self.board