from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np

SMALL = 1e0
MEDIUM = 1e3
LARGE = 1e6


def has_palette(arr: np.ndarray | int):
    return arr > 0


def move_palette(origin: int, destination: int) -> tuple[int, int, bool]:
    """
    Transfer a palette from origin to destination, if possible, retuning new origin,
    destination, and whether the transfer was successful
    """

    if origin == 0:
        return origin, destination, False
    if destination != 0:
        return origin, destination, False
    return 0, origin, True


class Port(ABC):
    """
    Represents loading or unloading
    """

    def __init__(
        self, size: int, palette_types: int, prob: float, render_mode="console"
    ):
        super(Port, self).__init__()
        self.size = size
        self.palette_types = palette_types
        self.prob = prob
        self.render_mode = render_mode

        # The port can have a palette or be empty
        self.observation_space = gym.spaces.MultiDiscrete(
            tuple((palette_types + 1) * np.ones(self.size, dtype=np.int_))
        )

        self.reset()

    def reset(self):
        self.state = np.zeros((self.size,), dtype=np.int_)

    @abstractmethod
    def step(self, grid_cell: np.ndarray) -> np.ndarray:
        pass

    def reward(self):
        return -1 * has_palette(self.state).sum() * LARGE

    def _generate_palette(self, old_palette: int, dist: np.ndarray) -> int:
        """
        Generate a new palette randomly.  If an old palette is present, always return that
        """
        if has_palette(old_palette):
            return old_palette

        if not np.random.choice([True, False], p=[self.prob, 1 - self.prob]):
            return 0
        if dist.sum() == 0:
            return 0

        return np.random.choice(np.arange(1, self.palette_types + 1), p=dist)

    def generate_palettes(self, dist: np.ndarray) -> None:
        """
        Generate new palettes randomly.  If an old palette is present, always return that
        """

        self.state = np.vectorize(lambda x: self._generate_palette(x, dist))(self.state)


class Loading(Port):
    def step(self, grid_cell: np.ndarray) -> np.ndarray:
        for k in range(self.size):
            origin, destination, _ = move_palette(self.state[k], grid_cell[k])
            self.state[k] = origin
            grid_cell[k] = destination
        return grid_cell


class Unloading(Port):
    @staticmethod
    def unload(origin: int, destination: int) -> tuple[int, int, bool]:
        """
        Unload a palette to unloading, where destination needs to match origin
        """

        if origin == 0:
            return origin, destination, False
        if destination != origin:
            return origin, destination, False
        return 0, 0, True

    def step(self, grid_cell: np.ndarray) -> np.ndarray:
        for k in range(self.size):
            origin, destination, _ = self.unload(grid_cell[k], self.state[k])
            self.state[k] = destination
            grid_cell[k] = origin
        return grid_cell


class Logistics(gym.Env):
    """
    A warehouse logistics problem
    """

    def __init__(
        self,
        n_rows: int = 4,
        n_cols: int = 4,
        palette_types: int = 4,
        prob_loading: float = 0.05,
        prob_unloading: float = 0.04,
        render_mode="console",
    ):
        super(Logistics, self).__init__()
        # parameters
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.palette_types = palette_types
        self.prob_loading = prob_loading
        self.prob_unloading = prob_unloading
        self.render_mode = render_mode

        self.loading_row = 0
        self.unloading_row = n_rows - 1

        # Port code
        self.loading = Loading(
            size=n_cols,
            palette_types=palette_types,
            prob=prob_loading,
            render_mode=render_mode,
        )
        self.unloading = Unloading(
            size=n_cols,
            palette_types=palette_types,
            prob=prob_unloading,
            render_mode=render_mode,
        )

        # X, Y coordinates can move in one of 4 cardinal directions
        self.action_space = gym.spaces.MultiDiscrete([self.n_rows, self.n_cols, 4])

        # The grid and loading and unloading can each have a palette or be empty
        self.observation_space = gym.spaces.Dict(
            {
                "grid": gym.spaces.MultiDiscrete(
                    tuple([self.palette_types + 1] * (n_rows * n_cols))
                ),
                "loading": self.loading.observation_space,
                "unloading": self.unloading.observation_space,
            }
        )

        self._reset()

    def _reset(self):
        self.grid = np.zeros((self.n_rows, self.n_cols), dtype=np.int_)
        self.loading.reset()
        self.unloading.reset()

    def _observation(self):
        return {
            "grid": tuple(self.grid.ravel()),
            "loading": tuple(self.loading.state),
            "unloading": tuple(self.unloading.state),
        }

    def _info(self):
        return {}

    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        self._reset()

        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return self._observation(), self._info()

    def _destination(self, x, y, direction):
        if direction == 0:
            return x, y - 1
        elif direction == 1:
            return x + 1, y
        elif direction == 2:
            return x, y + 1
        elif direction == 3:
            return x - 1, y
        else:
            raise ValueError(
                f"Received invalid direction={direction} which is not part of the action space"
            )

    def valid_coords(self, x, y):
        return 0 <= x < self.n_rows and 0 <= y < self.n_cols

    def _step_return(self, reward, terminated, truncated):
        return (
            self._observation(),
            reward,
            terminated,
            truncated,
            self._info(),
        )

    def step(self, action):
        reward = 0.0

        # Load palettes if we can, penalty if not
        self.grid[self.loading_row, :] = self.loading.step(
            self.grid[self.loading_row, :]
        )
        reward += self.loading.reward()

        # Unload palettes if we can, penalty if not
        self.grid[self.unloading_row, :] = self.unloading.step(
            self.grid[self.unloading_row, :]
        )
        reward += self.unloading.reward()

        # Move palette based on action
        orig_x, orig_y, direction = action
        dest_x, dest_y = self._destination(orig_x, orig_y, direction)

        if self.valid_coords(dest_x, dest_y):
            origin, destination, success = move_palette(
                self.grid[orig_x, orig_y], self.grid[dest_x, dest_y]
            )
            self.grid[orig_x, orig_y] = origin
            self.grid[dest_x, dest_y] = destination

        # Generate Loading Palettes
        flat_dist = np.ones([self.n_cols]) / self.n_cols
        self.loading.generate_palettes(dist=flat_dist)

        # Generate Unloading Palettes
        counts = np.maximum(
            0,
            (
                np.bincount(self.grid.ravel(), minlength=self.palette_types + 1)
                - np.bincount(self.unloading.state, minlength=self.palette_types + 1)
            )[1:],
        )
        dist = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)
        self.unloading.generate_palettes(dist=dist)

        return self._step_return(reward, False, False)

    @staticmethod
    def _row_string(row):
        return " ".join([chr(x + 96) if has_palette(x) else "." for x in row])

    def render(self):
        if self.render_mode != "console":
            return

        print("Loading:   ", self._row_string(self.loading.state))
        for row in range(self.n_rows):
            print("Grid:      ", self._row_string(self.grid[row, :]))
        print("Unloading: ", self._row_string(self.unloading.state))
