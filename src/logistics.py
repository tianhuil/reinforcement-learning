import gymnasium as gym
import numpy as np


class Logistics(gym.Env):
    """
    A warehouse logistics problem
    """

    SMALL = 1e0
    MEDIUM = 1e3
    LARGE = 1e6

    def __init__(
        self,
        n_rows: int = 4,
        n_cols: int = 4,
        palette_types: int = 4,
        render_mode="console",
    ):
        super(Logistics, self).__init__()
        # parameters
        self.render_mode = render_mode
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.palette_types = palette_types
        self.loading_row = n_rows - 1
        self.unloading_row = 0

        # X, Y coordinates can move in one of 4 cardinal directions
        self.action_space = gym.spaces.MultiDiscrete([self.n_rows, self.n_cols, 4])

        # The grid and loading and unloading can each have a palette or be empty
        self.observation_space = gym.spaces.Dict(
            {
                "grid": gym.spaces.MultiDiscrete(
                    (self.palette_types + 1) * np.ones([n_rows, n_cols], dtype=np.int_)
                ),
                "loading": gym.spaces.MultiDiscrete(
                    (self.palette_types + 1) * np.ones(n_cols, dtype=np.int_)
                ),
                "unloading": gym.spaces.MultiDiscrete(
                    (self.palette_types + 1) * np.ones(n_cols, dtype=np.int_)
                ),
            }
        )

        self._reset()

    def _reset(self):
        self.grid = np.zeros((self.n_rows, self.n_cols), dtype=np.int_)
        self.loading = np.zeros((self.n_cols,), dtype=np.int_)
        self.unloading = np.zeros((self.n_cols,), dtype=np.int_)

    def _observation(self):
        return {
            "grid": self.grid,
            "loading": self.loading,
            "unloading": self.unloading,
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

    @staticmethod
    def _has_palette(arr):
        return arr > 0

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
        for k in range(self.n_cols):
            origin, destination, success = self.transfer(
                self.loading[k], self.grid[self.loading_row, k]
            )
            self.grid[self.loading_row, k] = destination
            self.loading[k] = origin
            if not success:
                reward -= self.LARGE

        # Unload palettes if we can, penalty if not
        for k in range(self.n_cols):
            origin, destination, success = self.unload(
                self.grid[self.unloading_row, k], self.unloading[k]
            )
            self.grid[self.unloading_row, k] = origin
            self.unloading[k] = destination
            if not success:
                reward -= self.LARGE

        # Move palette based on action
        orig_x, orig_y, direction = action
        dest_x, dest_y = self._destination(orig_x, orig_y, direction)

        if not self.valid_coords(orig_x, orig_y):
            reward -= self.SMALL
            return self._step_return(reward, False, False)

        origin, destination, success = self.transfer(
            self.grid[orig_x, orig_y], self.grid[dest_x, dest_y]
        )

        self.grid[orig_x, orig_y] = origin
        self.grid[dest_x, dest_y] = destination

        if not success:
            reward -= self.LARGE

        # Add palettes to loading and unloading

    @staticmethod
    def transfer(origin: int, destination: int) -> tuple[int, int, bool]:
        """
        Transfer a palette from origin to destination, if possible, retuning new origin,
        destination, and whether the transfer was successful
        """

        if origin == 0:
            return origin, destination, False
        if destination != 0:
            return origin, destination, False
        return 0, origin, True

    @staticmethod
    def unload(origin: int, destination: int) -> tuple[int, int, bool]:
        """
        Transfer a palette to unloading, where destination needs to match origin
        """

        if origin == 0:
            return origin, destination, False
        if destination != origin:
            return origin, destination, False
        return 0, 0, True
