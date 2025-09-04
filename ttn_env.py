"""
Tic-Tac-Toe (3x3) — Gymnasium-compliant single-agent environment

- Agent can play **X (+1)** or **O (-1)**; set via `agent_player`.
- The environment can optionally play as the opponent using a random or perfect policy.
- Observation: Dict with
    - "board": (3, 3) int8 array in {-1, 0, +1}
    - "to_play": (1,) int8 array in {-1, +1}
- Action space: Discrete(9), indexing cells row-major [0..8].
- Rewards are **from the agent's perspective**, regardless of role:
    - `win_reward` if agent wins, `loss_reward` if opponent wins, `draw_reward` for draw/ongoing.
    - If an ILLEGAL action is taken and `illegal_move_ends=True`, episode ends with reward `loss_reward`.
- `info` always contains a `legal_action_mask` of shape (9,), dtype=int8, and `agent_player`.

Requires:
    pip install gymnasium numpy

Example usage (agent plays O):

```python
import gymnasium as gym
import numpy as np
from tictactoe_gymnasium_env import TicTacToeEnv

env = TicTacToeEnv(agent_player=-1, opponent="perfect", render_mode="ansi")
obs, info = env.reset(seed=0)
print(env.render())

terminated = truncated = False
while not (terminated or truncated):
    mask = info["legal_action_mask"]
    action = int(np.random.choice(np.flatnonzero(mask)))
    obs, reward, terminated, truncated, info = env.step(action)
    print(env.render())

print("Reward:", reward)
```
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

WIN_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diagonals
)


class TicTacToeEnv(gym.Env):
    """A standard & robust Tic-Tac-Toe environment for Gymnasium with agent role selection.

    Parameters
    ----------
    agent_player : {+1, -1}, default +1
        Which side the agent plays: +1 for X (first), -1 for O (second).
    opponent : {None, "random", "perfect"}, default "random"
        - None: no auto-opponent (self-play / two-player via one agent). The caller must act for both sides.
        - "random": environment plays the **non-agent** side uniformly at random among legal actions.
        - "perfect": environment plays the **non-agent** side optimally (minimax/negamax).
    illegal_move_ends : bool, default True
        If True, any illegal action immediately terminates the episode with reward `loss_reward` to the agent.
        If False, the illegal action is ignored and `info["illegal"] = True` is set; the turn does not change.
    win_reward : float, default 1.0
    draw_reward : float, default 0.0
    loss_reward : float, default -1.0
    render_mode : {None, "ansi"}
        If "ansi", `render()` returns a human-readable string board.
    """

    metadata = {"render_modes": ["ansi"], "name": "TicTacToe-v2"}

    def __init__(
        self,
        *,
        agent_player: int = 1,
        opponent: Optional[str] = "random",
        illegal_move_ends: bool = True,
        win_reward: float = 1.0,
        draw_reward: float = 0.0,
        loss_reward: float = -1.0,
        render_mode: Optional[str] = None,
    ) -> None:
        assert agent_player in (-1, 1), "agent_player must be +1 (X) or -1 (O)"
        assert opponent in (None, "random", "perfect"), "opponent must be None, 'random', or 'perfect'"
        assert render_mode in (None, "ansi")
        self.render_mode = render_mode

        self.agent_player = int(agent_player)
        self.opponent_mode = opponent
        self.illegal_move_ends = illegal_move_ends
        self.win_reward = float(win_reward)
        self.draw_reward = float(draw_reward)
        self.loss_reward = float(loss_reward)

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=-1, high=1, shape=(3, 3), dtype=np.int8),
                "to_play": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.int8),
            }
        )

        # internal state
        self._board: np.ndarray = np.zeros(9, dtype=np.int8)
        self._to_play: int = 1  # +1 starts (X always starts in Tic-Tac-Toe)
        self._last_move: Optional[int] = None
        self._terminated: bool = False
        self._truncated: bool = False

        # perfect policy cache
        self._tt_cache: Dict[Tuple[Tuple[int, ...], int], int] = {}

        # seeding / RNG (set in reset)
        self.np_random = np.random.default_rng()

    # --------------- Gymnasium core API ---------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self._board[:] = 0
        self._to_play = 1  # X starts
        self._last_move = None
        self._terminated = False
        self._truncated = False
        self._tt_cache.clear()

        # If agent plays O and an auto-opponent is enabled, let the environment (X) move once
        if self.opponent_mode is not None and self.agent_player != self._to_play:
            self._env_opponent_move_once()  # X moves once -> O to play

        obs = self._obs()
        info = {
            "legal_action_mask": self._legal_action_mask(),
            "last_move": self._last_move,
            "agent_player": self.agent_player,
        }
        return obs, info

    def step(self, action: int):
        if self._terminated or self._truncated:
            raise RuntimeError("Cannot call step() on a terminated episode. Call reset().")

        illegal = (action < 0 or action >= 9 or self._board[action] != 0)
        if illegal:
            if self.illegal_move_ends:
                self._terminated = True
                obs = self._obs()
                info = {
                    "illegal": True,
                    "legal_action_mask": self._legal_action_mask(),
                    "last_move": self._last_move,
                    "agent_player": self.agent_player,
                }
                return obs, float(self.loss_reward), True, False, info
            else:
                info = {
                    "illegal": True,
                    "legal_action_mask": self._legal_action_mask(),
                    "last_move": self._last_move,
                    "agent_player": self.agent_player,
                }
                return self._obs(), 0.0, False, False, info

        # Agent (or current self-play player) makes a move
        self._place(action, self._to_play)

        # Check terminal after agent move
        winner = self._check_winner()
        if winner is not None:
            self._terminated = True
            reward = self.win_reward if winner == self.agent_player else self.loss_reward
            info = {
                "winner": int(winner),
                "terminal_reason": "win" if winner == self.agent_player else "loss",
                "legal_action_mask": self._legal_action_mask(),
                "last_move": self._last_move,
                "agent_player": self.agent_player,
            }
            return self._obs(), float(reward), True, False, info

        if self._is_draw():
            self._terminated = True
            info = {
                "winner": 0,
                "terminal_reason": "draw",
                "legal_action_mask": self._legal_action_mask(),
                "last_move": self._last_move,
                "agent_player": self.agent_player,
            }
            return self._obs(), float(self.draw_reward), True, False, info

        # If it's now the opponent's turn and auto-opponent is enabled, let env play once
        if self.opponent_mode is not None and self._to_play != self.agent_player:
            self._env_opponent_move_once()

            # Re-check terminal after opponent move
            winner = self._check_winner()
            if winner is not None:
                self._terminated = True
                reward = self.win_reward if winner == self.agent_player else self.loss_reward
                info = {
                    "winner": int(winner),
                    "terminal_reason": "win" if winner == self.agent_player else "loss",
                    "legal_action_mask": self._legal_action_mask(),
                    "last_move": self._last_move,
                    "agent_player": self.agent_player,
                }
                return self._obs(), float(reward), True, False, info

            if self._is_draw():
                self._terminated = True
                info = {
                    "winner": 0,
                    "terminal_reason": "draw",
                    "legal_action_mask": self._legal_action_mask(),
                    "last_move": self._last_move,
                    "agent_player": self.agent_player,
                }
                return self._obs(), float(self.draw_reward), True, False, info

            # Otherwise, it should now be the agent's turn again

        info = {
            "legal_action_mask": self._legal_action_mask(),
            "last_move": self._last_move,
            "agent_player": self.agent_player,
        }
        return self._obs(), 0.0, False, False, info

    def render(self) -> Optional[str]:
        if self.render_mode != "ansi":
            return None
        return self._board_string()

    def close(self) -> None:  # no resources to free
        pass

    # --------------- Helpers ---------------
    def _obs(self) -> Dict[str, np.ndarray]:
        return {
            "board": self._board.reshape(3, 3).copy(),
            "to_play": np.array([self._to_play], dtype=np.int8),
        }

    def _legal_action_mask(self) -> np.ndarray:
        mask = (self._board == 0).astype(np.int8)
        return mask

    def _place(self, idx: int, player: int) -> None:
        assert self._board[idx] == 0, "attempted to place on occupied cell"
        self._board[idx] = np.int8(player)
        self._last_move = int(idx)
        self._to_play = -player

    def _check_winner(self) -> Optional[int]:
        b = self._board
        for a, c, d in WIN_LINES:
            s = int(b[a] + b[c] + b[d])
            if s == 3:
                return 1
            if s == -3:
                return -1
        return None

    def _is_draw(self) -> bool:
        return np.all(self._board != 0) and self._check_winner() is None

    def _board_string(self) -> str:
        def sym(v: int) -> str:
            return "X" if v == 1 else ("O" if v == -1 else ".")

        rows = []
        for r in range(3):
            rows.append(" ".join(sym(int(self._board[3 * r + c])) for c in range(3)))
        if self._terminated:
            w = self._check_winner()
            tail = (
                f"Game over. Winner: {'X' if w == 1 else 'O'}" if w is not None else "Game over. Draw."
            )
        else:
            tail = f"Turn: {'X' if self._to_play == 1 else 'O'}"
        return "\n".join(rows) + "\n" + tail


    def _env_opponent_move_once(self) -> None:
        """Let the environment play one move for the non-agent side.
        Assumes game is not terminal and it's currently the opponent's turn.
        """
        assert self.opponent_mode is not None
        assert self._to_play != self.agent_player
        opp_action = self._opponent_action(player=self._to_play)
        self._place(opp_action, self._to_play)
        # After this, it's either terminal or the turn switches back.

    # --------------- Opponent policies ---------------
    def _opponent_action(self, *, player: int) -> int:
        legal = np.flatnonzero(self._board == 0)
        if self.opponent_mode == "random":
            return int(self.np_random.choice(legal))
        elif self.opponent_mode == "perfect":
            return self._minimax_best_action(root_player=player)
        else:
            raise RuntimeError("Opponent mode is not set but _opponent_action was called.")

    def _minimax_best_action(self, root_player: int) -> int:
        # choose the action that maximizes the outcome for root_player
        best_val = -2
        best_actions: List[int] = []
        for a in np.flatnonzero(self._board == 0):
            self._board[a] = np.int8(root_player)
            val = self._minimax_value(current_player=-root_player, root_player=root_player)
            self._board[a] = np.int8(0)
            if val > best_val:
                best_val = val
                best_actions = [int(a)]
            elif val == best_val:
                best_actions.append(int(a))
        # tie-break randomly for variety
        return int(self.np_random.choice(best_actions))

    def _minimax_value(self, current_player: int, root_player: int) -> int:
        # memoization key
        key = (tuple(int(x) for x in self._board), int(current_player)*10 + int(root_player))
        if key in self._tt_cache:
            return self._tt_cache[key]

        winner = self._check_winner()
        if winner is not None:
            out = 1 if winner == root_player else -1
            self._tt_cache[key] = out
            return out
        if np.all(self._board != 0):
            self._tt_cache[key] = 0
            return 0

        if current_player == root_player:
            best = -2
            for a in np.flatnonzero(self._board == 0):
                self._board[a] = np.int8(current_player)
                v = self._minimax_value(-current_player, root_player)
                self._board[a] = np.int8(0)
                if v > best:
                    best = v
                if best == 1:
                    break
        else:
            best = 2
            for a in np.flatnonzero(self._board == 0):
                self._board[a] = np.int8(current_player)
                v = self._minimax_value(-current_player, root_player)
                self._board[a] = np.int8(0)
                if v < best:
                    best = v
                if best == -1:
                    break

        self._tt_cache[key] = int(best)
        return int(best)


# Optional: simple env checker when executed directly
if __name__ == "__main__":
    import numpy as _np

    # Example 1: Agent plays X (default)
    env = TicTacToeEnv(agent_player=+1, opponent="perfect", render_mode="ansi")
    obs, info = env.reset(seed=123)
    print(env.render())

    done = False
    while not done:
        mask = info["legal_action_mask"]
        action = int(_np.random.choice(_np.flatnonzero(mask)))
        obs, reward, terminated, truncated, info = env.step(action)
        print(env.render())
        done = terminated or truncated
    print("Final reward (X):", reward)

    # Example 2: Agent plays O
    env = TicTacToeEnv(agent_player=-1, opponent="perfect", render_mode="ansi")
    obs, info = env.reset(seed=123)
    print(env.render())

    done = False
    while not done:
        mask = info["legal_action_mask"]
        action = int(_np.random.choice(_np.flatnonzero(mask)))
        obs, reward, terminated, truncated, info = env.step(action)
        print(env.render())
        done = terminated or truncated
    print("Final reward (O):", reward)
