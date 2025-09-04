from __future__ import annotations
"""
Human vs AI GUI for Tic-Tac-Toe (3x3) using the provided Gymnasium environment.

- No extra dependencies (Tkinter only).
- Supports human as X or O.
- AI policies: random / perfect (minimax) / optional QAgent (tabular) if q_table provided.
- Environment is used as the single source of truth for rules & terminal checks.

Run:
    python gui_tictactoe_human_vs_ai.py

Optional (use QAgent):
    python gui_tictactoe_human_vs_ai.py --qtable q_table.json

You may need to adjust the import path of `TicTacToeEnv` if the file/module name differs.
"""
import argparse
import os
import random
from functools import lru_cache
from typing import List, Optional, Tuple

from ttn_env import TicTacToeEnv


try:
    from q_agent import QAgent, canonical_key
except Exception:
    QAgent = None  # type: ignore
    canonical_key = None  # type: ignore

import tkinter as tk
from tkinter import ttk, messagebox

WIN_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diagonals
)

# ---------------- Minimax (perfect) policy ----------------
def winner_of(board: Tuple[int, ...]) -> Optional[int]:
    for a, b, c in WIN_LINES:
        s = board[a] + board[b] + board[c]
        if s == 3:
            return 1
        if s == -3:
            return -1
    return None

def is_draw(board: Tuple[int, ...]) -> bool:
    return all(v != 0 for v in board) and winner_of(board) is None

@lru_cache(maxsize=None)
def minimax_value(board: Tuple[int, ...], current: int, root: int) -> int:
    w = winner_of(board)
    if w is not None:
        return 1 if w == root else -1
    if is_draw(board):
        return 0
    # choose best for the player whose turn it is
    if current == root:
        best = -2
        for i, v in enumerate(board):
            if v == 0:
                b2 = list(board)
                b2[i] = current
                val = minimax_value(tuple(b2), -current, root)
                if val > best:
                    best = val
                if best == 1:
                    break
        return best
    else:
        best = 2
        for i, v in enumerate(board):
            if v == 0:
                b2 = list(board)
                b2[i] = current
                val = minimax_value(tuple(b2), -current, root)
                if val < best:
                    best = val
                if best == -1:
                    break
        return best

def minimax_move(board_list: List[int], player: int) -> int:
    bf = tuple(board_list)
    best_val = -2
    best_moves: List[int] = []
    for i, v in enumerate(board_list):
        if v == 0:
            b2 = list(board_list)
            b2[i] = player
            val = minimax_value(tuple(b2), -player, player)
            if val > best_val:
                best_val = val
                best_moves = [i]
            elif val == best_val:
                best_moves.append(i)
    return random.choice(best_moves)

# ---------------- GUI ----------------
class TicTacToeGUI:
    def __init__(self, root: tk.Tk, args):
        self.root = root
        self.args = args
        self.root.title("Tic-Tac-Toe — Human vs AI")

        # --- Top controls ---
        ctrl = ttk.Frame(root, padding=8)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(ctrl, text="You play as:").pack(side=tk.LEFT)
        self.side_var = tk.StringVar(value="X")
        ttk.Radiobutton(ctrl, text="X (first)", variable=self.side_var, value="X").pack(side=tk.LEFT)
        ttk.Radiobutton(ctrl, text="O (second)", variable=self.side_var, value="O").pack(side=tk.LEFT)

        ttk.Label(ctrl, text="  AI:").pack(side=tk.LEFT, padx=(12, 2))
        self.ai_var = tk.StringVar(value="perfect")
        ttk.Combobox(ctrl, textvariable=self.ai_var, values=["perfect", "random", "qagent"], width=8, state="readonly").pack(side=tk.LEFT)


        ttk.Button(ctrl, text="New Game", command=self.new_game).pack(side=tk.LEFT, padx=(12, 0))

        # --- Status bar ---
        self.status_var = tk.StringVar(value="Click 'New Game' to start")
        status = ttk.Label(root, textvariable=self.status_var, anchor=tk.W, padding=6)
        status.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Board Canvas ---
        self.size = 420
        self.margin = 20
        self.cell = (self.size - 2*self.margin) // 3
        self.canvas = tk.Canvas(root, width=self.size, height=self.size, bg="white", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_click)

        # internal state
        self.env: Optional[TicTacToeEnv] = None
        self.obs = None
        self.info = None
        self.human_player: int = 1  # +1=X, -1=O
        self.ai_player: int = -1
        self.qagent = None

        self.draw_board_grid()

    # ---- Drawing ----
    def draw_board_grid(self):
        self.canvas.delete("all")
        # grid
        for i in range(4):
            x = self.margin + i * self.cell
            self.canvas.create_line(x, self.margin, x, self.margin + 3*self.cell, width=2)
            y = self.margin + i * self.cell
            self.canvas.create_line(self.margin, y, self.margin + 3*self.cell, y, width=2)
        # pieces if game already started
        if self.obs is not None:
            flat = [int(x) for x in self.obs["board"].reshape(-1)]
            for idx, v in enumerate(flat):
                if v != 0:
                    self._draw_piece(idx, v)

    def _cell_rect(self, idx: int):
        r, c = divmod(idx, 3)
        x0 = self.margin + c * self.cell
        y0 = self.margin + r * self.cell
        x1 = x0 + self.cell
        y1 = y0 + self.cell
        return x0, y0, x1, y1

    def _draw_piece(self, idx: int, player: int):
        x0, y0, x1, y1 = self._cell_rect(idx)
        pad = self.cell * 0.18
        if player == 1:  # X
            self.canvas.create_line(x0+pad, y0+pad, x1-pad, y1-pad, width=5)
            self.canvas.create_line(x0+pad, y1-pad, x1-pad, y0+pad, width=5)
        else:  # O
            self.canvas.create_oval(x0+pad, y0+pad, x1-pad, y1-pad, width=5)

    # ---- Game flow ----
    def new_game(self):
        # Load QAgent if needed 添加默认选择
        self.qagent = None
        if self.ai_var.get() == "qagent":
            if QAgent is None:
                messagebox.showwarning("QAgent not found", "q_agent module is not available; falling back to perfect AI.")
                self.ai_var.set("perfect")
            else:
                default_qtable_path = "q_table.json"
                
                if not os.path.exists(default_qtable_path):
                    messagebox.showwarning("Q-table missing", "Q-table file not found; falling back to perfect AI.")
                    self.ai_var.set("perfect")
                else:
                    self.qagent = QAgent(alpha=0.0, gamma=1.0, epsilon=0.0)
                    self.qagent.load(default_qtable_path)

        # Decide roles
        self.human_player = 1 if self.side_var.get() == "X" else -1
        self.ai_player = -self.human_player

        # fresh env (we control both sides => opponent=None)
        self.env = TicTacToeEnv(agent_player=+1, opponent=None)
        self.obs, self.info = self.env.reset()
        self.status_var.set(f"You: {'X' if self.human_player==1 else 'O'} | AI: {'O' if self.human_player==1 else 'X'} — Your turn" if self.human_player==1 else "AI thinking...")

        self.draw_board_grid()

        # If human chose O, AI (X) moves first
        if self.human_player == -1:
            self.root.after(150, self.ai_move)

    def on_click(self, event):
        if not self.env or self.obs is None:
            return
        if self.is_terminal():
            return
        # Only accept input if it's human's turn
        to_play = int(self.obs["to_play"][0])
        if to_play != self.human_player:
            return
        col = int((event.x - self.margin) // self.cell)
        row = int((event.y - self.margin) // self.cell)
        if not (0 <= row < 3 and 0 <= col < 3):
            return
        idx = row*3 + col
        # check legality via mask
        mask = self.info.get("legal_action_mask", None)
        if mask is not None and int(mask[idx]) == 0:
            return
        self.step_env(idx)
        if not self.is_terminal():
            self.root.after(80, self.ai_move)

    def step_env(self, action: int):
        if not self.env:
            return
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        self.draw_board_grid()
        if terminated or truncated:
            self.on_game_over(reward)
        else:
            to_play = int(self.obs["to_play"][0])
            self.status_var.set("Your turn" if to_play == self.human_player else "AI thinking...")

    def ai_move(self):
        if not self.env or self.is_terminal():
            return
        to_play = int(self.obs["to_play"][0])
        if to_play != self.ai_player:
            return
        legal = [i for i, m in enumerate(self.info.get("legal_action_mask", [1]*9)) if int(m) == 1]
        board = [int(x) for x in self.obs["board"].reshape(-1)]

        policy = self.ai_var.get()
        if policy == "random":
            action = random.choice(legal)
        elif policy == "qagent" and self.qagent is not None and canonical_key is not None:
            s_key = canonical_key(board, self.ai_player)
            action = self.qagent.select_action(board, self.ai_player, legal, greedy=True)
        else:
            action = minimax_move(board, self.ai_player)

        self.step_env(action)

    def is_terminal(self) -> bool:
        if not self.env or self.obs is None:
            return True
        # Gymnasium 5-tuple uses info plus our own state; we track terminal via step
        # Quick heuristic: if no legal actions left -> terminal
        mask = self.info.get("legal_action_mask", None)
        if mask is None:
            return False
        return sum(int(m) for m in mask) == 0

    def on_game_over(self, reward: float):
        # Environment rewards are from the perspective of whoever is set as agent in env (we used agent_player=+1),
        # but since we controlled both sides, we need to infer winner from board.
        # Simpler: use message based on symbols count.
        # Better: consult env-rendered text would be okay, but we don't expose the env internals here.
        # We'll compute winner directly.
        flat = [int(x) for x in self.obs["board"].reshape(-1)]
        w = winner_of(tuple(flat))
        if w is None:
            self.status_var.set("Game over: Draw")
            messagebox.showinfo("Result", "Draw")
        else:
            you_win = (w == self.human_player)
            self.status_var.set("Game over: You Win!" if you_win else "Game over: You Lose")
            messagebox.showinfo("Result", "You Win!" if you_win else "You Lose")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qtable", type=str, default="", help="Path to a saved Q table (optional)")
    args = parser.parse_args()

    root = tk.Tk()
    # macOS: use Aqua style fonts a bit larger
    try:
        root.call('tk', 'scaling', 1.25)
    except Exception:
        pass

    app = TicTacToeGUI(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
