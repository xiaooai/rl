from __future__ import annotations
from typing import List, Optional, Tuple

# 井字棋：3x3，X=+1 先手，O=-1 后手
# 状态：board[9] in {-1,0,1}；current_player in {+1,-1}

WIN_LINES = [
    (0,1,2),(3,4,5),(6,7,8),  # rows
    (0,3,6),(1,4,7),(2,5,8),  # cols
    (0,4,8),(2,4,6)           # diagonals
]

class TicTacToe:
    def __init__(self):
        self.board = [0]*9
        self.current_player = 1  # X starts
        self.last_move = None

    def reset(self):
        self.board = [0]*9
        self.current_player = 1
        self.last_move = None
        return self.board, self.current_player

    def legal_actions(self) -> List[int]:
        return [i for i, v in enumerate(self.board) if v == 0]

    def step(self, action: int):
        # Returns: next_state, reward, done, info
        if action not in self.legal_actions():
            # 非法落子：直接判负
            # 奖励从“执行者视角”给，因此非法即 -1
            # 终止，下一状态为当前（不重要）
            return (self.board, -1.0, True, {"illegal": True})

        self.board[action] = self.current_player
        self.last_move = action

        winner = self.check_winner()
        if winner is not None:
            # 胜者 winner: +1 or -1； 奖励从“执行者视角”
            reward = 1.0 if winner == self.current_player else -1.0
            return (self.board, reward, True, {})

        if all(v != 0 for v in self.board):
            # 平局
            return (self.board, 0.0, True, {})

        # 轮到对手
        self.current_player *= -1
        return (self.board, 0.0, False, {})

    def check_winner(self) -> Optional[int]:
        b = self.board
        for a,b1,c in WIN_LINES:
            s = b[a] + b[b1] + b[c]
            if s == 3:
                return 1
            if s == -3:
                return -1
        return None

    def render(self) -> str:
        def sym(v):
            return 'X' if v == 1 else ('O' if v == -1 else '.')
        rows = []
        for r in range(3):
            rows.append(' '.join(sym(self.board[3*r + c]) for c in range(3)))
        turn = 'X' if self.current_player == 1 else 'O'
        return "\n".join(rows) + f"\nTurn: {turn}"
