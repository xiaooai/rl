from __future__ import annotations
import json, random
from typing import Dict, List, Tuple

# 表格型 Q-learning 智能体（共享一张表）
# 关键：把状态标准化到“当前行动方视角”（我方=1，对方=-1）
# 编码：state_key = ''.join(map(str, [me, opp, empty])) with values in {-1,0,1} after perspective transform

def canonical_key(board: List[int], current_player: int) -> str:
    # 视角标准化：让当前行动方总是“我方=+1”。
    # 做法：把棋盘乘以 current_player：我的棋子=1，对手=-1。
    return ''.join(str(v * current_player) for v in board)

class QAgent:
    def __init__(self, alpha=0.5, gamma=0.99, epsilon=0.2):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q: Dict[str, List[float]] = {}  # key -> q[9]

    def _ensure_state(self, key: str):
        if key not in self.Q:
            self.Q[key] = [0.0]*9  # 初始 0

    def select_action(self, board: List[int], current_player: int, legal_actions: List[int], greedy: bool=False) -> int:
        key = canonical_key(board, current_player)
        self._ensure_state(key)

        # ε-贪心（或纯贪心）
        if (not greedy) and random.random() < self.epsilon:
            return random.choice(legal_actions)

        q = self.Q[key]
        # 从合法动作中选 Q 最大者；平手随机
        best_a, best_q = None, -1e9
        for a in legal_actions:
            if q[a] > best_q:
                best_q = q[a]
                best_a = a
        return best_a if best_a is not None else random.choice(legal_actions)

    def update(self, s_key: str, a: int, reward: float, next_board: List[int], next_player: int, next_legal: List[int], done: bool):
        self._ensure_state(s_key)
        q_sa = self.Q[s_key][a]

        if done:
            target = reward
        else:
            next_key = canonical_key(next_board, next_player)
            self._ensure_state(next_key)
            # max over next legal
            next_max = max(self.Q[next_key][na] for na in next_legal) if next_legal else 0.0
            target = reward + self.gamma * next_max

        self.Q[s_key][a] = q_sa + self.alpha * (target - q_sa)

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.Q, f)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            self.Q = json.load(f)
