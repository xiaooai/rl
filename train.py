from __future__ import annotations
import argparse
import os

import numpy as np
from ttn_env import TicTacToeEnv
from q_agent import QAgent, canonical_key

# ---------------------- helpers ----------------------
def _flatten_board(obs) -> list[int]:
    # obs["board"] is (3,3) int8 ndarray
    return [int(x) for x in obs["board"].reshape(-1)]

def _to_play(obs) -> int:
    return int(obs["to_play"][0])

def _legal_from_info(info) -> list[int]:
    mask = info.get("legal_action_mask")
    if mask is None:
        # fallback: allow all empties
        return [i for i in range(9) if i < 9]
    return [i for i, m in enumerate(mask) if int(m) == 1]

# ---------------------- evaluation ----------------------
def evaluate(agent: QAgent, episodes: int = 500, opponent: str = "random") -> tuple[int,int,int]:
    """Play as X vs an environment-controlled O and report W/D/L.
    opponent in {"random", "perfect"}
    """
    win = draw = lose = 0
    for _ in range(episodes):
        env = TicTacToeEnv(opponent=opponent)
        obs, info = env.reset()
        done = False
        last_info = info
        while not done:
            legal = _legal_from_info(last_info)
            board = _flatten_board(obs)
            player = _to_play(obs)  # should be +1 at every agent turn
            action = agent.select_action(board, player, legal, greedy=True)
            obs, reward, terminated, truncated, last_info = env.step(action)
            done = terminated or truncated
        winner = int(last_info.get("winner", 0))
        if winner == 1:
            win += 1
        elif winner == -1:
            lose += 1
        else:
            draw += 1
    return win, draw, lose

# ---------------------- training ----------------------
def main():
    ap = argparse.ArgumentParser(description="Tabular Q-learning for Tic-Tac-Toe (Gymnasium env)")
    ap.add_argument("--episodes", type=int, default=500000)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--eval_every", type=int, default=10_000)
    ap.add_argument("--save", type=str, default="q_table.json")
    ap.add_argument("--load", type=str, default="q_table.json")
    ap.add_argument("--opponent", type=str, default="random", choices=["random", "perfect", "None"])
    args = ap.parse_args()

    agent = QAgent(alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)

    # Optional warm-start
    if args.load and os.path.exists(args.load):
        agent.load(args.load)
        print(f"Loaded Q table from {args.load}")

    # Training loop: agent (X) vs env-controlled O
    for ep in range(1, args.episodes + 1):
        # Linear decay with floors
        agent.epsilon = max(0.2, args.epsilon * (1 - ep / args.episodes))
        agent.alpha = max(0.1, args.alpha * (1 - ep / args.episodes))

        # 随机设置agent——player
        agent_player = np.random.choice([1, -1])
        env = TicTacToeEnv(agent_player=agent_player, opponent=args.opponent)
        obs, info = env.reset()
        done = False
        while not done:
            legal = _legal_from_info(info)
            board = _flatten_board(obs)
            player = _to_play(obs)

            s_key = canonical_key(board, player)
            action = agent.select_action(board, player, legal, greedy=False)

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            next_board = _flatten_board(next_obs)
            next_player = _to_play(next_obs)
            next_legal = _legal_from_info(next_info)

            agent.update(s_key, action, reward, next_board, next_player, next_legal, done)

            obs, info = next_obs, next_info

        if args.eval_every > 0 and ep % args.eval_every == 0:
            w, d, l = evaluate(agent, episodes=1_000, opponent="random")
            denom = max(1, (w + l))
            print(f"[{ep}/{args.episodes}] vs Random -> Win {w}, Draw {d}, Lose {l}  (WinRate={w/denom:.2f})")

    if args.save:
        agent.save(args.save)
        print(f"Saved Q table to {args.save}")


if __name__ == "__main__":
    main()
