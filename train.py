from __future__ import annotations
import argparse, random
from statistics import mean
from ttt_env import TicTacToe
from q_agent import QAgent, canonical_key
import os

def play_selfplay_episode(agent: QAgent):
    env = TicTacToe()
    board, player = env.reset()

    trajectory = []  # (s_key, a, r, next_board, next_player, done)
    done = False
    while not done:
        legal = env.legal_actions()
        s_key = canonical_key(board, player)
        a = agent.select_action(board, player, legal, greedy=False)
        next_board, r, done, _ = env.step(a)
        next_legal = env.legal_actions()
        next_player = env.current_player
        trajectory.append((s_key, a, r, next_board[:], next_player, done))
        board = next_board[:]
        player = next_player

    # 回放更新（也可以边走边更，这里边走边更）
    # 这里已经在循环内更新也完全可以；我们按“在线更新”实现：
    # 不过现在为了简洁，直接在走的时候更新。
    # （已在 select_action 之后 step 之后调用 agent.update）
    return env

def evaluate(agent: QAgent, episodes=500):
    # 与随机对手对战（各执先后）统计胜率
    win, draw, lose = 0, 0, 0
    for ep in range(episodes):
        env = TicTacToe()
        board, player = env.reset()
        # 随机决定谁先（智能体固定为“当前行动方视角下的玩家”，无需额外表）
        # 这里不需要显式指定，因为 agent 用的是当前行动方视角。
        done = False
        while not done:
            legal = env.legal_actions()
            if env.current_player == 1:
                # X 回合：用 agent
                a = agent.select_action(board, env.current_player, legal, greedy=True)
            else:
                # O 回合：随机
                a = random.choice(legal)
            next_board, r, done, info = env.step(a)
            board = next_board[:]

        winner = env.check_winner()
        if winner == 1:
            win += 1
        elif winner == -1:
            lose += 1
        else:
            draw += 1
    return win, draw, lose

def main():
    ap = argparse.ArgumentParser(description="Tabular Q-learning for Tic-Tac-Toe")
    ap.add_argument("--episodes", type=int, default=500000)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--eval_every", type=int, default=10000)
    ap.add_argument("--save", type=str, default="q_table.json")
    args = ap.parse_args()

    agent = QAgent(alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)
    if args.save and os.path.exists(args.save):
        agent.load(args.save)
    print(f"Loaded Q table from {args.save}")

    for ep in range(1, args.episodes+1):
        # 自我对弈一局，并在对弈过程中在线更新
        env = TicTacToe()
        board, player = env.reset()
        done = False
        while not done:
            legal = env.legal_actions()
            s_key = canonical_key(board, player)
            a = agent.select_action(board, player, legal, greedy=False)
            next_board, r, done, info = env.step(a)
            next_legal = env.legal_actions()
            next_player = env.current_player
            agent.update(s_key, a, r, next_board[:], next_player, next_legal, done)
            board = next_board[:]
            player = next_player

        if args.eval_every > 0 and ep % args.eval_every == 0:
            # 评估：与随机对手对战（智能体执先）
            w, d, l = evaluate(agent, episodes=200)
            print(f"[{ep}/{args.episodes}] vs Random -> Win {w}, Draw {d}, Lose {l}  (WinRate={w/(w+l+1e-9):.2f})")

    if args.save:
        agent.save(args.save)
        print(f"Saved Q table to {args.save}")

if __name__ == "__main__":
    main()
