from __future__ import annotations
import argparse
import random
from ttt_env import TicTacToe
from q_agent import QAgent

def parse_args():
    ap = argparse.ArgumentParser(description="Agent vs Agent Tic-Tac-Toe")
    ap.add_argument("--load1", type=str, default="q_table.json", help="Agent1 Q-table path")
    ap.add_argument("--load2", type=str, default="q_table.json", help="Agent2 Q-table path")
    ap.add_argument("--games", type=int, default=100, help="Number of games")
    ap.add_argument("--show", action="store_true", help="Show board each step")
    return ap.parse_args()

def print_board(env: TicTacToe):
    print(env.render())

def main():
    args = parse_args()
    agent1 = QAgent(alpha=0.0, gamma=0.99, epsilon=0.1)
    agent2 = QAgent(alpha=0.0, gamma=0.99, epsilon=0.1)
    if args.load1:
        agent1.load(args.load1)
        print(f"Agent1 loaded Q table from {args.load1}")
    if args.load2:
        agent2.load(args.load2)
        print(f"Agent2 loaded Q table from {args.load2}")

    win1, win2, draw = 0, 0, 0
    for g in range(args.games):
        env = TicTacToe()
        # 随机交换先后手
        if random.random() < 0.5:
            agent_first, agent_second = agent1, agent2
            first_label = 1
        else:
            agent_first, agent_second = agent2, agent1
            first_label = -1
        env.current_player = first_label
        board = env.board[:]
        done = False

        if args.show:
            print(f"\n第{g+1}局开始")
            print_board(env)
        while not done:
            legal = env.legal_actions()
            if env.current_player == 1:
                a = agent1.select_action(env.board, env.current_player, legal, greedy=True)
            else:
                a = agent2.select_action(env.board, env.current_player, legal, greedy=True)
            board, r, done, info = env.step(a)
            if args.show:
                print_board(env)
        winner = env.check_winner()
        if winner == 1:
            win1 += 1
        elif winner == -1:
            win2 += 1
        else:
            draw += 1
    print(f"\nAgent1胜: {win1}, Agent2胜: {win2}, 平局: {draw}，总局数: {args.games}")

if __name__ == "__main__":
    main()