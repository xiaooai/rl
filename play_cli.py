from __future__ import annotations
import argparse, random
from ttt_env import TicTacToe
from q_agent import QAgent

def parse_args():
    ap = argparse.ArgumentParser(description="Play Tic-Tac-Toe vs Q-learning agent")
    ap.add_argument("--load", type=str, default=None, help="path to q_table.json")
    ap.add_argument("--human", type=str, default="X", choices=["X","O"], help="you play as X or O")
    return ap.parse_args()

def print_board(env: TicTacToe):
    print(env.render())

def human_move(env: TicTacToe):
    legal = env.legal_actions()
    while True:
        try:
            s = input("输入落子(0-8)，或 'q' 退出：").strip()
            if s.lower() in ("q","quit","exit"):
                return None
            a = int(s)
            if a in legal:
                return a
            print(f"非法动作。可选位置：{legal}")
        except Exception:
            print("请输入数字 0..8，对应九宫格位置。")

def main():
    args = parse_args()
    agent = QAgent(alpha=0.0, gamma=0.99, epsilon=0.0)  # 对战时用贪心
    if args.load:
        agent.load(args.load)
        print(f"Loaded Q table from {args.load}")

    human_is_x = (args.human.upper() == "X")

    while True:
        env = TicTacToe()
        board, player = env.reset()
        print("新对局开始！你是", "X(先手)" if human_is_x else "O(后手)")
        print_board(env)

        done = False
        while not done:
            legal = env.legal_actions()
            if (env.current_player == 1 and human_is_x) or (env.current_player == -1 and not human_is_x):
                a = human_move(env)
                if a is None:
                    print("退出游戏。")
                    return
            else:
                a = agent.select_action(env.board, env.current_player, legal, greedy=True)

            board, r, done, info = env.step(a)
            print_board(env)

        winner = env.check_winner()
        if winner == 1:
            print("X 胜！")
        elif winner == -1:
            print("O 胜！")
        else:
            print("平局。")

        again = input("再来一局？(y/n) ").strip().lower()
        if again != 'y':
            break

if __name__ == "__main__":
    main()
