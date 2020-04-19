from state import State
from network import load_network
import torch
import chess
import numpy as np

net = load_network()

def evaluate(state):
    serialized = state.serialize()
    input_tensor = torch.tensor(serialized).unsqueeze(dim=0).to(dtype=torch.float32)
    output = net(input_tensor)[0].item()
    return output

def parse_input(raw_string, board):
    stripped = raw_string.strip()
    if stripped == "QUIT":
        return None
    move = board.parse_san(raw_string.strip())
    return move

def get_best_possible(state, minimum=True, rec=True, undeterministic=True):
    evaluations = []
    edges = state.edges()
    best_evaluation = None
    best_move = None
    resultmap = {
        '1-0': 1,
        '0-1': -1,
        '1/2-1/2': 0
    }
    # print(state.board)
    if edges is None and state.board.is_game_over() is True:
        return resultmap[state.board.result]
    # print("OPPONENT MOVES:")
    moves = []
    for i, m in enumerate(edges):
        new_state = State(board=state.board.copy())
        new_state.board.push(m)
        # print(new_state.board)
        evaluation = None
        if rec is False:
            evaluation = evaluate(new_state)
            #evaluations.append(evaluation)
        else:
            evaluation = get_best_possible(new_state, minimum=(not minimum), rec=False)
        evaluations.append(evaluation)
        moves.append(m)
        if best_evaluation is None:
            best_evaluation = evaluation
            best_move = m
        elif minimum is True and best_evaluation is not None:
            if best_evaluation > evaluation:
                best_evaluation = evaluation
                best_move = m
        elif minimum is False and best_evaluation is not None:
            if best_evaluation < evaluation:
                best_evaluation = evaluation
                best_move = m

    # print(evaluations, rec)
    
    # Get some move, not necessarily the best possible
    if undeterministic is True:
        possibilities = list(map(lambda x: abs(x), evaluations))
        real_move = None
        # print(possibilities)
        while real_move is None:
            random = np.random.uniform()
            for i, e in enumerate(possibilities):
                if e >= random:
                    real_move = moves[i]
                    best_move = real_move
                    best_evaluation = evaluations[i]
                    break

    return (best_evaluation, best_move)

def self_play():
    current_state = State()
    while current_state.board.is_game_over() is False:
        eval_white, move_white = get_best_possible(current_state, minimum=False, rec=False)
        current_state.board.push(move_white)
        print(move_white)
        if current_state.board.is_game_over() is False:
            eval_black, move_black = get_best_possible(current_state, minimum=True, rec=False)
            current_state.board.push(move_black)
            print(move_black)
        #print(current_state.board)
    print(current_state.board.result())

def play():
    current_state = State()
    r = evaluate(current_state)
    print(r)
    playing = True
    while playing:
        raw = input("Please make a move...\n")
        m = parse_input(raw, current_state.board)
        if m is None:
            playing = False
        else:
            current_state.board.push(m)
            best_evaluation, best_move = get_best_possible(current_state)
            print(best_move, best_evaluation)
            current_state.board.push(best_move)

self_play()

