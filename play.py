from state import State
from network import load_network
import torch
import chess

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

def get_best_possible(state, minimum=True):
    evaluations = []
    edges = state.edges()
    best_evaluation = None
    for i, m in enumerate(edges):
        new_state = State(board=state.board.copy())
        new_state.board.push(m)
        e = evaluate(new_state)
        evaluations.append(e)
        if best_evaluation is None:
            best_evaluation = (e, m)
        else:
            best_e, best_m = best_evaluation
            if minimum is True:
                if best_e > e:
                    best_evaluation = (e, m)
            else:
                if best_e < e:
                    best_evaluation = (e, m)
    print(evaluations)
    return best_evaluation



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

play()

