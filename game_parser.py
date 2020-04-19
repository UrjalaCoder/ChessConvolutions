import sys
import numpy as np


TEST_FILENAME = "ficsgamesdb_2019_standard2000_nomovetimes_123895.pgn"

def load_game_file(path=TEST_FILENAME):
    return open(f"raw_data/{path}", encoding="utf-8-sig")

def generate_training_dataset(games):
    boards = []
    for game in games:
        for board in game:
            boards.append(board)
    boards = np.array(boards)
    return boards

def get_game_boards(game_count=10000):
    import chess.pgn
    from state import State
    def get_game_serialization(moves, result):
        s = State()
        serializations = []
        for move in moves:
            s.board.push(move)
            ser = s.serialize()
            serializations.append([ser, result])
        return serializations

    pgn_data = load_game_file()
    games = []
    counter = 0
    while 1 and counter < game_count:
        print(f"counter: {counter} / {game_count}")
        game = None
        try:
            game = chess.pgn.read_game(pgn_data)
        except Exception:
            print("at the end!")
            break
        moves = list(game.mainline_moves())
        #print(moves)
        result_str = game.headers['Result']
        # print(result_str)
        result = None
        # Parse result
        white, black = result_str.split('-')
        #print()
        if white == '1' and black == '0':
            result = 1
        elif white == '0' and black == '1':
            # print("WHITE LOST!")
            result = -1
        elif white == '1/2' and black == '1/2':
            result = 0
        
        if result is not None:
            serializations = np.array(get_game_serialization(moves, result))
            games.append(serializations)
        counter += 1
    return np.array(games)

def save_games(games, filename="data"):
    try:
        np.save(f"datasets/{filename}.npy", games)
        return True
    except Exception:
        error = sys.exc_info()[0]
        print(f"Failed to save!, error: {error}.")
        return False

def load(filename="data"):
    return np.load(f"datasets/{filename}.npy", allow_pickle=True)

def parse_and_save():
    games = get_game_boards()
    boards = generate_training_dataset(games)
    print(len(boards))
    save_games(boards)

# parse_and_save()

