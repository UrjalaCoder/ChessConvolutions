import sys
import numpy as np
import argparse

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

def get_game_boards(board_count, date_print_iteration=10):
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
    result_dict = {
        "1-0": 1,
        "0-1": -1,
        "1/2-1/2": 0
    }
    date_counter = 0
    while 1 and counter < board_count:
        print(f"counter: {counter} / {board_count}")
        game = None
        try:
            game = chess.pgn.read_game(pgn_data)
        except Exception:
            print("at the end!")
            break
        moves = list(game.mainline_moves())
        result_str = game.headers['Result'].strip()
        if date_counter % date_print_iteration == 0:
            date_str = game.headers['Date'].strip()
            print(date_str)
            date_counter = 0
        result = None
        # Parse result
        result = result_dict[result_str]

        if result is not None:
            serializations = np.array(get_game_serialization(moves, result))
            counter += len(serializations)
            games.append(serializations)
        date_counter += 1
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

def main():
    parser = argparse.ArgumentParser("Chess game data parser")
    parser.add_argument("filename", type=str, help="Filename to store the dataset")
    parser.add_argument("board_count", type=int, help="Amount of boards to store")
    
    args = parser.parse_args()
    
    filename = args.filename
    board_count = args.board_count

    games = get_game_boards(board_count)
    boards = generate_training_dataset(games)
    save_games(boards, filename=filename)
    
if __name__ == "__main__":
    main()

