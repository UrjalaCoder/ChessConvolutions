import chess
import numpy as np

class State():
    def __init__(self, board=None):
        self.board = chess.Board() if board is None else board
    
    # We need a bit representation for serialization
    def serialize(self):
        pieces = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6}
        state_array = np.zeros(64)
        
        en_passant = self.board.ep_square
        print(en_passant)
        
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece is not None:
                # If it is white, add 8
                piece_integer = pieces[piece.symbol().lower()]
                if piece.color == chess.WHITE:
                    piece_integer += 8
                state_array[i] = piece_integer
        
        # If white can castle on queenside, set the rook at pos 0 to 7
        if self.board.has_queenside_castling_rights(chess.WHITE):
                state_array[0] = 7

        # Similarly for kingside castling
        if self.board.has_kingside_castling_rights(chess.WHITE):
                state_array[7] = 7
        
        if self.board.has_queenside_castling_rights(chess.BLACK):
                state_array[] = 7
        
        if self.board.has_kingside_castling_rights(chess.WHITE):
                state_array[0] = 7
        
        print(state_array.reshape((8, 8)))
        

s = State()
s.serialize()
