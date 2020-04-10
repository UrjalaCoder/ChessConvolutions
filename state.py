import chess
import numpy as np

class State():
    def __init__(self, board=None):
        self.board = chess.Board() if board is None else board
    
    def serialize(self):
        pieces = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6}
        state_array = np.zeros(64, np.uint8)
        
        en_passant = self.board.ep_square
        # print(en_passant)
        
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
                state_array[0] = 7 + 8

        # Similarly for kingside castling
        if self.board.has_kingside_castling_rights(chess.WHITE):
                state_array[7] = 7 + 8
        
        if self.board.has_queenside_castling_rights(chess.BLACK):
                state_array[56] = 7
        
        if self.board.has_kingside_castling_rights(chess.WHITE):
                state_array[63] = 7
        # En passant    
        if self.board.ep_square is not None:
            state_array[self.board.ep_square] = 8

        # Make a bit vector
        bstate = state_array.reshape(8, 8)
        # print(bstate)

        bit_array = np.zeros((5, 8, 8), np.uint8)
        bit_array[0] = (bstate>>3)&1
        bit_array[1] = (bstate>>2)&1
        bit_array[2] = (bstate>>1)&1
        bit_array[3] = (bstate>>0)&1
        bit_array[4] = (self.board.turn*1.0)
        # print(bit_array)

        return bit_array
    
    def edges(self):
        return list(self.board.legal_moves)

# s = State()
# s.serialize()
# print(s.edges())
