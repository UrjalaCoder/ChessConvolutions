# ChessConvolutions

## A chess AI

Utilizes a convolutional neural network

#### Board serialization

8x8 square:

* Blanks:
** Empty
* Pieces:
** Pawn
** Knight
** Rook
** Rook (castle from queenside)
** Rook (castle from kingside)
** Bishop
** Queen
** King

8 pieces * 2 sides => 16 so 4 bits to represent ONE position

Also en passant square: 1 bit

In total
8 * 8 * 4 = 256
256 + 1 = 257 bits


