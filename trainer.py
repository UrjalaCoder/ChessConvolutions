from network import ChessNet
from game_parser import load
import torch
import numpy as np

games = load()
# print(games[0:100])
game = games[0]
board = game[-1]

net = ChessNet()

def get_random_sample(arr, count=10):
    choice_indices = [i for i in range(len(arr))]
    choices = []
    for i in range(count):
        index = np.random.choice(choice_indices)
        choices.append(arr[index])
    return choices

def train(net, games, epochs=5, bs_size=10, lr=0.05):
    for epoch in range(epochs):
        bs = get_random_sample(games, count=bs_size)
        current_loss = 0
        for game in bs:
            game_loss = 0
            for board in game:
                net.zero_grad()
                input_arr, output_label = board
                # To tensor
                input_tensor = torch.tensor(input_arr).unsqueeze(dim=0).to(dtype=torch.float32)
                result = net(input_tensor)
                # print(result)

                # Correct label
                label_tensor = torch.tensor([output_label]).unsqueeze(dim=0).to(dtype=torch.float32)
                loss = net.loss(result, label_tensor)
                #print(loss.item())
                current_loss += loss.item()
                game_loss += loss.item()
                loss.backward() 
                for p in net.parameters():
                    p.data.add_(-lr, p.grad.data)
            # print(f"Game loss: {game_loss}")
        avg_loss = current_loss / bs_size
        print(f"LOSS: {avg_loss}")

def evaluate(net, games, threshold=0.05, count=10):
    random_sample = get_random_sample(games, count=count)
    correct_boards = 0
    incorrect_boards = 0
    for game in random_sample:
        random_game_boards = get_random_sample(game, count=10)
        for board in random_game_boards:
            input_arr, output_label = board
            input_tensor = torch.tensor(input_arr).unsqueeze(dim=0).to(dtype=torch.float32)
            result = net(input_tensor)
            real_result = result[0].item()
            # print(f"CORRECT_RESULT: {output_label}")
            # print(f"PREDICTION: {real_result}")
            if abs(real_result - output_label) < threshold:
                correct_boards += 1
            else:
                incorrect_boards += 1
    print(f"CORRECT: {correct_boards}")
    print(f"INCORRECT: {incorrect_boards}")
    total = correct_boards + incorrect_boards
    print(f"{(correct_boards / total) * 100.0} %")


            
evaluate(net, games, threshold=0.4)
train(net, games, lr=0.0002, bs_size=20)
evaluate(net, games, threshold=0.4)
