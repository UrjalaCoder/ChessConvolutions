from network import ChessNet
from game_parser import load
import torch
from torch import optim
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

def get_vectorset(games):
    input_tensors = []
    output_tensors = []
    for game in games:
        input_tens, output_tens = game
        input_tensors.append(np.array(input_tens))
        output_tensors.append(np.array([output_tens]))
    input_tensors = np.array(input_tensors)
    return (torch.tensor(input_tensors).to(dtype=torch.float32), torch.tensor(output_tensors).to(dtype=torch.float32))

def train(net, games, epochs=5, bs_size=10, lr=0.05):
    optimizer = optim.Adam(net.parameters())
    for epoch in range(epochs):
        current_loss = 0
        for batch in range(20):
            bs = get_random_sample(games, count=bs_size)
            input_tensors, output_tensors = get_vectorset(bs)
            optimizer.zero_grad()
            # Old loop
            # To tensor
            result = net(input_tensors)
            # print(result)

            # Correct label
            loss = net.loss(result, output_tensors)
            current_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        avg_loss = current_loss / 20
        print(f"EPOCH: {epoch}, LOSS: {avg_loss}")

def evaluate(net, games, threshold=0.05, count=10):
    random_sample = get_random_sample(games, count=count)
    correct_boards = 0
    incorrect_boards = 0
    for board in random_sample:
        random_game_boards = get_random_sample(game, count=10)
        input_arr, output_label = board
        input_tensor = torch.tensor(input_arr).unsqueeze(dim=0).to(dtype=torch.float32)
        result = net(input_tensor)
        real_result = result[0].item()
        if abs(real_result - output_label) < threshold:
            correct_boards += 1
        else:
            incorrect_boards += 1
    print(f"CORRECT: {correct_boards}")
    print(f"INCORRECT: {incorrect_boards}")
    total = correct_boards + incorrect_boards
    print(f"{(correct_boards / total) * 100.0} %")

def load_premade(filename="value"):
    p = f"nets/{filename}.pth"
    net.load_state_dict(torch.load(p))

load_premade(filename="network2")

threshold = 0.4
evaluate(net, games, threshold=threshold, count=500)
#train(net, games, lr=1, bs_size=256, epochs=30)
#evaluate(net, games, threshold=threshold, count=500)

# Saving:
#torch.save(net.state_dict(), "nets/network2.pth")
