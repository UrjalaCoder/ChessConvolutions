from network import ChessNet
from game_parser import load
import torch
from torch import optim
import numpy as np
import math
import argparse

def get_random_sample(arr, count=10):
    choices = []
    s = len(arr)
    for i in range(count):
        index = math.floor(np.random.uniform() * s)
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
    device_str = ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device_str}")
    device = torch.device(device_str)
    optimizer = optim.Adam(net.parameters())
    losses = []
    
    # Convert net to CUDA (if possible)
    if device_str != "cpu":
        net = net.to(device)

    for epoch in range(epochs):
        current_loss = 0
        for batch in range(20):
            bs = get_random_sample(games, count=bs_size)
            input_tensors, output_tensors = get_vectorset(bs)
            
            # CUDA support
            if device_str != "cpu":
                input_tensors = input_tensors.to(device)
                output_tensors = output_tensors.to(device)

            optimizer.zero_grad()
            result = net(input_tensors)
            loss = net.loss(result, output_tensors)
            current_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_loss = current_loss / 20
        losses.append(avg_loss)
        print(f"EPOCH: {epoch}, LOSS: {avg_loss}")
    return losses

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

def load_premade(net, filename="network1"):
    p = f"nets/{filename}.pth"
    net.load_state_dict(torch.load(p))
    return net

def save_net(net, filename="network1"):
    p = f"nets/{filename}.pth"
    torch.save(net.state_dict(), p)

def main():
    parser = argparse.ArgumentParser(description="Network trainer and evaluator")
    parser.add_argument("filename", type=str, help="Filename of file used for training/evaluation (without extension!)", default="network1")
    
    # Evaluation args
    parser.add_argument("--threshold", type=float, help="Threshold for evaluation", default=1.0)
    parser.add_argument("--count", type=int, help="Evaluation board count", default=10)
    
    # Training args
    parser.add_argument("--train", type=bool, help="Train the network", default=False)
    parser.add_argument("--bs", type=int, help="Batch size for training", default=0)
    parser.add_argument("--epochs", type=int, help="Epoch count for training", default=0)
    parser.add_argument("--dataset", type=str, help="Dataset used for training", default="data") 

    args = parser.parse_args()
    dataset_name = args.dataset
    
    games = load(filename=dataset_name)

    filename = args.filename
    training = args.train
    if training is True:
        bs = args.bs
        epochs = args.epochs
        print("Training the network...")
        print(f"Dataset {dataset_name}, size: {len(games)}")
        net = ChessNet()
        train(net, games, epochs=epochs, bs_size=bs)
        np.save
    else:
        threshold = args.threshold
        count = args.count
        print("Evaluating the network...")
        net = ChessNet()
        net = load_premade(net, filename=filename)
        evaluate(net, games, threshold=threshold, count=count)

if __name__ == "__main__":
    main()
