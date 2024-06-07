import torch
from torch import nn
from torch.optim import Adam
import time
from tqdm import tqdm
import json



def train_model(train_data_loader, architecture, file_name, epochs=50, learning_rate=0.001, model=None, device="cpu"):
    """
    Trains the model with the given examples.
    :param train_data_loader: DataLoader for the training dataset.
    :param architecture: The architecture of the model.
    :param file_name: The file name prefix to be used for storing model and results.
    :param epochs: The number of epochs.
    :param learning_rate: The learning rate.
    :param model: (Optional) The pre-trained model.
    """
    # Fijarse si deberia pasar el modelo a la GPU
    architecture = architecture.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(architecture.parameters(), lr=learning_rate)

    running_losses = []

    # Checking if there is a pre-trained model to be loaded.
    if model is not None:
        checkpoint = torch.load(model)
        architecture.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        initial_time = checkpoint["time"]
        running_losses = checkpoint["running_losses"]
    else:
        initial_time = 0

    print(f"Number of parameters: {sum(p.numel() for p in architecture.parameters())}")

    start = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        epoch_running_loss = 0
        progress_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc=f"Epoch {epoch+1}", leave=True)
        for i, data in progress_bar:
            gray, color = data
            gray = gray.float()
            color = color.float()

            optimizer.zero_grad()
            outputs = architecture(gray)
            loss = criterion(outputs, color)
            loss.backward()
            optimizer.step()

            epoch_running_loss += loss.item()
            progress_bar.set_postfix(loss=f'{loss.item():.4f}')
        
        running_losses.append(epoch_running_loss)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": architecture.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "time": time.time() - start + initial_time,
                "running_losses": running_losses
            }, f"./{file_name}_{learning_rate}_{epoch}.pt")
    
    print("Finished Training")

    results = {"losses": running_losses}

    # Store losses in json file
    with open(f"figures/training_{file_name}_{learning_rate}_losses.json", "w") as results_file:
        json.dump(results, results_file)

    # Save the final model
    torch.save({
        "epoch": epochs,
        "model_state_dict": architecture.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "time": time.time() - start + initial_time,
        "running_losses": running_losses
    }, f"./{file_name}_{learning_rate}_full.pt")


def load_model(model_path, architecture, device="cpu"):
    """
    Loads a pre-trained model.
    :param model_path: The path to the model.
    :param architecture: The architecture of the model.
    :return: The pre-trained model.
    """
    architecture = architecture.to(device)
    checkpoint = torch.load(model_path)
    architecture.load_state_dict(checkpoint["model_state_dict"])
    return architecture