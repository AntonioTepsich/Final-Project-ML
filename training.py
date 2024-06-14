import torch
from torch import nn
from torch.optim import Adam
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json




def train_model(train_data_loader, architecture, file_name, epochs=50, learning_rate=0.001, model=None, device="cpu"):
    architecture = architecture.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(architecture.parameters(), lr=learning_rate)
    running_losses = []

    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=f'runs/{file_name}')

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
        epoch_running_loss = 0
        progress_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc=f"Epoch {epoch+1}", leave=True)
        for i, data in progress_bar:
            gray, color = data
            gray = gray.float().to(device)
            color = color.float().to(device)

            optimizer.zero_grad()
            outputs = architecture(gray)
            loss = criterion(outputs, color)
            loss.backward()
            optimizer.step()

            epoch_running_loss += loss.item()
            progress_bar.set_postfix(loss=f'{loss.item():.4f}')
        
        epoch_loss = epoch_running_loss / len(train_data_loader)
        running_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

        # Log the loss to TensorBoard
        writer.add_scalar('Training Loss', epoch_loss, epoch)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": architecture.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "time": time.time() - start + initial_time,
                "running_losses": running_losses
            }, f"./trained_models/{file_name}_{learning_rate}_{epoch}.pt")
    
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
    }, f"./trained_models/{file_name}_{learning_rate}_full.pt")

    writer.close()