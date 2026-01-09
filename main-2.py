import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import os
from torchvision.utils import save_image 

torch.manual_seed(42)

NUM_CLIENTS = 5
ROUNDS = 3            # Reduced rounds slightly as CNNs learn faster
LOCAL_EPOCHS = 1      # How many times each client trains on their own data
BATCH_SIZE = 32
LR = 0.01             # Lower learning rate is better for Deep Learning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2. LOAD REAL DATASET (MNIST)
def get_mnist_datasets():
    # Transform: Convert images to tensors and normalize them (standard DL practice)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download MNIST (Train & Test)
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset

# 3. SPLIT DATA AMONG CLIENTS
def split_data(dataset, num_clients):
    # We split the 60,000 training images into 5 chunks (12,000 each)
    total_len = len(dataset)
    part_len = total_len // num_clients
    
    # Randomly shuffle indices so clients get random mix of digits
    indices = torch.randperm(total_len)
    
    client_data = []
    for i in range(num_clients):
        # Slice the indices for this client
        start = i * part_len
        end = start + part_len
        client_indices = indices[start:end]
        
        # Create a "Subset" just for this client
        subset = torch.utils.data.Subset(dataset, client_indices)
        client_data.append(subset)
        
    return client_data

# 4. save the data as images for each client
def save_client_samples(client_datasets, num_clients, num_samples=10):
    # Create a main directory
    base_dir = "client_data_visuals"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    print(f"Saving sample images to folder: {base_dir}/ ...")

    for cid in range(num_clients):
        # Create a subfolder for each client (e.g., client_0, client_1)
        client_dir = os.path.join(base_dir, f"client_{cid}")
        if not os.path.exists(client_dir):
            os.makedirs(client_dir)
            
        # Get the dataset for this client
        dataset = client_datasets[cid]
        
        # Save 'num_samples' images
        for i in range(num_samples):
            # dataset[i] returns (image_tensor, label)
            img, label = dataset[i] 
            
            # We need to un-normalize the image to make it look normal again
            # (MNIST was normalized with mean 0.1307 and std 0.3081)
            img = img * 0.3081 + 0.1307
            
            filename = os.path.join(client_dir, f"img_{i}_label_{label}.png")
            save_image(img, filename)
            
    print("Done saving images.")

# 5. DEFINE THE DEEP LEARNING MODEL (CNN)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # A simple Deep Learning architecture
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) # Layer 1: Convolution
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)# Layer 2: Convolution
        self.pool = nn.MaxPool2d(2, 2)                          # Downscaling
        self.fc1 = nn.Linear(32 * 7 * 7, 64)                    # Fully Connected Layer
        self.fc2 = nn.Linear(64, 10)                            # Output: 10 digits (0-9)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass input through layers
        x = self.pool(self.relu(self.conv1(x))) # Image size becomes 14x14
        x = self.pool(self.relu(self.conv2(x))) # Image size becomes 7x7
        x = x.view(-1, 32 * 7 * 7)              # Flatten (Matrix -> Vector)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Helper functions for Federated Learning
def get_params(model):
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}

def set_params(model, params):
    model.load_state_dict(params)

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def plot_figures(acc_history, client_histories=None, figure_size=(8, 5), title_figure="Federated Learning on MNIST (CNN)"):
    plt.figure(figsize=figure_size)
    
    if client_histories:
        # Plot global thick
        plt.plot(range(1, ROUNDS + 1), acc_history, marker='o', linewidth=3, color='black', label='Global Model')
        
        # Plot each client faintly
        for cid in range(NUM_CLIENTS):
                plt.plot(
                    range(1, ROUNDS + 1), 
                    client_histories[cid],
                    linestyle='--', alpha=0.5, label=f'Client {cid}'
                )
        plt.legend()
    else:
        plt.plot(range(1, ROUNDS + 1), acc_history, marker='o')

    plt.title(title_figure)
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(title_figure + ".png")
    plt.show()
    plt.close()

# === MAIN FEDERATED TRAINING LOOP ===
train_data, test_data = get_mnist_datasets()
client_datasets = split_data(train_data, NUM_CLIENTS)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)
save_client_samples(client_datasets, NUM_CLIENTS)

# Initialize Global Server Model
global_model = CNN().to(device)
global_params = get_params(global_model)

acc_history = []
client_histories = [[] for _ in range(NUM_CLIENTS)] # New list to store each client's history

for rnd in range(1, ROUNDS + 1):
    print(f"\n=== Round {rnd} ===")
    local_params = []
    local_sizes = []
    
    # --- CLIENTS TRAINING ---
    for cid in range(NUM_CLIENTS):
        # 1. Receive model from server
        local_model = CNN().to(device)
        set_params(local_model, global_params)
        
        # 2. Prepare local data loader
        train_loader = torch.utils.data.DataLoader(client_datasets[cid], batch_size=BATCH_SIZE, shuffle=True)
        optimizer = optim.SGD(local_model.parameters(), lr=LR)
        loss_fn = nn.CrossEntropyLoss()
        
        # 3. Local Training (Deep Learning happens here)
        local_model.train()
        for epoch in range(LOCAL_EPOCHS):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                output = local_model(images)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()
        
        # 4. Collect updates
        local_params.append(get_params(local_model))
        local_sizes.append(len(client_datasets[cid]))
        
        # === UNCOMMENT THESE LINES BELOW ===
        acc = evaluate(local_model, test_loader) 
        client_histories[cid].append(acc) # Save this client's score
        print(f" Client {cid} local accuracy: {acc*100:.2f}%")

    # --- SERVER AGGREGATION ---
    total_samples = sum(local_sizes)
    new_params = {k: torch.zeros_like(v) for k, v in global_params.items()}
    
    for params, size in zip(local_params, local_sizes):
        weight = size / total_samples
        for k in params:
            new_params[k] += params[k] * weight
            
    global_params = new_params
    set_params(global_model, global_params)
    
    # --- GLOBAL EVALUATION ---
    global_acc = evaluate(global_model, test_loader)
    acc_history.append(global_acc)
    print(f" Global Accuracy after round {rnd}: {global_acc*100:.2f}%")

torch.save(global_model.state_dict(), "fedavg_mnist_cnn.pth")
print("Saved global model to fedavg_mnist_cnn.pth")

plot_figures(acc_history)
plot_figures(acc_history, client_histories=client_histories, figure_size=(10, 6), title_figure="Federated Learning Clients vs Global Model")