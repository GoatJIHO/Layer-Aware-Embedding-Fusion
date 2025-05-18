import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from data_processing.dataset_loader import CustomDataset

def train_fuc(model, train_embedding, test_embedding, train_label, test_label, device="cuda"):
    BATCH_SIZE = 1024
    EPOCHS = 100
    LR = 1e-4

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    train_dataset = CustomDataset(train_embedding, train_label)
    test_dataset = CustomDataset(test_embedding, test_label)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            *x_batch, y_batch = batch
            x_batch = tuple(x.to(device) for x in x_batch)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(*x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                *x_val, y_val = batch
                x_val = tuple(x.to(device) for x in x_val)
                y_val = y_val.to(device)
                outputs = model(*x_val)
                _, preds = torch.max(outputs, 1)
                correct += (preds == y_val).sum().item()
                total += y_val.size(0)
        valid_acc = correct / total
        best_val_acc = max(best_val_acc, valid_acc)

    print(f"{best_val_acc:.4f}")