import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CustomDataset
from utils import simple_ctc_decode as ctc_decode
from utils import itos
from model import Conv3DLSTMModel, Conv3DLSTMModelMini, LipNet
from tqdm import tqdm
import glob
from pathlib import Path

def train_one_epoch(optimizer,
                    train_loader,
                    ctc_loss,
                    device,
                    model):
    total_loss = 0
    for frame, align in tqdm(train_loader):
        frame = frame.type(torch.FloatTensor)
        frame = frame.to(device)
        y = np.array(align)
        y = torch.from_numpy(y)
        y = y.to(device)
        
        pred = model(frame)

        probs = pred.permute(1,0,2) # (B, T, C) -> (T, B, C)

        target_lengths = []
        y_true = []

        for seq in y:
            length = (seq != 38).sum()
            y_true.extend(seq[:length].tolist())
            target_lengths.append(length)

        target_lengths = torch.tensor(target_lengths, dtype=torch.long).to(device)
        targets = torch.tensor(y_true, dtype=torch.long).to(device)

        # All input sequences use the full 75 timesteps
        input_lengths = torch.full((frame.size(0),), 75, dtype=torch.long).to(device)

        loss = ctc_loss(probs, targets, input_lengths, target_lengths)
        total_loss += loss.item()
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        word = [ ]
        words= [ ]

        for i in range(y.shape[0]):
            for n in range(75):
                max = torch.argmax(pred[i][n][:])
                word.append(max.cpu().detach().numpy())
            words.append(word)
            word = []
        words = np.stack(words , axis=0)
    
    # prinint the first sentence of last batch
    print(f"Predicted Sentence: {itos(words[0])}")
    print("CTC Decoded Sentence:", ctc_decode(itos(words[0])))
    print("Original Sentence:", itos(y[0]) )
    
    total_loss /= len(train_loader)
    return total_loss, words[0]

def valid_one_epoch(valid_loader,
                    ctc_loss,
                    device,
                    model):
    total_loss = 0
    for frame, align in tqdm(valid_loader):
        frame = frame.type(torch.FloatTensor)
        frame = frame.to(device)
        y = np.array(align)
        y = torch.from_numpy(y)
        y = y.to(device)

        probs = model(frame).permute(1,0,2) # (B, T, C) -> (T, B, C)

        target_lengths = []
        y_true = []

        for seq in y:
            length = (seq != 38).sum()
            y_true.extend(seq[:length].tolist())
            target_lengths.append(length)

        target_lengths = torch.tensor(target_lengths, dtype=torch.long).to(device)
        targets = torch.tensor(y_true, dtype=torch.long).to(device)

        # All input sequences use the full 75 timesteps
        input_lengths = torch.full((frame.size(0),), 75, dtype=torch.long).to(device)

        loss = ctc_loss(probs, targets, input_lengths, target_lengths)
        total_loss += loss.item()

    total_loss /= len(valid_loader)
    return total_loss


def train_lipnet(opts):
    EPOCHS = opts.epoch
    lr = opts.lr
    hidden_size = opts.hidden_size
    model_name = opts.model
    batch_size = opts.batch
    num_workers = opts.workers
    device = opts.device
    vocab_size = 40 

    dataset_path = os.getcwd() + "/data/alignments/s1/*.align"
    files = glob.glob(dataset_path)

    criterion = nn.CTCLoss(blank=39)

    if model_name == 'conv3dlstm':
        model = Conv3DLSTMModel(vocab_size, hidden_size).to(device)
    elif model_name == 'conv3dlstmmini':
        model = Conv3DLSTMModelMini(vocab_size, hidden_size).to(device)
    else:
        model = LipNet(vocab_size, hidden_size).to(device)

    MODEL_PATH = Path('models')

    if not os.path.isdir(MODEL_PATH):
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
    elif len(os.listdir(MODEL_PATH)) > 0:
        MODEL_LOAD_PATH = MODEL_PATH / os.listdir(MODEL_PATH)[-1]
        print("Loading model..")
        model.load_state_dict(torch.load(f=MODEL_LOAD_PATH))

    optimizer = optim.Adam(model.parameters(), lr)

    train_data = CustomDataset(files[:900])
    valid_data = CustomDataset(files[900:])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    summary_loss = {"train_loss" : [],
                    "valid_loss" : []}
    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch}")
        print(optimizer.param_groups[0]["lr"])
        
        train_loss, words = train_one_epoch(optimizer=optimizer,
                                               train_loader=train_loader,
                                               ctc_loss=criterion,
                                               device=device,
                                               model=model)
        
        valid_loss = valid_one_epoch(valid_loader=valid_loader,
                                     ctc_loss=criterion,
                                     device=device,
                                     model=model)
        
        print(f"Train Loss: {train_loss} Valid Loss: {valid_loss}")
        
        if ((epoch + 1) > ((epoch // 100) * 100 + 60)) and ((epoch + 1) <= ((epoch // 100) + 1) * 100):
            optimizer.param_groups[0]["lr"] *= np.exp(-0.1)
        else:
            optimizer.param_groups[0]["lr"] = lr

        if epoch % 20 == 0:
            print("Saving model..")
            MODEL_SAVE_PATH = MODEL_PATH/ (model_name + (f"-{epoch:>03d}.pth"))
            torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

        summary_loss["train_loss"].append(train_loss)
        summary_loss["valid_loss"].append(valid_loss)
        

    return summary_loss