import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 
import numpy as np
from tqdm import tqdm 
import argparse 
import os 
import json 
import copy
import time 


torch.manual_seed(10)
np.random.seed(10)
device = torch.device('cuda:0')

# Argparse 
args = argparse.ArgumentParser(description='args')
args.add_argument('--dataset', default='DM', type=str)
args.add_argument('--batch', default=128, type=int)
args.add_argument('--dims', default=768, type=int)
args.add_argument('--lr', default=0.0005, type=float)
args.add_argument('--emb_hidden', default=768, type=int)
args.add_argument('--hidden', default=320, type=int)
args.add_argument('--early_stop', default=30, type=int)
args.add_argument('--epochs', default=10000, type=int)
args.add_argument('--alpha', default=0.57, type=float)
args = args.parse_args() 


class ReviewDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        self.data = []
        for user_id, samples in data.items():
            for sample in samples:
                self.data.append((int(user_id), sample[0], sample[1]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def load_data(train_path, val_path, test_path):
    train_data = ReviewDataset(train_path)
    val_data = ReviewDataset(val_path)
    test_data = ReviewDataset(test_path)
    return train_data, val_data, test_data


def collate_fn(batch):
    user_ids = torch.tensor([x[0] for x in batch], dtype=torch.long)
    ratings = torch.tensor([x[1] for x in batch], dtype=torch.float32)
    items = torch.tensor([x[2] for x in batch], dtype=torch.long)
    
    return user_ids, items, ratings

def process_target_review(user_ids, item_ids, map_data, target_reviews):
        
        user_ids_b = user_ids.unsqueeze(1)
        item_ids_b = item_ids.unsqueeze(1)
        
        map_data = torch.tensor(map_data, dtype=torch.long, device=device)
        
        import pandas as pd
        df_map = pd.DataFrame(map_data.cpu().numpy(), columns=["user_id", "item_id"])
        df_map_unique = df_map.drop_duplicates(subset=["user_id", "item_id"], keep="first")
        map_data = torch.tensor(df_map_unique.values, dtype=torch.long, device=device)

        user_match = (map_data[:,0] == user_ids_b)
        item_match = (map_data[:,1] == item_ids_b)
        
        match_indices = torch.nonzero(user_match & item_match, as_tuple=False)
        
        tr_reviews = target_reviews[match_indices[:,1].cpu().numpy()]
        
        return tr_reviews

class TR_predictor(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_dim, emb_hidden, reviews, maps):
        super(TR_predictor, self).__init__()
        
        self.encoded_review = reviews[0]
        self.map_pair = maps[0]

        self.user_rep = nn.Embedding(num_users, 1024)
        self.item_rep = nn.Embedding(num_items, 1024)
        
        self.user_rep.weight = nn.Parameter(torch.from_numpy(reviews[1]).to(torch.float32))
        self.item_rep.weight = nn.Parameter(torch.from_numpy(reviews[2]).to(torch.float32))
        self.user_rep.weight.requires_grad = False
        self.item_rep.weight.requires_grad = False
        
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        nn.init.xavier_uniform_(self.user_bias.weight)
        nn.init.xavier_uniform_(self.item_bias.weight)
        

        self.UFC = nn.Sequential(
            nn.Linear(1024, emb_hidden),
            nn.LayerNorm(emb_hidden),
            nn.SiLU(),
            nn.Dropout()
        )
        
        self.IFC = nn.Sequential(
            nn.Linear(1024, emb_hidden),
            nn.LayerNorm(emb_hidden),
            nn.SiLU(),
            nn.Dropout()
        ) 
        

        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim + emb_hidden * 2 , hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, 1, bias = False)
        )
    
    def forward(self, user_ids, item_ids, clip):
        
        user_rep = self.user_rep(user_ids)
        item_rep = self.item_rep(item_ids)
        
        user_embs = self.UFC(user_rep)
        item_embs = self.IFC(item_rep)
        
        f_rep = torch.cat((user_embs,item_embs), dim=1).to(device)
        
        user_biases = self.user_bias(user_ids).squeeze()
        item_biases = self.item_bias(item_ids).squeeze()
       
        target_review = process_target_review(user_ids, item_ids, self.map_pair, self.encoded_review)
        target_embeds = torch.tensor(target_review, dtype=torch.float32, device=device)
        
        embeds = torch.cat((f_rep,target_embeds), dim=1)
        
        predict = self.predictor(embeds).squeeze()
        prediction = predict + user_biases + item_biases 

        if clip==1:
            prediction = torch.clamp(prediction,1,5) 
        
        return prediction


class TR_hidden(nn.Module): 
    def __init__(self, num_users, num_items, embedding_dim, emb_hidden, hidden_dim, hidden_embeds, reviews, map_hidden):
        super(TR_hidden, self).__init__()
        
        self.user_rep = nn.Embedding(num_users, 1024)
        self.item_rep = nn.Embedding(num_items, 1024)
        
        self.user_rep.weight = nn.Parameter(torch.from_numpy(reviews[0]).to(torch.float32))
        self.item_rep.weight = nn.Parameter(torch.from_numpy(reviews[1]).to(torch.float32))
        self.user_rep.weight.requires_grad = False
        self.item_rep.weight.requires_grad = False
        
        self.h_hidden = hidden_embeds
        self.map_hidden = map_hidden
        
        self.UFC = nn.Sequential(
            nn.Linear(1024, emb_hidden),
            nn.LayerNorm(emb_hidden),
            nn.SiLU(),
            nn.Dropout()
        )
        
        self.IFC = nn.Sequential(
            nn.Linear(1024, emb_hidden),
            nn.LayerNorm(emb_hidden),
            nn.SiLU(),
            nn.Dropout()
        ) 
        
        self.RG = nn.Sequential(
            nn.Linear(emb_hidden * 2, emb_hidden),
            nn.LayerNorm(emb_hidden),
            nn.SiLU(),
            nn.Linear(emb_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
         
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
    
    def forward(self, user_ids, item_ids, clip):
        
        user_rep = self.user_rep(user_ids)
        item_rep = self.item_rep(item_ids)
        
        user_embs = self.UFC(user_rep)
        item_embs = self.IFC(item_rep)
        
        f_rep = torch.cat((user_embs,item_embs), dim=1).to(device)
        f_hidden = self.RG(f_rep)
        
        h_embeds = process_target_review(user_ids, item_ids, self.map_hidden, self.h_hidden)
        h_hidden = torch.tensor(h_embeds, dtype=torch.float32, device=device)
        
        h_loss = self.mse_loss(f_hidden, h_hidden)
        target = torch.ones(len(f_hidden)).to(device)
        cos_loss = self.cosine_loss(f_hidden, h_hidden, target)
        
        return h_loss, cos_loss
        

class Predictor(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, emb_hidden, hidden_dim, reviews):
        super(Predictor, self).__init__()

        self.user_rep = nn.Embedding(num_users, 1024)
        self.item_rep = nn.Embedding(num_items, 1024)
        
        self.user_rep.weight = nn.Parameter(torch.from_numpy(reviews[0]).to(torch.float32))
        self.item_rep.weight = nn.Parameter(torch.from_numpy(reviews[1]).to(torch.float32))
        self.user_rep.weight.requires_grad = False
        self.item_rep.weight.requires_grad = False
       
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        nn.init.xavier_uniform_(self.user_bias.weight)
        nn.init.xavier_uniform_(self.item_bias.weight)
        
        self.UFC = nn.Sequential(
            nn.Linear(1024, emb_hidden),
            nn.LayerNorm(emb_hidden),
            nn.SiLU(),
            nn.Dropout()
        )
        
        self.IFC = nn.Sequential(
            nn.Linear(1024, emb_hidden),
            nn.LayerNorm(emb_hidden),
            nn.SiLU(),
            nn.Dropout()
        ) 
        
        self.RG = nn.Sequential(
            nn.Linear(emb_hidden * 2, emb_hidden),
            nn.LayerNorm(emb_hidden),
            nn.SiLU(),
            nn.Linear(emb_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, 1, bias = False)
        )
        
    
    def forward(self, user_ids, item_ids, clip):
        
        user_rep = self.user_rep(user_ids)
        item_rep = self.item_rep(item_ids)
        
        user_embs = self.UFC(user_rep)
        item_embs = self.IFC(item_rep)
    
        user_biases = self.user_bias(user_ids).squeeze()
        item_biases = self.item_bias(item_ids).squeeze()
        
        f_rep = torch.cat((user_embs,item_embs), dim=1).to(device)
        f_embeds = self.RG(f_rep)
        
        predict = self.predictor(f_embeds).squeeze()
        prediction = predict + user_biases + item_biases 
        
        if clip==1:
            prediction = torch.clamp(prediction,1,5)
        
        return prediction
        
def train1(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
   
    for user_ids, item_ids, ratings in dataloader:
        user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)
        optimizer.zero_grad()
        outputs = model(user_ids, item_ids, clip=0)

        if outputs.ndimension() == 0:
            outputs = outputs.unsqueeze(0)
            
        loss = criterion(outputs, ratings)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        total_loss += loss.item()
        
        
    return total_loss / len(dataloader)

def train2(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    total_trloss = 0
    total_cosloss = 0
   
    for user_ids, item_ids, ratings in dataloader:
        user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)
        optimizer.zero_grad()
        output_loss, cos_loss = model(user_ids, item_ids, clip=0)

        if output_loss.ndimension() == 0:
            output_loss = output_loss.unsqueeze(0)
        
        loss = args.alpha * output_loss + (1-args.alpha) * cos_loss 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        total_loss += loss.item()
        total_trloss += output_loss.item()
        total_cosloss += cos_loss.item()
        
        
    return total_loss / len(dataloader), total_trloss / len(dataloader), total_cosloss / len(dataloader)

def evaluate1(model, dataloader, device, criterion):
    model.eval()
    predictions = []
    actuals = []
    total_mae = 0
    total_mse = 0
    total_rmse = 0
    
    with torch.no_grad(): 
        for user_ids, item_ids, ratings in dataloader:
            user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)
            
            outputs = model(user_ids, item_ids, clip=1)

            if outputs.ndimension() == 0:
                outputs = outputs.unsqueeze(0)
                
            mae = (ratings - outputs).abs().mean()
            mse = criterion(outputs,ratings)
            rmse = np.sqrt(mse.item())
            total_mae += mae.item()
            total_mse += mse.item()
            total_rmse += rmse.item()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(ratings.cpu().numpy())
    
    return total_mse / len(dataloader), total_rmse / len(dataloader), total_mae / len(dataloader)

def evaluate2(model, dataloader):
    model.eval()
    total_trloss = 0
    total_cosloss = 0
    
    with torch.no_grad(): 
        for user_ids, item_ids, ratings in dataloader:
            user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)
            
            output_loss, cos_loss = model(user_ids, item_ids, clip=1)
            
            if output_loss.ndimension() == 0:
                output_loss = output_loss.unsqueeze(0)

            total_trloss += output_loss.item()
            total_cosloss += cos_loss.item()
    
    return total_trloss / len(dataloader), total_cosloss / len(dataloader)

def save_hidden_maps(model, dataloader, device):
    model.eval()
    map_hidden = []
    with torch.no_grad():
        for user_ids, item_ids, ratings in dataloader:
            user_ids, item_ids = user_ids.to(device), item_ids.to(device)

            _ = model(user_ids, item_ids, clip=1)
            
            for u_id, i_id in zip(user_ids.cpu().numpy(), item_ids.cpu().numpy()):
                map_hidden.append([u_id, i_id])
    
    return map_hidden

def save_model(model, path):
    torch.save(model.state_dict(), path)
    

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
              
def main():

    train_path = ''
    val_path = ''
    test_path = ''
    
    train_data, val_data, test_data = load_data(train_path, val_path, test_path)

    num_users = max([int(user_id) for user_id, _, _ in train_data.data]) + 1
    num_items = max([asin for _,_, asin in train_data.data]) + 1
    
    print('Data Load Start.....\n')
    print('Datasets: ', args.dataset)
    print('Total user: ', num_users)
    print('Total item: ', num_items)
    print('\n')
    
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
   
    encoded_review = np.load('', allow_pickle=True)
    item_review = np.load('', allow_pickle=True)
    user_review = np.load('', allow_pickle=True)
    map_pair = np.load('', allow_pickle=True)

    
    print('Data Load Finished! \n')
    
    print('Start Training..... \n')
    print('learning_rate: ', args.lr)
    print('batch_size: ', args.batch)
    print('emb_hidden: ', args.emb_hidden)
    print('hidden: ', args.hidden)
    print('\n')
    
    
    print('Training 1: TR Predictor \n')
    model1 = TR_predictor(num_users, num_items, embedding_dim=args.dims, hidden_dim=args.hidden, emb_hidden=args.emb_hidden, reviews=[encoded_review,user_review,item_review],maps=[map_pair]).to(device)
    optimizer1 = optim.Adam(model1.parameters(),lr=args.lr, weight_decay=1e-4)
    criterion1 = nn.MSELoss()
    
    early_stop_epochs = args.early_stop
    updated = 0
    best = 100 
    best_model1 = None
    best_path1 = None

    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_loss = train1(model1, train_loader, optimizer1, criterion1, device)    
        val_mse, val_rmse, val_mae = evaluate1(model1, val_loader, device, criterion1)
        
        end_time = time.time()
        epoch_duration = end_time - start_time
        
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val MSE: {val_mse:.4f}, Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}, Epoch Duration: {epoch_duration:.2f} seconds') 
        
        if best > val_mse:
            print('best model changed!')
            best = val_mse
            best_path1 = f'./saved_model/{args.dataset}/3step_step1_PFD_v2_lr{args.lr}_hidden{args.hidden}_{epoch}'
            best_model1 = copy.deepcopy(model1.state_dict())
            updated = 0
        
        updated += 1 
        if updated > early_stop_epochs and epoch > 0:
            print('Early Stopping Triggered! \n')
            break
        
    if best_model1 is not None:
        torch.save(best_model1, best_path1)
        print('Model1 saved! \n')    
    
    print('Training 3: Generate Hidden embeds \n')
    

    model1 = TR_predictor(num_users, num_items, embedding_dim=args.dims, hidden_dim=args.hidden, emb_hidden=args.emb_hidden, reviews=[encoded_review,user_review,item_review],maps=[map_pair]).to(device)
    model1.load_state_dict(best_model1)
    model1.to(device)
    
    hidden_embeds = []
    def hook_fn(module, input, output):
        hidden_embeds.append(output.detach().cpu().numpy()) 
    
    hook = model1.predictor[5].register_forward_hook(hook_fn)
    map_hidden = save_hidden_maps(model1, train_loader, device)
    map_hidden.extend(save_hidden_maps(model1, val_loader, device))
    map_hidden.extend(save_hidden_maps(model1, test_loader, device))
    hidden_embeds = np.vstack(hidden_embeds)
    hook.remove()
    
    model3 = TR_hidden(num_users, num_items, embedding_dim=args.dims, emb_hidden=args.emb_hidden, hidden_dim=args.hidden, hidden_embeds=hidden_embeds, reviews=[user_review,item_review], map_hidden=map_hidden).to(device)    
    
    for layer_m4, layer_m3 in zip(model3.IFC, model1.IFC):
        if hasattr(layer_m4, 'weight') and hasattr(layer_m3, 'weight'):
            layer_m4.weight.data = layer_m3.weight.data.clone()
        if hasattr(layer_m4, 'bias') and hasattr(layer_m3, 'bias'):
            layer_m4.bias.data = layer_m3.bias.data.clone()
    
    for layer_m4, layer_m3 in zip(model3.UFC, model1.UFC):
        if hasattr(layer_m4, 'weight') and hasattr(layer_m3, 'weight'):
            layer_m4.weight.data = layer_m3.weight.data.clone()
        if hasattr(layer_m4, 'bias') and hasattr(layer_m3, 'bias'):
            layer_m4.bias.data = layer_m3.bias.data.clone()
    
    optimizer3 = optim.Adam(model3.parameters(),lr=args.lr, weight_decay=1e-4)
    
    early_stop_epochs = args.early_stop
    updated = 0
    best = 100 
    best_model3 = None
    best_path3 = None
  
    for epoch in range(0,args.epochs):
        start_time = time.time()
        
        train_loss, train_tr, train_cos = train2(model3, train_loader, optimizer3)
        val_tr, val_cos = evaluate2(model3, val_loader)
        
        end_time = time.time()
        epoch_duration = end_time - start_time
        
        print(f'Epoch {epoch+1}/{args.epochs}, Train loss: {train_loss:.4f}, Train tr: {train_tr:.4f}, Train cos: {train_cos:.4f}, Val tr: {val_tr:.4f}, Val cos: {val_cos:.4f}, Epoch Duration: {epoch_duration:.2f} seconds') 
        
        if best > val_tr:
            print('best model changed!')
            best = val_tr
            best_path3 = f'./saved_model/{args.dataset}/3step_step2_PFD_v2_lr{args.lr}_hidden{args.hidden}_{epoch}'
            best_model3 = copy.deepcopy(model3.state_dict())
            updated = 0
        
        updated += 1 
        if updated > early_stop_epochs and epoch > 0:
            print('Early Stopping Triggered! \n')
            break

    if best_model3 is not None:
        torch.save(best_model3, best_path3)
        print('Model3 saved! \n')
    
    print('Training 4: Final Prediction \n')
    

    model3 = TR_hidden(num_users, num_items, embedding_dim=args.dims, emb_hidden=args.emb_hidden, hidden_dim=args.hidden, hidden_embeds=hidden_embeds, reviews=[user_review, item_review], map_hidden=map_hidden ).to(device)
    model3.load_state_dict(best_model3)
    model3.to(device)
    
    model4 = Predictor(num_users, num_items, embedding_dim=args.dims, emb_hidden=args.emb_hidden, hidden_dim=args.hidden, reviews=[user_review, item_review]).to(device)
    
    for layer_m4, layer_m3 in zip(model4.IFC, model3.IFC):
        if hasattr(layer_m4, 'weight') and hasattr(layer_m3, 'weight'):
            layer_m4.weight.data = layer_m3.weight.data.clone()
        if hasattr(layer_m4, 'bias') and hasattr(layer_m3, 'bias'):
            layer_m4.bias.data = layer_m3.bias.data.clone()
    
    for layer_m4, layer_m3 in zip(model4.UFC, model3.UFC):
        if hasattr(layer_m4, 'weight') and hasattr(layer_m3, 'weight'):
            layer_m4.weight.data = layer_m3.weight.data.clone()
        if hasattr(layer_m4, 'bias') and hasattr(layer_m3, 'bias'):
            layer_m4.bias.data = layer_m3.bias.data.clone() 
            
    for layer_m4, layer_m3 in zip(model4.RG, model3.RG):
        if hasattr(layer_m4, 'weight') and hasattr(layer_m3, 'weight'):
            layer_m4.weight.data = layer_m3.weight.data.clone()
        if hasattr(layer_m4, 'bias') and hasattr(layer_m3, 'bias'):
            layer_m4.bias.data = layer_m3.bias.data.clone()
    
    for param in model4.RG.parameters():
        param.requires_grad = False
    
    for param in model4.IFC.parameters():
        param.requires_grad = False
    
    for param in model4.UFC.parameters():
        param.requires_grad = False
    
    optimizer4 = optim.Adam(model4.parameters(),lr=args.lr, weight_decay=1e-4) 
    criterion4 = nn.MSELoss()
    
    early_stop_epochs = args.early_stop
    updated = 0
    best = 100 
    best_model4 = None
    best_path4 = None
    
    for epoch in range(0,args.epochs):
        start_time = time.time()
        
        train_loss = train1(model4, train_loader, optimizer4, criterion4, device)
        val_mse, val_rmse, val_mae = evaluate1(model4, val_loader, device, criterion4)
        
        end_time = time.time()
        epoch_duration = end_time - start_time
        
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val MSE: {val_mse:.4f}, Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}, Epoch Duration: {epoch_duration:.2f} seconds') 
        
        if best > val_mse:
            print('best model changed!')
            best = val_mse
            best_path4 = f'./saved_model/{args.dataset}/3step_step3_PFD_v2_lr{args.lr}_hidden{args.hidden}_{epoch}'
            best_model4 = copy.deepcopy(model4.state_dict())
            updated = 0
        
        updated += 1 
        if updated > early_stop_epochs and epoch > 0:
            step3_end_time = time.time()
            print('Early Stopping Triggered! \n')
            break

    if best_model4 is not None:
        torch.save(best_model4, best_path4)
        print('Model4 saved! \n')

    model1.load_state_dict(best_model1)
    model1.to(device)
    mse, rmse, mae = evaluate1(model1, test_loader, device, criterion1)
    print(f'Step1 RMSE: {rmse:.4f}, Step1 MSE: {mse:.4f}, Step1 MAE: {mae:.4f}')
    
    model4.load_state_dict(best_model4)
    model4.to(device)

    test_mse, test_rmse, test_mae = evaluate1(model4, test_loader, device, criterion4)
    print(f'Test RMSE: {test_rmse:.4f}, Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}') 
    
    model3.load_state_dict(best_model3)
    model3.to(device)

    test_tr, test_cos = evaluate2(model3, test_loader)
    print(f'Test tr: {test_tr:.4f}, Test cos: {test_cos:.4f} \n')

if __name__ == '__main__': 
    main()

    