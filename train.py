from dataprocess import *
from gnn_model import *
import torch
from torch_geometric.data import DataLoader
import sklearn
from sklearn.metrics import r2_score
from tqdm import tqdm

train_data = makedataset(root='/path/test_data/')
test_data = makedataset(root='/path/test_data/')

np.random.seed(0)
model = Net(19).to(device)
kfold = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=0)
learning_rate = 1e-4
weight_decay = 1e-6
num_epochs = 230
R2_TRAIN = []
R2_VAL = []
best_r2_val = -float('inf')
best_model_path = './best_model.pth'

for fold, (tr_idx, va_idx) in enumerate(kfold.split(train_data)):
    train_dataset = train_data[tr_idx]
    batch_data = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = train_data[va_idx]
    batch_data_val = DataLoader(val_dataset, batch_size=32, shuffle=True)
    batch_data_test = DataLoader(test_data, batch_size=32, shuffle=True)
    criterion = torch.nn.MSELoss()
    steps = len(train_dataset) * num_epochs  
    warmup_steps = int(steps * 0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineLRScheduler(optimizer, t_initial=steps, warmup_t=warmup_steps, warmup_lr_init=1e-6, lr_min=1e-5)
    
    for epoch in range(num_epochs):
        model.train()
        out_list = []
        target_list = []
        loss_sum = 0
        n = 0
        pbar = tqdm(batch_data, leave=True)
        
        for i, training in enumerate(pbar):
            x, edge_index, y, batch = training.x, training.edge_index, training.y, training.batch
            x, edge_index, y, batch = x.to(device), edge_index.to(device), y.to(device), batch.to(device)
            optimizer.zero_grad()
            out = model(x, edge_index, batch)
            loss = criterion(out.squeeze(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-2)
            scheduler.step(i + len(train_dataset) * epoch)
            optimizer.step()
            out_list.extend(out.cpu().detach().numpy().flatten())
            target_list.extend(y.cpu().detach().numpy().flatten())
            loss_sum += loss.item()
            n += 1
        
        print(f'Running loss: {(loss_sum/n):.2f}, Current loss: {loss.item():.2f}')
        R2 = r2_score(target_list, out_list)
        R2_TRAIN.append(R2)
        print(f'Fold {fold}, Epoch {epoch}: R2 = {R2}')
        pbar.close()
        
        # Validate
        model.eval()
        val_target = []
        val_out = []
        pbar_val = tqdm(batch_data_val, leave=False)
        
        for val in pbar_val:
            x, edge_index, y, batch = val.x, val.edge_index, val.y, val.batch
            x, edge_index, y, batch = x.to(device), edge_index.to(device), y.to(device), batch.to(device)
            out = model(x, edge_index, batch)
            val_out.extend(out.cpu().detach().numpy().flatten())
            val_target.extend(y.cpu().detach().numpy().flatten())
        
        R2_val = r2_score(val_target, val_out)
        R2_VAL.append(R2_val)
        print(f'R2_val: {R2_val}')
        pbar_val.close()
        
        # Save the best model
        if R2_val > best_r2_val:
            best_r2_val = R2_val
            torch.save(model.state_dict(), best_model_path)
            print('################### Model saved ###################')
            
            # Test
            te_target = []
            te_out = []
            pbar_test = tqdm(batch_data_test, leave=False)
            
            for te in pbar_test:
                x, edge_index, y, batch = te.x, te.edge_index, te.y, te.batch
                x, edge_index, y, batch = x.to(device), edge_index.to(device), y.to(device), batch.to(device)
                out_te = model(x, edge_index, batch)
                te_out.extend(out_te.cpu().detach().numpy().flatten())
                te_target.extend(y.cpu().detach().numpy().flatten())
            
            R2_te = r2_score(te_target, te_out)
            print(f'Test R2: {R2_te}')
            pbar_test.close()