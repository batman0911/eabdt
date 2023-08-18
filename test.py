import time
tol_acc = 0.01
losses = []
losses_val = []
acc_train = []
acc_val = []
Iterations = []
best_loss = torch.tensor(np.inf).to(device)
best_acc = 0
best_weights = None
iter = 0
early = 0
start = time.time()
mini_batch_size = 100000
num_mini_batch = int(X_train.shape[0] / mini_batch_size)
# X_gpu_train_shuffle = X_gpu_train
for epoch in tqdm.tqdm(range(int(epochs)),desc='Training Epochs'):
    
    for bid in range(num_mini_batch):
        idx = np.random.randint(X_train.shape[0], size=mini_batch_size)
        X_train_batch = X_train[idx, :]
        x = torch.tensor(X_train_batch, dtype=torch.float32).to(device)
        # indices = torch.randperm(X_gpu_train.shape[0])[:mini_batch_size]
        # x = X_gpu_train[indices]
        # x = X_gpu_train_shuffle[bid*mini_batch_size:(bid+1)*mini_batch_size, :]
    
        # start = time.time()
        # x = X_gpu_train
        # labels = y_gpu_train[indices]
        y_train_batch = y_train[idx]
        labels = torch.flatten(torch.tensor(y_train_batch).type(torch.float32)).to(device)
        optimizer.zero_grad() # Setting our stored gradients equal to zero
        outputs = model(x)
        loss = criterion(torch.squeeze(outputs), labels) 

        loss.backward() # Computes the gradient of the given tensor w.r.t. the weights/bias

        optimizer.step() # Updates weights and biases with the optimizer (SGD)
        
        # print(f"Train -  Loss: {loss.item()}, batch: {bid}")
    
    iter+=1
    early+=1
    if iter%trigger==0:
        early+=1
        with torch.no_grad():
            # Calculating the loss and accuracy for the val dataset
            correct_val = 0
            total_val = 0
            outputs_val = torch.squeeze(model(X_gpu_val))
            loss_val = criterion(outputs_val, y_gpu_val)
            
            # predicted_val = outputs_val.round()
            total_val += y_gpu_val.size(0)
            correct_val += torch.eq(outputs_val.round(), y_gpu_val).sum()
            accuracy_val = 100 * correct_val/total_val
            losses_val.append(loss_val.item())
            acc_val.append(accuracy_val.item())
            
            # Calculating the loss and accuracy for the train dataset
            # total = 0
            # correct = 0
            # total += y_gpu_train.size(0)
            # # correct += np.sum(torch.squeeze(outputs).round().detach().numpy() == y_gpu_train.detach().numpy())
            # correct += torch.eq(torch.squeeze(outputs).round(), y_gpu_train).sum()
            # accuracy = 100 * correct/total
            # losses.append(loss.item())
            Iterations.append(iter)
            
            print(f"Iteration: {iter}. \nVal - Loss: {loss_val.item()}. Accuracy: {accuracy_val}")
            # print(f"Train -  Loss: {loss.item()}. Accuracy: {accuracy}\n")
            
            if accuracy_val > best_acc:
                print(f'\tBetter accuracy: {accuracy_val}')
                best_acc = accuracy_val
                best_weights = copy.deepcopy(model.state_dict())
            elif (best_acc - accuracy_val) > tol_acc:
                print(f'early stopping')
                break    

            # if (early%early_stopping_round == 0) and (loss_val > torch.min(torch.tensor(losses_val[-4:-1]))):
            #     print(f'Early stopping, loss val: {loss_val}')
            #     break
            
end = time.time()
total_time = round(end - start, 3)