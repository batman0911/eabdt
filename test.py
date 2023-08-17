losses = []
losses_val = []
Iterations = []
iter = 0
for epoch in tqdm.tqdm(range(int(epochs)),desc='Training Epochs'):
    x = X_gpu_train
    labels = y_gpu_train
    optimizer.zero_grad() # Setting our stored gradients equal to zero
    outputs = model(X_gpu_train)
    loss = criterion(torch.squeeze(outputs), labels) 

    loss.backward() # Computes the gradient of the given tensor w.r.t. the weights/bias

    optimizer.step() # Updates weights and biases with the optimizer (SGD)

    iter+=1
    if iter%100==0:
        with torch.no_grad():
            # Calculating the loss and accuracy for the val dataset
            correct_val = 0
            total_val = 0
            outputs_val = torch.squeeze(model(X_gpu_val))
            loss_val = criterion(outputs_val, y_gpu_val)
            
            predicted_val = outputs_val.round().detach().numpy()
            total_val += y_gpu_val.size(0)
            correct_val += np.sum(predicted_val == y_gpu_val.detach().numpy())
            accuracy_val = 100 * correct_val/total_val
            losses_val.append(loss_val.item())
            
            # Calculating the loss and accuracy for the train dataset
            total = 0
            correct = 0
            total += y_gpu_train.size(0)
            correct += np.sum(torch.squeeze(outputs).round().detach().numpy() == y_gpu_train.detach().numpy())
            accuracy = 100 * correct/total
            losses.append(loss.item())
            Iterations.append(iter)
            
            print(f"Iteration: {iter}. \nval - Loss: {loss_val.item()}. Accuracy: {accuracy_val}")
            print(f"Train -  Loss: {loss.item()}. Accuracy: {accuracy}\n")
            
            
            
            
            
losses = []
losses_val = []
Iterations = []
iter = 0
for epoch in tqdm.tqdm(range(int(epochs)),desc='Training Epochs'):
    x = X_gpu_train
    labels = y_gpu_train
    optimizer.zero_grad() # Setting our stored gradients equal to zero
    outputs = model(X_gpu_train)
    loss = criterion(torch.squeeze(outputs), labels) 

    loss.backward() # Computes the gradient of the given tensor w.r.t. the weights/bias

    optimizer.step() # Updates weights and biases with the optimizer (SGD)

    iter+=1
    if iter%trigger==0:
        with torch.no_grad():
            # Calculating the loss and accuracy for the val dataset
            correct_val = 0
            total_val = 0
            outputs_val = torch.squeeze(model(X_gpu_val))
            loss_val = criterion(outputs_val, y_gpu_val)
            
            total_val += y_gpu_val.size(0)
            correct_val += torch.eq(outputs_val.round(), y_gpu_val).sum()
            accuracy_val = 100 * correct_val/total_val
            losses_val.append(loss_val.item())
            
            # Calculating the loss and accuracy for the train dataset
            total = 0
            correct = 0
            total += y_gpu_train.size(0)
            # correct += np.sum(torch.squeeze(outputs).round().detach().numpy() == y_gpu_train.detach().numpy())
            correct += torch.eq(torch.squeeze(outputs).round(), y_gpu_train).sum()
            accuracy = 100 * correct/total
            losses.append(loss.item())
            Iterations.append(iter)
            
            print(f"Iteration: {iter}. \nval - Loss: {loss_val.item()}. Accuracy: {accuracy_val}")
            print(f"Train -  Loss: {loss.item()}. Accuracy: {accuracy}\n")