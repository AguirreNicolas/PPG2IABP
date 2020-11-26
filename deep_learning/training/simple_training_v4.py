
import time
import torch
import numpy as np

def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs


def train(model, dataloader, optimizer, criterion_signal,criterion_label,weight_label, clip,n_elements,len_y):
  model.train()
  total_epoch_loss = 0
  epoch_loss_signal = 0 
  epoch_loss_label = 0
  correct = 0
  for i, batch in enumerate(dataloader):
    x = batch[0].permute(0,2,1)
    y = batch[1].permute(0,2,1) #(batch,time,features)
    y_mask = batch[2].permute(0,2,1).squeeze()
    optimizer.zero_grad()
    output ,_ = model(x, y,y_mask)

    output_signal = output[:,:,0] #Signal
    output_label = output[:,:,1:5].permute(0,2,1) #segmentation

    y_signal = y[:,:,0]
    y_label = y[:,:,1:5].permute(0,2,1)
    y_label = torch.argmax(y_label, 1)

    loss_signal = criterion_signal(output_signal, y_signal)
    loss_signal = torch.sum((loss_signal*y_mask)) / y_mask.sum()
    
    loss_label = weight_label * criterion_label(output_label, y_label)
    

    loss = loss_signal + loss_label
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    label_predicted = torch.argmax(output_label, 1)
    correct += (label_predicted == y_label).sum().item()

    epoch_loss_signal += loss_signal.item()
    epoch_loss_label += loss_label.item()
    total_epoch_loss += loss_signal.item() + loss_label.item()

  total_epoch_loss = total_epoch_loss/n_elements
  epoch_loss_signal =  epoch_loss_signal/n_elements
  epoch_loss_label = epoch_loss_label/n_elements
  accuracy = (correct/(n_elements*len_y))*100

  return total_epoch_loss, epoch_loss_signal, epoch_loss_label, accuracy

def evaluate(model, dataloader, criterion_signal,criterion_label, weight_label,n_elements,len_y):
  model.eval()
  total_epoch_loss = 0
  epoch_loss_signal = 0 
  epoch_loss_label = 0
  correct = 0
  with torch.no_grad():
    for i, batch in enumerate(dataloader):
      x = batch[0].permute(0,2,1)
      y = batch[1].permute(0,2,1) #(batch,time,features)
      y_mask = batch[2].permute(0,2,1).squeeze()
      output, _ = model(x, y, y_mask, 0) #turn off teacher forcing
      #Segmentation
      output_label = output[:,:,1:5].permute(0,2,1)
      y_label = y[:,:,1:5].permute(0,2,1)
      y_label = torch.argmax(y_label, 1)
      loss_label = weight_label * criterion_label(output_label, y_label)
      label_predicted = torch.argmax(output_label, 1)
      correct += (label_predicted == y_label).sum().item()
      #Signal
      output_signal = output[:,:,0]
      y_signal = y[:,:,0]
      loss_signal = criterion_signal(output_signal, y_signal)
      loss_signal = torch.sum((loss_signal*y_mask)) / y_mask.sum()
      
      #SUM LOSS
      loss = loss_signal + loss_label
      epoch_loss_signal += loss_signal.item()
      epoch_loss_label += loss_label.item()
      total_epoch_loss += loss_signal.item() + loss_label.item()

  total_epoch_loss = total_epoch_loss/n_elements
  epoch_loss_signal =  epoch_loss_signal/n_elements
  epoch_loss_label = epoch_loss_label/n_elements
  accuracy = (correct/(n_elements*len_y))*100
  
  return total_epoch_loss, epoch_loss_signal, epoch_loss_label, accuracy


def predict(model, dataloader,criterion_signal,criterion_label, weight_label,final_len_x,final_len_y):
  model.eval()
  n_elements = len(dataloader.dataset)
  num_batches = len(dataloader)
  batch_size = dataloader.batch_size
  predictions = torch.zeros(n_elements,final_len_y,5)
  attentions = torch.zeros(n_elements,final_len_x,final_len_y)
  input = torch.zeros(n_elements,final_len_x,2)
  total_epoch_loss = 0
  epoch_loss_signal = 0 
  epoch_loss_label = 0
  correct = 0
  with torch.no_grad():
    for i, batch in enumerate(dataloader):
      
        x = batch[0].permute(0,2,1)
        y = batch[1].permute(0,2,1) #(batch,time,features)
        y_mask = batch[2].permute(0,2,1).squeeze()
        #CORRECTO?
        y_mask = torch.ones_like(y_mask)

        start = i*batch_size
        end = start + batch_size
        if i == num_batches - 1:
          end = n_elements
        output, att_weight = model(x, y,y_mask, 0) #turn off teacher forcing

        output_label = output[:,:,1:5].permute(0,2,1)
        y_label = y[:,:,1:5].permute(0,2,1)
        y_label = torch.argmax(y_label, 1)
        loss_label = weight_label * criterion_label(output_label, y_label)
        _, label_predicted = torch.max(output_label.data, 1)
        correct += (label_predicted == y_label).sum().item()

        mask_predicted = label_predicted>0
        mask_predicted_att = mask_predicted.unsqueeze(1).repeat(1,att_weight.size(1),1)

        att_weight = att_weight * mask_predicted_att
        attentions[start:end] = att_weight 
        predictions[start:end] = output
        input[start:end] = x[:,:,:2]

        output_signal = output[:,:,0]
        y_signal = y[:,:,0]

        #SIGNAL
        loss_signal = criterion_signal(output_signal, y_signal)
        loss_signal = torch.sum((loss_signal*mask_predicted)) / mask_predicted.sum()

        loss = loss_signal + loss_label

        epoch_loss_signal += loss_signal.item()
        epoch_loss_label += loss_label.item()
        total_epoch_loss += loss_signal.item() + loss_label.item()
  total_epoch_loss = total_epoch_loss/n_elements
  epoch_loss_signal =  epoch_loss_signal/n_elements
  epoch_loss_label = epoch_loss_label/n_elements
  accuracy = (correct/(n_elements*final_len_y))*100
  return input, predictions,total_epoch_loss, epoch_loss_signal, epoch_loss_label,accuracy,attentions


def load_checkpoint(model,optimizer,scheduler,path,stage='validation'):
  if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
  else:
    map_location='cpu'

  if stage == 'validation':
    history = np.load(path[:-3]+'_history_best.npz',allow_pickle=True)
    checkpoint = torch.load(path,map_location=map_location)
  if stage == 'final':
    history = np.load(path[:-3]+'_history_best_final.npz',allow_pickle=True)
    checkpoint = torch.load(path+'_final',map_location=map_location) 

  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
  epoch = checkpoint['epoch']
  best_valid_loss = checkpoint['loss']
  loss_train_history = history['arr_0'][0][:epoch]
  loss_val_history = history['arr_0'][1][:epoch]
  return (model,optimizer,scheduler,epoch,best_valid_loss,loss_train_history,loss_val_history)

def fit(n_epochs,model,optimizer,scheduler,criterion_signal,criterion_label,weight_label,clip_val,
train_dl,q_train,val_dl,q_val,final_len_y,model_save,save=False,final=False, e_i = 0, history=[]):
  
  if e_i != 0:
    #Train
    loss_train_history = history[0,0]
    loss_signal_train_history = history[0,1]
    loss_label_train_history = history[0,2] 
    accuracy_train_history = history[0,3] 
    #Valid
    loss_valid_history = history[1,0]
    loss_signal_valid_history = history[1,1]
    loss_label_valid_history = history[1,2]
    accuracy_valid_history = history[1,3]
    n_epochs = n_epochs + e_i
    patience = 0
    best_valid_loss = min(loss_valid_history)
  else:
    best_valid_loss = float('inf')


  for epoch in range(e_i,n_epochs):
    start_time = time.time()
    total_train_loss,signal_train_loss,label_train_loss, train_accuracy = train(model, train_dl, optimizer, criterion_signal,criterion_label,weight_label,clip_val,q_train,final_len_y)
    total_valid_loss,signal_valid_loss,label_valid_loss, valid_accuracy = evaluate(model, val_dl, criterion_signal,criterion_label,weight_label,q_val,final_len_y)
    scheduler.step(total_valid_loss)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if epoch == 0:
      #Train
      loss_train_history = total_train_loss
      loss_signal_train_history = signal_train_loss
      loss_label_train_history = label_train_loss
      accuracy_train_history = train_accuracy
      #Valid
      loss_valid_history = total_valid_loss
      loss_signal_valid_history = signal_valid_loss
      loss_label_valid_history = label_valid_loss
      accuracy_valid_history = valid_accuracy
      patience = 0
    else:
      #Train        
      loss_train_history = np.append(loss_train_history, total_train_loss)
      loss_signal_train_history = np.append(loss_signal_train_history, signal_train_loss)
      loss_label_train_history = np.append(loss_label_train_history, label_train_loss)
      accuracy_train_history = np.append(accuracy_train_history, train_accuracy)
      #Train
      loss_valid_history = np.append(loss_valid_history, total_valid_loss)
      loss_signal_valid_history = np.append(loss_signal_valid_history, signal_valid_loss)
      loss_label_valid_history = np.append(loss_label_valid_history, label_valid_loss)
      accuracy_valid_history = np.append(accuracy_valid_history, valid_accuracy)
    
    if total_valid_loss < best_valid_loss:
      best_valid_loss = total_valid_loss
      patience = 0
      if save:
        if final:
          torch.save({'epoch':epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': best_valid_loss,'scheduler_state_dict': scheduler.state_dict()}, model_save+'_final')
          history = np.array([[loss_train_history,loss_signal_train_history,loss_label_train_history,accuracy_train_history],
                            [loss_valid_history,loss_signal_valid_history,loss_label_valid_history,accuracy_valid_history]])
          np.savez(model_save[:-3]+'_history_best_final',history)
        else:
          torch.save({'epoch':epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),
                      'loss': best_valid_loss,'scheduler_state_dict': scheduler.state_dict()}, model_save)
          history = np.array([[loss_train_history,loss_signal_train_history,loss_label_train_history,accuracy_train_history],
                              [loss_valid_history,loss_signal_valid_history,loss_label_valid_history,accuracy_valid_history]])
          np.savez(model_save[:-3]+'_history_best',history)
    else:
      patience += 1
    
    print('Epoch:{}|Patience: {}|Time:{}:{}s|TT_Loss: {:.8f}|TV_Loss: {:.8f}|ST_Loss: {:.8f}|SV_Loss: {:.8f}|LT_Loss: {:.8f}|LV_Loss: {:.8f}|Acc_T: {:.1f}%|Acc_V: {:.1f}%|Min_V_Loss: {:.8f} '.format(
            epoch, patience,epoch_mins,epoch_secs,total_train_loss, total_valid_loss,signal_train_loss,signal_valid_loss,label_train_loss,label_valid_loss,train_accuracy,valid_accuracy, best_valid_loss))
    #EarlyStopping
    if patience > scheduler.patience*2 + scheduler.patience/2:
      history = np.asarray([[loss_train_history,loss_signal_train_history,loss_label_train_history,accuracy_train_history],
                    [loss_valid_history,loss_signal_valid_history,loss_label_valid_history,accuracy_valid_history]])
      return model, history              
  history = np.asarray([[loss_train_history,loss_signal_train_history,loss_label_train_history,accuracy_train_history],
                    [loss_valid_history,loss_signal_valid_history,loss_label_valid_history,accuracy_valid_history]])
  return model, history
