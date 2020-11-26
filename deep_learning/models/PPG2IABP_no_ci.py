import torch
import torch.nn.functional as F
import torch.nn as nn
import random

class Encoder(nn.Module):
  def __init__(self, input_size_encoder,hidden_size,bidirectional):
    super(Encoder, self).__init__()
    self.hidden_size = hidden_size
    self.input_size_encoder = input_size_encoder
    self.bidirectional = bidirectional

    # input Layer N (iLN) | output Layer N (oLN)
    self.iL1= self.input_size_encoder
    self.oL1= self.hidden_size
    self.iL2= self.iL1 + self.oL1*2 #(Bidirectional)
    self.oL2= self.iL2*2
    self.iL3= self.oL2*2 + self.oL1*2 + self.iL1   #(Bidirectional)
    self.oL3= self.iL3*2

    self.gru1 = nn.GRU(self.iL1,self.oL1, batch_first=True, bidirectional= self.bidirectional)
    self.gru2 = nn.GRU(self.iL2,self.oL2, batch_first=True, bidirectional = self.bidirectional)
    self.gru3 = nn.GRU(self.iL3,self.oL3, batch_first=True, bidirectional = self.bidirectional)

    self.rnn_layers = [self.gru1,self.gru2,self.gru3]
    #INIT RESET GATES
    for rrn_i in self.rnn_layers:
      
      for names in rrn_i._all_weights:
        for name in filter(lambda n: "bias_ih_l" in n,  names):
            bias = getattr(rrn_i, name)
            n = bias.size(0)
            start, end = 0, n//3
            bias.data[start:end].fill_(0.)
          # orthogonal initialization of recurrent weights
      for _, hh, _, _ in rrn_i.all_weights:
        print(rrn_i.hidden_size)
        for i in range(0, hh.size(0), rrn_i.hidden_size):
          nn.init.orthogonal_(hh[i:i + rrn_i.hidden_size], gain=1)
    
  def forward(self, x):
    output1, h_n1 = self.gru1(x)
    #concatenate x to fw & bw (out1)
    output1_residual = torch.cat((x,output1),dim=2)
    output2, h_n2 = self.gru2(output1_residual)
    #concatenate x&out1 to fw & bw (out2)
    output2_residual = torch.cat((x,output1,output2),dim=2)
    output3, h_n3 = self.gru3(output2_residual)
    return output3,[h_n1,h_n2,h_n3]


class LuongAttention(nn.Module):
  def __init__(self, hidden_size, method="general"):
    super(LuongAttention, self).__init__()
    self.method = method
    self.hidden_size = hidden_size
    # Defining the layers/weights required depending on alignment scoring method
    if method == "general":
      self.W = nn.Linear(hidden_size, hidden_size, bias=False) # una de las dos tiene que corresponder a las dimensiones del encoder y otra al de decoder ! para que dsp puedan multiplicarse!
    elif method == "concat":
      self.W = nn.Linear(hidden_size*2, hidden_size, bias=False) # Dimension del encoder + dimension del decoder
      #self.V = nn.Parameter(torch.FloatTensor(1, hidden_size))
      self.V = nn.Linear(hidden_size,1,bias=False)
      
  def forward(self, decoder_hidden, encoder_outputs):
    if self.method == "dot":
      # For the dot scoring method, no weights or linear layers are involved
      score = encoder_outputs.bmm(decoder_hidden.view(1,-1,1)).squeeze(-1)
      return score

    elif self.method == "general":
      # For general scoring, decoder hidden state is passed through linear layers to introduce a weight matrix
      score = self.W(decoder_hidden)
      score = encoder_outputs.bmm(score.permute(0,2,1))
      return score

    elif self.method == "concat":
      # For concat scoring, decoder hidden state and encoder outputs are concatenated first
      #decoder_hidden = decoder_hidden.repeat(1,encoder_outputs.size()[1],1) # SLOW!
      #decoder_hidden = decoder_hidden.expand(1,encoder_outputs.size()[1],1)
      score = torch.cat((decoder_hidden.expand(-1,encoder_outputs.size()[1],-1), encoder_outputs), 2)
      score = torch.cat((decoder_hidden, encoder_outputs), 2)
      score = torch.tanh(self.W(score))
      score = self.V(score)
      return score


class Decoder(nn.Module):
  def __init__(self, input_size_decoder,input_size_encoder,hidden_size,
               output_size_signal,output_size_segment,
               bidirectional_encoder,do,method='concat'):
    super(Decoder, self).__init__()
    self.hidden_size = hidden_size
    self.input_size_decoder = input_size_decoder
    self.input_size_encoder = input_size_encoder 
    self.output_size_signal = output_size_signal
    self.output_size_segment = output_size_segment
    self.output_size = output_size_signal + output_size_segment
    self.bidirectional_encoder = bidirectional_encoder

    # input Layer N (iLN) | output Layer N (oLN)
    self.eiL1= self.input_size_encoder
    self.eoL1= self.hidden_size
    self.eiL2= self.eiL1 + self.eoL1*2 
    self.eoL2= self.eiL2*2
    self.eiL3= self.eoL2*2 + self.eoL1*2 + self.eiL1   
    self.eoL3= self.eiL3*2
    
    self.iL1 = self.input_size_decoder
    self.oL1 = self.eoL1*2
    self.iL2 = self.oL1 + self.iL1 
    self.oL2 = self.eoL2*2
    self.iL3 = self.oL2 + self.oL1 + self.iL1
    self.oL3 = self.eoL3*2
    (print(f'Decoder Output GRU:{self.oL3}'))
    
    self.gru1 = nn.GRU(self.iL1, self.oL1, batch_first=True, bidirectional = False)
    self.gru2 = nn.GRU(self.iL2, self.oL2, batch_first=True, bidirectional = False)
    self.gru3 = nn.GRU(self.iL3, self.oL3, batch_first=True, bidirectional = False)

    self.rnn_layers = [self.gru1,self.gru2,self.gru3]
    #INIT RESET GATES
    for rrn_i in self.rnn_layers:
      for names in rrn_i._all_weights:
        for name in filter(lambda n: "bias_ih_l" in n,  names):
            bias = getattr(rrn_i, name)
            n = bias.size(0)
            start, end = 0, n//3
            bias.data[start:end].fill_(0.)
          # orthogonal initialization of recurrent weights
      for _, hh, _, _ in rrn_i.all_weights:
        print(rrn_i.hidden_size)
        for i in range(0, hh.size(0), rrn_i.hidden_size):
            nn.init.orthogonal_(hh[i:i + rrn_i.hidden_size], gain=1)

    self.attention = LuongAttention(self.oL3,method)

    self.fc_label = nn.Linear(self.oL3,self.output_size_segment)
    self.do_label = nn.Dropout(do)   
    self.fc_signal =nn.Linear(self.oL3*2,self.output_size_signal)
    self.do_signal = nn.Dropout(do)
    

  def forward(self, input, h_n, enc_outputs,mask):
    output1, h_n1 = self.gru1(input,h_n[0])
    output1_res = torch.cat((input,output1),dim=2)
    output2, h_n2 = self.gru2(output1_res,h_n[1])
    output2_res = torch.cat((input,output1,output2),dim=2)
    output3, h_n3 = self.gru3(output2_res,h_n[2])

    #ATTENTION
    score = self.attention(output3,enc_outputs)
    att_weight = F.softmax(score,dim=1)
    #EXPAN TO DIMENTIONS OF ATT_WEIGHT
    mask_att = mask.unsqueeze(1).unsqueeze(1).expand(-1,att_weight.size(1),-1)
    att_weight = mask_att * att_weight

    context_vector = att_weight * enc_outputs
    context_vector = context_vector.sum(1)
    context_vector = context_vector.unsqueeze(1)
    
    output_signal = torch.cat((output3,context_vector),dim=-1)

    #Outputs
    #Labels
    out_label = F.log_softmax(self.do_label(self.fc_label(output3)),dim=2)
    #Signal
    output_signal = F.elu(self.do_signal(self.fc_signal(output_signal)))
    
    #Concat Outputs
    output = torch.cat((output_signal,out_label),dim=2)
    
    return output, [h_n1, h_n2, h_n3], att_weight


class Modelo(nn.Module):
  def __init__(self, encoder,decoder,device):
    super(Modelo, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.device = device
    self.max_len_output = 200

  def forward(self, input_tensor, target_tensor,mask,teacher_forcing_ratio = 0.01):
    batch_size = input_tensor.shape[0] 
    output_size = self.decoder.output_size
    # ENCODER
    enc_output, enc_h_n = self.encoder(input_tensor)
    # DECODER
    decoder_hidden = []
    if self.decoder.bidirectional_encoder:
      for i in range(len(enc_h_n)):
        enc_h_n_i = enc_h_n[i].view(1,2,batch_size,-1)
        enc_fw = enc_h_n_i[:,0,:,:]
        enc_bw = enc_h_n_i[:,1,:,:]
        enc_h_n_i = torch.cat((enc_fw,enc_bw),dim=2)
        decoder_hidden.append(enc_h_n_i)

    input_len = input_tensor.shape[1]
    if target_tensor is None:
      inference = True
      target_len = self.max_len_output
      decoder_input = torch.ones((1,1,output_size)).to(self.device)
      outputs = torch.ones((1, target_len,output_size)).to(self.device)
      attentions = torch.zeros(1,input_len,target_len).to(self.device)

    else:
      inference = False
      target_len = target_tensor.shape[1]
      decoder_input = torch.ones((batch_size,1,output_size)).to(self.device)
      outputs = torch.zeros(batch_size,target_len,output_size).to(self.device)
      attentions = torch.zeros(batch_size,input_len,target_len).to(self.device) 
    
    for t in range(0, target_len):
      decoder_output, decoder_hidden, att_weight = self.decoder(decoder_input, decoder_hidden,enc_output,mask[:,t])
      outputs[:,t:t+1,:] = decoder_output
      attentions[:,:,t:t+1] = att_weight
      teacher_force = random.random() < teacher_forcing_ratio
      if teacher_force:
        decoder_input = target_tensor[:,t:t+1]
      else:
        decoder_output_label = decoder_output[:,:,1:self.decoder.output_size_segment+1]
        topv, topi = decoder_output_label.topk(1)
        decoder_output_label = F.one_hot(topi,num_classes=self.decoder.output_size_segment).squeeze(1).detach()
    
        decoder_input = torch.cat((decoder_output[:,:,0:1].detach(), #Signal
                                   decoder_output_label), dim=2) # Label
      if inference and topi == 0:
                return outputs[:,:t,:], attentions[:,:,:t]
    return outputs , attentions