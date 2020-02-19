import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.batch_size = 0


        self.embeddings = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first = True, bidirectional = False)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        
        
      
        

    
    def forward(self, features, captions):
        
        
        captions = captions[:,:-1]
        
        self.batch_size = features.size(0)
      
        
        hidden_state = torch.randn(self.num_layers, self.batch_size, self.hidden_size)
        cell_state = torch.randn(self.num_layers, self.batch_size, self.hidden_size)
        self.hidden = (hidden_state, cell_state)
        
        outputs = torch.empty((self.batch_size, captions.size(1), self.vocab_size))
        
        captions_embed = self.embeddings(captions)
        
        
        inputs = torch.cat((features.unsqueeze(dim=1), captions_embed), dim = 1)
        lstm_out, self.hidden = self.lstm(inputs)
        
        
        out = self.linear(lstm_out)
        
        
        return out
        
        
        
        

        

        
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output = []
        self.batch_size = inputs.size(0)
        hidden_state = torch.randn(self.num_layers, self.batch_size, self.hidden_size).cuda()
        cell_state = torch.randn(self.num_layers, self.batch_size, self.hidden_size).cuda()
        self.hidden = (hidden_state, cell_state)

    
        
        for i in range(max_len):
            outputs, self.hidden = self.lstm(inputs, self.hidden)
      
            outputs = self.linear(outputs.squeeze(1))
            target_index = outputs.max(1)[1]
            output.append(target_index.item())
            inputs = self.embeddings(target_index).unsqueeze(1)
        return output
        