import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
from torch.nn import init
import torch.nn.functional as F


class VisualFeatureExtractor(nn.Module):
    def __init__(self, hidden_size, embed_size):
        super(VisualFeatureExtractor, self).__init__()

        #resnet152 frontal
        resnet_frontal = models.resnet152()
        modules_frontal =  list(resnet_frontal.children())[:-2]
        resnet_conv_frontal = nn.Sequential(*modules_frontal)

        # lateral
        resnet_lateral = models.resnet152()
        modules_lateral = list(resnet_lateral.children())[:-2]
        resnet_conv_lateral = nn.Sequential(*modules_lateral)

        self.avgpool_fun = nn.AvgPool2d( 7 ) # 
        self.resnet_conv_frontal = resnet_conv_frontal
        self.affine_frontal_a = nn.Linear(2048, hidden_size) # v_i = W_a * V
        self.affine_frontal_b = nn.Linear(2048, embed_size)  # v_g = W_b * a^g

        self.resnet_conv_lateral =resnet_conv_lateral
        self.affine_lateral_a = nn.Linear(2048, hidden_size) #  v_i = W_a * V
        self.affine_lateral_b = nn.Linear(2048, embed_size)  # v_g =W_b * a^g

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        init.kaiming_uniform( self.affine_frontal_a.weight, mode='fan_in' )
        self.affine_frontal_a.bias.data.fill_( 0 )
        init.kaiming_uniform( self.affine_frontal_b.weight, mode='fan_in')
        self.affine_frontal_b.bias.data.fill_( 0 )

        init.kaiming_uniform( self.affine_lateral_a.weight, mode='fan_in')
        self.affine_lateral_a.bias.data.fill_( 0 )
        init.kaiming_uniform( self.affine_lateral_b.weight, mode='fan_in')
        self.affine_lateral_b.bias.data.fill_( 0 )

    def forward(self, image_frontal, image_lateral):
        """
        inputs: image_frontal, image_lateral
        outputs: V_frontal, v_g_frontal,  V_lateral, v_g_lateral
        """
    
        A_frontal = self.resnet_conv_frontal( image_frontal ) # batch_size x 2048x7x7
        # a^g
        a_g_frontal = self.avgpool_fun( A_frontal ).squeeze()  # batch_size x 2048
        #V=[v1, ... v49]
        V_frontal = A_frontal.view(A_frontal.size(0), A_frontal.size(1), -1).transpose(1, 2)
        V_frontal = self.relu( self.affine_frontal_a( self.dropout( V_frontal )))
        v_g_frontal = self.relu( self.affine_frontal_b( self.dropout( a_g_frontal)))

        
        A_lateral = self.resnet_conv_lateral( image_lateral ) # batch_size x 2048 x7 x7
        #a^g
        a_g_lateral =self.avgpool_fun( A_lateral ).squeeze()
        V_lateral = A_lateral.view( A_lateral.size(0), A_lateral.size(1), -1).transpose(1, 2)
        V_lateral = self.relu( self.affine_lateral_a( self.dropout( V_lateral )))
        v_g_lateral = self.relu( self.affine_lateral_b( self.dropout( a_g_lateral )))

        return V_frontal, v_g_frontal, V_lateral, v_g_lateral


class SentenceLSTM(nn.Module):
    def __init__(self,
                #version = 'v1',
                embed_size = 512,
                hidden_size = 512,
                num_spatial = 49, # input_image_size=(224,224)
                num_layers = 1,
                #dropout = 0.3,
                momentum = 0.1):
        super(SentenceLSTM, self).__init__()


        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.affine_y = nn.Linear(embed_size*2, hidden_size, bias=True)
        self.W_stop = nn.Linear(in_features = hidden_size,
                                out_features = 2, bias=True)
        self.init_wordh = nn.Linear(hidden_size, hidden_size, bias=True)
        self.init_wordc = nn.Linear(hidden_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.5)
        self.__init_weights()

    def __init_weights(self):
        self.W_stop.weight.data.uniform_(-0.1, 0.1)
        self.W_stop.bias.data.fill_(0)
        self.init_wordh.weight.data.uniform_(-0.1, 0.1)
        self.init_wordh.bias.data.fill_(0)
        self.init_wordc.weight.data.uniform_(-0.1, 0.1)
        self.init_wordc.bias.data.fill_(0)
        self.affine_y.weight.data.uniform_(-0.1, 0.1)
        self.affine_y.bias.data.fill_(0)



    def forward(self, avg_cat_features, state):
        """
        topic is transformed by two fully connection layers
        """
        avg_cat_features = avg_cat_features.unsqueeze(1)
        input_y = self.affine_y(avg_cat_features)
        output, state_t = self.lstm(input_y, state)
        p_stop = self.W_stop(output)
        h0_word = self.tanh(self.init_wordh( self.dropout(output) )).transpose(0, 1)
        c0_word = self.tanh(self.init_wordc( self.dropout(output) )).transpose(0, 1)

        return h0_word, c0_word, p_stop, state_t



class WordLSTM(nn.Module):
    def __init__(self,
                 embed_size,
                 hidden_size,
                 vocab_size,
                 num_layers,
                 num_spatial,
                 n_max=50):
        super(WordLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.__init_weights()
        self.n_max = n_max
        self.vocab_size = vocab_size

    def __init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    
    def forward(self, captions, states):
        """
        state is initialized from topic_vec
        """
        embeddings = self.embed(captions).unsqueeze(1)
        hidden, states_t = self.lstm(embeddings, states)
        outputs = self.linear(hidden[:, -1, :])
        return outputs, states_t
 

    def sample(self, start_tokens, states):
        sampled_ids = np.zeros((states[0].shape[1], self.n_max))
        sampled_ids = _to_var(torch.Tensor(sampled_ids).long(), requires_grad=False)
        sampled_ids[:, 0] = start_tokens
        predicted = start_tokens

        

        for i in range(1, self.n_max):
            outputs, states_t = self.forward(predicted, states)
            states = states_t
            predicted = torch.max(outputs, 1)[1] # argmax predicted.shape=(batch_size, 1)
            sampled_ids[:, i] = predicted
            
        return sampled_ids

def _to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)