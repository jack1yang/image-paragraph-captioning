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
                embed_size = 512,
                hidden_size = 512,
                num_spatial = 49, # input_image_size=(224,224)
                num_layers = 1,
                #dropout = 0.3,
                momentum = 0.1):
        super(SentenceLSTM, self).__init__()



        self.lstm_frontal = nn.LSTM(input_size = embed_size, # visual features = embed_size
                            hidden_size = hidden_size,
                            batch_first = True)

        self.lstm_lateral = nn.LSTM(input_size = embed_size, # visual features = embed_size
                            #embed_size = embed_size,
                            hidden_size = hidden_size,
                            num_layers= num_layers,
                            #dropout = dropout,
                            batch_first = True)        
        # a = W_v_att * tanh(W_v*V + W_v,h*Ht), V is the visual spatial features
        # a = softmax(a)                      , Ht is the hidden state of sentencelstm
        # vt = V * a      , vt is the time step t of sentencelstm, and regards as topic
        self.W_v_att = nn.Linear(in_features = num_spatial,           #W_v_att
                                 out_features=1, bias = True)
                                 
        self.W_v_frontal = nn.Linear(in_features = embed_size,                #
                             out_features = num_spatial, bias = True)
        self.W_v_lateral = nn.Linear(in_features = embed_size,                #
                             out_features = num_spatial, bias = True)
        self.W_v_h_frontal = nn.Linear(in_features = hidden_size,
                               out_features = num_spatial, bias =True)
        self.W_v_h_lateral = nn.Linear(in_features = hidden_size,
                               out_features = num_spatial, bias =True)                               
        #reduce the visual spatial features dimension
        #self.affine_b = nn.Linear(in_features = hidden_size,
        #                          out_features = embed_size, bias = True)
        self.W_stop = nn.Linear(in_features = hidden_size*2,
                                out_features = 2, bias=True)

        self.init_wordh = nn.Linear(embed_size*2, hidden_size, bias=True)
        self.init_wordc = nn.Linear(embed_size*2, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.5)
        self.__init_weights()

    def __init_weights(self):
        self.W_v_att.weight.data.uniform_(-0.1, 0.1)
        self.W_v_att.bias.data.fill_(0)    

        self.W_v_frontal.weight.data.uniform_(-0.1, 0.1)
        self.W_v_frontal.bias.data.fill_(0)
        self.W_v_lateral.weight.data.uniform_(-0.1, 0.1)
        self.W_v_lateral.bias.data.fill_(0)

        self.W_v_h_frontal.weight.data.uniform_(-0.1, 0.1)
        self.W_v_h_frontal.bias.data.fill_(0)
        self.W_v_h_lateral.weight.data.uniform_(-0.1, 0.1)
        self.W_v_h_lateral.bias.data.fill_(0)        
        #self.affine_b.weight.data.uniform_(-0.1, 0.1)
        #self.affine_b.bias.data.fill_(0)
        self.W_stop.weight.data.uniform_(-0.1, 0.1)
        self.W_stop.bias.data.fill_(0)

        self.init_wordh.weight.data.uniform_(-0.1, 0.1)
        self.init_wordh.bias.data.fill_(0)
        self.init_wordc.weight.data.uniform_(-0.1, 0.1)
        self.init_wordc.bias.data.fill_(0)


    def forward(self, v_input_frontal, v_input_lateral, V_frontal, V_lateral, last_frontal_state=None, last_lateral_state=None):
        """
        v_t: (batch_size, embed_size)  the input of sentencelstm at timestep t
        visual_features: (batch_size, num_spatial, embed_size) the last conv features of resnet
        h_t: (batch_size, hidden_size) the hidden state of sentencelstm at timestep t
        """

        # transform the viusal_features' size into embed_size
        # V.shape = (batch_size, 49, hidden_size)
        # V = self.relu(self.affine_b( self.dropout( visual_features )))

        # V_frontal = V_lateral = (batch_size, 49, hidden_size)
        # frontal 
        v_input_frontal = v_input_frontal.unsqueeze(1) # shape=(batch_size, 1, embed_size)
        h_t_frontal, state_frontal = self.lstm_frontal(v_input_frontal, last_frontal_state)
        # W_v * V + W_v,h * h_t * 1^T
        content_v_input_frontal = self.W_v_frontal(self.dropout( V_frontal )).unsqueeze(1) + \
                        self.W_v_h_frontal( self.dropout(h_t_frontal)).unsqueeze(2)

        # visual_t = W_v_att * tanh( content_v_input )
        visual_t_input_frontal = self.W_v_att(self.dropout(self.tanh(content_v_input_frontal))).squeeze(3)
        alpha_t_input_frontal = self.softmax(visual_t_input_frontal.view(-1, visual_t_input_frontal.size(2))).\
                    view(visual_t_input_frontal.size(0), visual_t_input_frontal.size(1), -1)
        # z_t_input.shape = (batch_size, 1, embed_size)
        z_t_input_frontal = torch.bmm(alpha_t_input_frontal, V_frontal).squeeze(1)



        # lateral 
        v_input_lateral = v_input_lateral.unsqueeze(1)
        h_t_lateral, state_lateral = self.lstm_lateral(v_input_lateral, last_lateral_state)
        # W_v * V + W_v,h * h_t * 1^T
        content_v_input_lateral = self.W_v_lateral(self.dropout( V_lateral )).unsqueeze(1) + \
                        self.W_v_h_lateral( self.dropout(h_t_lateral)).unsqueeze(2)

        # visual_t = W_v_att * tanh( content_v_input )
        visual_t_input_lateral = self.W_v_att(self.dropout(self.tanh(content_v_input_lateral))).squeeze(3)
        alpha_t_input_lateral = self.softmax(visual_t_input_lateral.view(-1, visual_t_input_lateral.size(2))).\
                    view(visual_t_input_lateral.size(0), visual_t_input_lateral.size(1), -1)
        # z_t_input.shape = (batch_size, 1, embed_size)
        z_t_input_lateral = torch.bmm(alpha_t_input_lateral, V_lateral).squeeze(1)



        # v_input = v_input.unsqueeze(1) # shape = (batch_size, 1, embed_size)
        # h_t, states_t = self.lstm(v_input, states) # h_t.shape = (batch_size, 1, hidden_size)
        # # content_v_input = (W_v*V + W_v,h*Ht)
        # # visual_t_input = W_v_att * tanh(content_v_input)
        # # a = softmax(visual_t_input)

        # # W_v * V + W_v,h * h_t * 1^T
        # content_v_input = self.W_v(self.dropout( V )).unsqueeze(1) + self.W_v_h(self.dropout( h_t )).unsqueeze(2)
        
        # # visual_t = W_v_att * tanh( content_v_input )
        # visual_t_input = self.W_v_att(self.dropout(self.tanh(content_v))).squeeze(3)
        # alpha_t_input = self.softmax(visual_t_input.view(-1, visual_t_input.size(2))).view(visual_t_input.size(0), visual_t_input.size(1), -1)
        # z_t_input = torch.bmm(alpha_t_input, visual_features).squeeze(2)
        # # z_t_input.shape = (batch_size, 1, embed_size)


        p_stop = self.W_stop(torch.cat((h_t_frontal, h_t_lateral),axis=2))

        v_t = torch.cat((v_input_frontal, v_input_lateral), axis=2) #batch_size x embed_size*2
        h0_word = self.tanh(self.init_wordh( self.dropout(v_t) )).transpose(0, 1)
        c0_word = self.tanh(self.init_wordc( self.dropout(v_t) )).transpose(0, 1)
        return h0_word, c0_word, z_t_input_frontal, z_t_input_lateral, p_stop, state_frontal, state_lateral



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
        #v_t = v_t.unsqueeze(1)
        #embeddings = torch.cat((v_t, embeddings), 2)
        #inputx = self.affine_x(embeddings)
        hidden, states_t = self.lstm(embeddings, states)
        outputs = self.linear(hidden[:, -1, :])
        return outputs, states_t
 
  
    def sample(self, start_tokens, states):
            sampled_ids = np.zeros((np.shape(states[0])[1], self.n_max))
            sampled_ids = _to_var(torch.Tensor(sampled_ids).long(), requires_grad=False)
            sampled_ids[:, 0] = start_tokens
            predicted = start_tokens

            #states = None
            for i in range(1, self.n_max):
    
                outputs, state_t = self.forward(predicted, states)
                states = state_t

                predicted = torch.max(outputs, 1)[1] # argmax predicted.shape=(batch_size, 1)
                sampled_ids[:, i] = predicted

            return sampled_ids        


def _to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)