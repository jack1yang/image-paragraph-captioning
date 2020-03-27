import time
import pickle
import argparse
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from pycocoevalcap.eval import calculate_metrics
from utils.pure_hier_cnn_model import *
from utils.dataset import *
#from utils.loss import *
from utils.build_tag import *


class CaptionSampler(object):
    def __init__(self, args):
        self.args = args

        self.vocab = self.__init_vocab()
        self.tagger = self.__init_tagger()
        self.transform = self.__init_transform()
        self.data_loader = self.__init_data_loader(self.args.file_lits)
        self.model_state_dict = self.__load_mode_state_dict()

        self.extractor = self.__init_visual_extractor()

        self.sentence_model = self.__init_sentence_model()
        self.word_model = self.__init_word_word()

        self.ce_criterion = self._init_ce_criterion()
        self.mse_criterion = self._init_mse_criterion()

    @staticmethod
    def _init_ce_criterion():
        return nn.CrossEntropyLoss(size_average=False, reduce=False)

    @staticmethod
    def _init_mse_criterion():
        return nn.MSELoss()

    def test(self):
        tag_loss, stop_loss, word_loss, loss = 0, 0, 0, 0


        for i, (images, _, label, captions, prob) in enumerate(self.data_loader):
            batch_tag_loss, batch_stop_loss, batch_word_loss, batch_loss = 0, 0, 0, 0
            images = self.__to_var(images, requires_grad=False)

            visual_features, avg_features = self.extractor.forward(images)
            tags, semantic_features = self.mlc.forward(avg_features)

            batch_tag_loss = self.mse_criterion(tags, self.__to_var(label, requires_grad=False)).sum()

            sentence_states = None
            prev_hidden_states = self.__to_var(torch.zeros(images.shape[0], 1, self.args.hidden_size))

            context = self.__to_var(torch.Tensor(captions).long(), requires_grad=False)
            prob_real = self.__to_var(torch.Tensor(prob).long(), requires_grad=False)

            for sentence_index in range(captions.shape[1]):
                ctx, v_att, a_att = self.co_attention.forward(avg_features,
                                                       semantic_features,
                                                       prev_hidden_states)

                topic, p_stop, hidden_states, sentence_states = self.sentence_model.forward(ctx,
                                                                                            prev_hidden_states,
                                                                                            sentence_states)

                batch_stop_loss += self.ce_criterion(p_stop.squeeze(), prob_real[:, sentence_index]).sum()

                for word_index in range(1, captions.shape[2]):
                    words = self.word_model.forward(topic, context[:, sentence_index, :word_index])
                    word_mask = (context[:, sentence_index, word_index] > 0).float()
                    batch_word_loss += (self.ce_criterion(words, context[:, sentence_index, word_index])
                                        * word_mask).sum()

            batch_loss = self.args.lambda_tag * batch_tag_loss \
                         + self.args.lambda_stop * batch_stop_loss \
                         + self.args.lambda_word * batch_word_loss

            tag_loss += self.args.lambda_tag * batch_tag_loss.data
            stop_loss += self.args.lambda_stop * batch_stop_loss.data
            word_loss += self.args.lambda_word * batch_word_loss.data
            loss += batch_loss.data

        return tag_loss, stop_loss, word_loss, loss

    def generate(self):
        self.extractor.eval()

        self.sentence_model.eval()
        self.word_model.eval()

        progress_bar = tqdm(self.data_loader, desc='Generating')
        results = {}


        for frontal_images, lateral_images, image_names, captions, probs in progress_bar:
            #images = self.__to_var(images, requires_grad=False)
            frontal_images = self.__to_var(frontal_images, requires_grad=False)
            lateral_images = self.__to_var(lateral_images, requires_grad=False)

            #visual_features, avg_features = self.extractor.forward(images)
            V_frontal, v_g_frontal, V_lateral, v_g_lateral = self.extractor.forward(frontal_images, lateral_images)
            to_var(torch.zeros(images.shape[0], 1, self.args.hidden_size))
            frontal_input = v_g_frontal # (batch_size, embed_size)
            lateral_input = v_g_lateral
            last_sent_states_frontal = None
            last_sent_states_lateral = None

            pred_sentences = {}
            real_sentences = {}
            for image_name in image_names:
                image_prefix = image_name[0].split('_')[0]
                pred_sentences[image_prefix] = {}
                real_sentences[image_prefix] = {}

            for i in range(self.args.s_max):
                                                                        sentence_states)
                v_input_frontal, v_input_lateral, p_stop, sent_states_frontal, sent_states_lateral = \
                                        self.sentence_model.forward(frontal_input,
                                                                    lateral_input,
                                                                    V_frontal,
                                                                    V_lateral,
                                                                    last_sent_states_frontal,
                                                                    last_sent_states_lateral)     

                frontal_input = v_input_frontal
                lateral_input = v_input_lateral
                last_sent_states_frontal = sent_states_frontal
                last_sent_states_lateral = sent_states_lateral

                # define whether continue or stop
                p_stop = p_stop.squeeze(1)
                p_stop = torch.max(p_stop, 1)[1].unsqueeze(1)

                # topic
                v_t = torch.cat((frontal_input, lateral_input), axis=1) #batch_size x embed_size*2
                avg_features = torch.cat((v_g_frontal, v_g_lateral), axis=1) #batch_size x embed_size*2


                start_tokens = np.zeros((v_t.shape[0]))
                start_tokens[:] = self.vocab('<start>')
                start_tokens = self.__to_var(torch.Tensor(start_tokens).long(), requires_grad=False)

                #sampled_ids = self.word_model.sample(topic, start_tokens)
                sampled_ids = self.word_model.sample(v_t, avg_features, start_tokens, self.args)#, captions[:,i,:])
                #prev_hidden_states = hidden_state
                sampled_ids = sampled_ids * p_stop

                # self._generate_cam(image_id, visual_features, alpha_v, i)

                #for id, array in zip(image_id, sampled_ids):
                for id, array in zip(image_names, sampled_ids):
                    image_prefix = id[0].split('_')[0]
                    pred_sentences[image_prefix][i] = self.__vec2sent(array.cpu().detach().numpy())

            #for id, array in zip(image_id, captions):
            for id, array in zip(image_names, captions):
                image_prefix = id[0].split('_')[0]
                for i, sent in enumerate(array):
                    real_sentences[image_prefix][i] = self.__vec2sent(sent[1:])

            #for id, pred_tag, real_tag in zip(image_id, tags, label):
            
            for image_name in image_names:
                id = image_name[0].split('_')[0]
                results[id] = {
                    #'Real Tags': self.tagger.inv_tags2array(real_tag),
                    #'Pred Tags': self.tagger.array2tags(torch.topk(pred_tag, self.args.k)[1].cpu().detach().numpy()),
                    'Pred Sent': pred_sentences[id],
                    'Real Sent': real_sentences[id]
                }

        self.__save_json(results)

    def generate_pure_hier_cnn(self):
        self.extractor.eval()
        # self.mlc.eval()
        # self.co_attention.eval()
        self.sentence_model.eval()
        self.word_model.eval()

        progress_bar = tqdm(self.data_loader, desc='Generating')
        results = {}


        for frontal_images, lateral_images, image_names, captions, probs in progress_bar:

            frontal_images = self.__to_var(frontal_images, requires_grad=False)
            lateral_images = self.__to_var(lateral_images, requires_grad=False)


            V_frontal, v_g_frontal, V_lateral, v_g_lateral = self.extractor.forward(frontal_images, lateral_images)

            
            avg_features = torch.cat((v_g_frontal, v_g_lateral), axis=1)
            states = None

            pred_sentences = {}
            real_sentences = {}
            for image_name in image_names:
                image_prefix = image_name[0].split('_')[0]
                pred_sentences[image_prefix] = {}
                real_sentences[image_prefix] = {}

            for i in range(self.args.s_max):
                
                topic, p_stop, state_t = self.sentence_model.forward(avg_features, states)      
                states = state_t
                
                # define whether continue or stop
                p_stop = p_stop.squeeze(1)
                p_stop = torch.max(p_stop, 1)[1].unsqueeze(1)

                # topic

                # start_tokens[:, 0] = self.vocab('<start>')
                start_tokens = np.zeros((topic.shape[0]))
                start_tokens[:] = self.vocab('<start>')
                start_tokens = self.__to_var(torch.Tensor(start_tokens).long(), requires_grad=False)

                #sampled_ids = self.word_model.sample(topic, start_tokens)
                sampled_ids = self.word_model.sample(topic, start_tokens)#, captions[:,i,:])
                #prev_hidden_states = hidden_state
                sampled_ids = sampled_ids * p_stop


                #for id, array in zip(image_id, sampled_ids):
                for id, array in zip(image_names, sampled_ids):
                    image_prefix = id[0].split('_')[0]
                    pred_sentences[image_prefix][i] = self.__vec2sent(array.cpu().detach().numpy())


            for id, array in zip(image_names, captions):
                image_prefix = id[0].split('_')[0]
                for i, sent in enumerate(array):
                    real_sentences[image_prefix][i] = self.__vec2sent(sent[1:])


            
            for image_name in image_names:
                id = image_name[0].split('_')[0]
                results[id] = {
                    #'Real Tags': self.tagger.inv_tags2array(real_tag),
                    #'Pred Tags': self.tagger.array2tags(torch.topk(pred_tag, self.args.k)[1].cpu().detach().numpy()),
                    'Pred Sent': pred_sentences[id],
                    'Real Sent': real_sentences[id]
                }
        
        datasetGTS = {'annotations': []}
        datasetRES = {'annotations': []}

        for i, image_id in enumerate(results):
            array = []
            for each in results[image_id]['Pred Sent']:
                array.append(results[image_id]['Pred Sent'][each])
            pred_sent = '. '.join(array)

            array = []
            for each in results[image_id]['Real Sent']:
                sent = results[image_id]['Real Sent'][each]
                if len(sent) != 0:
                    array.append(sent)
            real_sent = '. '.join(array)
            datasetGTS['annotations'].append({
                'image_id': i,
                'caption': real_sent
            })
            datasetRES['annotations'].append({
                'image_id': i,
                'caption': pred_sent
            })

        rng = range(len(results))
        print (calculate_metrics(rng, datasetGTS, datasetRES))
        self.__save_json(results)

  

    def _generate_cam(self, images_id, visual_features, alpha_v, sentence_id):
        alpha_v *= 100
        cam = torch.mul(visual_features, alpha_v.view(alpha_v.shape[0], alpha_v.shape[1], 1, 1)).sum(1)
        cam.squeeze_()
        cam = cam.cpu().data.numpy()
        for i in range(cam.shape[0]):
            image_id = images_id[i]
            cam_dir = self.__init_cam_path(images_id[i])

            org_img = cv2.imread(os.path.join(self.args.image_dir, image_id), 1)
            org_img = cv2.resize(org_img, (self.args.cam_size, self.args.cam_size))

            heatmap = cam[i]
            heatmap = heatmap / np.max(heatmap)
            heatmap = cv2.resize(heatmap, (self.args.cam_size, self.args.cam_size))
            heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

            img = heatmap * 0.5 + org_img
            cv2.imwrite(os.path.join(cam_dir, '{}.png'.format(sentence_id)), img)

    def __init_cam_path(self, image_file):
        generate_dir = os.path.join(self.args.model_dir, self.args.generate_dir)
        if not os.path.exists(generate_dir):
            os.makedirs(generate_dir)

        image_dir = os.path.join(generate_dir, image_file)

        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        return image_dir

    def __save_json(self, result):
        result_path = os.path.join(self.args.model_dir, self.args.result_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(os.path.join(result_path, '{}.json'.format(self.args.result_name)), 'w') as f:
            json.dump(result, f)

    def __load_mode_state_dict(self):
        try:
            model_state_dict = torch.load(os.path.join(self.args.model_dir, self.args.load_model_path))
            print("[Load Model-{} Succeed!]".format(self.args.load_model_path))
            print("Load From Epoch {}".format(model_state_dict['epoch']))
            return model_state_dict
        except Exception as err:
            print("[Load Model Failed] {}".format(err))
            raise err

    def __init_tagger(self):
        return Tag()

    def __vec2sent(self, array):
        sampled_caption = []
        for word_id in array:
            word = self.vocab.get_word_by_id(word_id)
            if word == '<start>':
                continue
            if word == '<end>' or word == '<pad>':
                break
            sampled_caption.append(word)
        return ' '.join(sampled_caption)

    def __init_vocab(self):
        with open(self.args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        return vocab

    def __init_data_loader(self, file_list):
        data_loader = get_loader(image_dir=self.args.image_dir,
                                 caption_json=self.args.caption_json,
                                 file_list=file_list,
                                 vocabulary=self.vocab,
                                 transform=self.transform,
                                 batch_size=self.args.batch_size,
                                 s_max=self.args.s_max,
                                 n_max=self.args.n_max,
                                 shuffle=False)
        return data_loader

    def __init_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.resize, self.args.resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def __to_var(self, x, requires_grad=True):
        if self.args.cuda:
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)

    def __init_visual_extractor(self):
        # model = VisualFeatureExtractor(model_name=self.args.visual_model_name,
        #                                pretrained=self.args.pretrained)
        model = VisualFeatureExtractor( self.args.hidden_size, self.args.embed_size)

        if self.model_state_dict is not None:
            print("Visual Extractor Loaded!")
            model.load_state_dict(self.model_state_dict['extractor'], strict= False)

        if self.args.cuda:
            model = model.cuda()

        return model

    def __init_sentence_model(self):
        model = SentenceLSTM(#version=self.args.sent_version,
                             embed_size=self.args.embed_size,
                             hidden_size=self.args.hidden_size,
                             num_spatial=self.args.num_spatial,
                             num_layers=self.args.sentence_num_layers,
                             #dropout=self.args.dropout,
                             momentum=self.args.momentum)

        if self.model_state_dict is not None:
            print("Sentence Model Loaded!")
            model.load_state_dict(self.model_state_dict['sentence_model'], strict= False)

        if self.args.cuda:
            model = model.cuda()

        return model

    def __init_word_word(self):
        model = WordLSTM(vocab_size=len(self.vocab),
                         embed_size=self.args.embed_size,
                         hidden_size=self.args.hidden_size,
                         num_layers=self.args.word_num_layers,
                         num_spatial=self.args.num_spatial,
                         n_max=self.args.n_max)

        if self.model_state_dict is not None:
            print("Word Model Loaded!")
            model.load_state_dict(self.model_state_dict['word_model'], strict= False)

        if self.args.cuda:
            model = model.cuda()

        return model


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    """
    Data Argument
    """
    # Path Argument
    parser.add_argument('--model_dir', type=str, default='./report_models/v1/20200228-14:23')
    parser.add_argument('--image_dir', type=str, default='/NLMCXR_png_pairs',
                        help='the path for images')
    parser.add_argument('--caption_json', type=str, default='.json',
                        help='path for captions')
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl',
                        help='the path for vocabulary object')
    parser.add_argument('--file_lits', type=str, default='.test_split.txt',
                        help='the path for test file list')
    parser.add_argument('--load_model_path', type=str, default='train_best_loss.pth.tar',
                        help='The path of loaded model')

    # transforms argument
    parser.add_argument('--resize', type=int, default=224,
                        help='size for resizing images')

    # CAM
    parser.add_argument('--cam_size', type=int, default=224)
    parser.add_argument('--generate_dir', type=str, default='cam')

    # Saved result
    parser.add_argument('--result_path', type=str, default='results',
                        help='the path for storing results')
    parser.add_argument('--result_name', type=str, default='val',
                        help='the name of results')

    """
    Model argument
    """
    parser.add_argument('--momentum', type=int, default=0.1)
    # VisualFeatureExtractor

    parser.add_argument('--num_spatial', type=int, default=100, help='num spatial of last cnn features')

    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)

    # Sentence Model

    parser.add_argument('--sentence_num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Word Model
    parser.add_argument('--word_num_layers', type=int, default=1)

    """
    Generating Argument
    """
    parser.add_argument('--s_max', type=int, default=8)
    parser.add_argument('--n_max', type=int, default=50)

    parser.add_argument('--batch_size', type=int, default=2)

    # Loss function
    parser.add_argument('--lambda_tag', type=float, default=10000)
    parser.add_argument('--lambda_stop', type=float, default=10)
    parser.add_argument('--lambda_word', type=float, default=1)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    print(args)

    sampler = CaptionSampler(args)


    #sampler.generate()
    sampler.generate_pure_hier_cnn()
