import os
import sys
import argparse
import numpy as np
import random
import pickle
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
sys.path.append('./scripts/')
from data import *
from model import SpeakerClassifier
from loss import *
from utils import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self, params, device):
        self.writer = SummaryWriter('./runs/'+params.out_dir)

        self.params = params
        self.device = device
        self.__load_network()
        self.__load_data()
        self.__load_optimizer()
        self.__load_criterion()
        self.__initialize_training_variables()

    def __load_previous_states(self):

        list_files = os.listdir('./models/' + self.params.out_dir)
        list_files = ['./models/' + self.params.out_dir + '/' + f for f in list_files if '.chkpt' in f]
        if list_files:
            file2load = max(list_files, key=os.path.getctime)
            checkpoint = torch.load(file2load, map_location=self.device)
            try:
                self.net.load_state_dict(checkpoint['model'])
            except RuntimeError:
                self.net.module.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.params = checkpoint['settings']
            self.starting_epoch = checkpoint['epoch']+1
            self.step = checkpoint['step']
            print('Model "%s" is Loaded for requeue process' % file2load)
        else:
            self.step = 0
            self.starting_epoch = 1

    def __initialize_training_variables(self):

        if self.params.requeue:
            self.__load_previous_states()
        else:
            self.step = 0
            self.starting_epoch = 0

        self.best_f1_macro = 0.0
        self.best_f1_micro = 0.0
        self.stopping = 0.0


    def __load_network(self):

        self.net = SpeakerClassifier(self.params, self.device)
        self.net.to(self.device)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.net = nn.DataParallel(self.net)


    def __load_data(self):
        print('Loading Data and Labels')
        with open(self.params.train_labels_path, 'r') as data_labels_file:
            train_labels=data_labels_file.readlines()

        data_loader_parameters = {'batch_size': self.params.batch_size, 'shuffle': True, 'num_workers': 1}
        self.training_generator = DataLoader(Dataset(train_labels, self.params, True, True, self.params.padding == 'yes', self.params.features == 'old', self.params.num_accents), **data_loader_parameters)
        with open(self.params.val_labels_path, 'r') as data_labels_file:
            val_labels=data_labels_file.readlines()

        data_loader_parameters = {'batch_size': 1, 'shuffle': False, 'num_workers': self.params.num_workers}
        self.validation_generator = DataLoader(Dataset(val_labels, self.params, False, self.params.data_mode_dev == 'window', self.params.padding == 'yes', self.params.features == 'old', self.params.num_accents), **data_loader_parameters)


    def __load_optimizer(self):
        if self.params.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.params.learning_rate, weight_decay=self.params.weight_decay)
        if self.params.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.params.learning_rate, weight_decay=self.params.weight_decay)
        if self.params.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.net.parameters(), lr=self.params.learning_rate, weight_decay=self.params.weight_decay)

    def __update_optimizer(self):

        if self.params.optimizer == 'SGD' or self.params.optimizer == 'Adam':
            for paramGroup in self.optimizer.param_groups:
                paramGroup['lr'] *= 0.5
            print('New Learning Rate: {}'.format(paramGroup['lr']))
    
    def __load_criterion(self):
            self.criterion = nn.CrossEntropyLoss()

    def __initialize_batch_variables(self):

        self.print_time = time.time()
        self.train_loss = 0.0
        self.train_accuracy = 0.0
        self.train_f1_macro = 0.0
        self.train_f1_micro = 0.0
        self.train_batch = 0

    def __extractInputFromFeature(self, sline):

        features1 = normalizeFeatures(featureReader(self.params.valid_data_dir + '/' + sline[0] + '.pickle'), normalization=self.params.normalization)
        features2 = normalizeFeatures(featureReader(self.params.valid_data_dir + '/' + sline[1] + '.pickle'), normalization=self.params.normalization)

        input1 = torch.FloatTensor(features1).to(self.device)
        input2 = torch.FloatTensor(features2).to(self.device)
        
        return input1.unsqueeze(0), input2.unsqueeze(0)

    def __extract_scores(self, trials):

        scores = []
        for line in trials:
            sline = line[:-1].split()

            input1, input2 = self.__extractInputFromFeature(sline)

            if torch.cuda.device_count() > 1:
                emb1, emb2 = self.net.module.getEmbedding(input1), self.net.module.getEmbedding(input2)
            else:
                emb1, emb2 = self.net.getEmbedding(input1), self.net.getEmbedding(input2)

            dist = scoreCosineDistance(emb1, emb2)
            scores.append(dist.item())

        return scores

    def __getAnnealedFactor(self):
        if torch.cuda.device_count() > 1:
            return self.net.module.predictionLayer.getAnnealedFactor(self.step)
        else:
            return self.net.predictionLayer.getAnnealedFactor(self.step)

    def __validate(self):

        with torch.no_grad():
            valid_time = time.time()
            self.net.eval()
            f1_macro = 0
            accuracy = 0
            loss = 0
            final_preds = torch.tensor([])
            final_labels = torch.tensor([])
            final_ampreds = torch.tensor([])
            for input, label in self.validation_generator:
                input, label = input.float().to(self.device), label.long().to(self.device)
                prediction, AMPrediction  = self.net(input, label=label, step=self.step)
                if final_preds.shape[0] == 0:
                    final_preds = prediction
                    final_labels = label
                else:
                    final_labels = torch.cat((final_labels, label), 0)
                    final_preds = torch.cat((final_preds, prediction), 0)
                    final_ampreds = torch.cat((final_ampreds, AMPrediction.cpu()), 0)
            
            self.writer.add_figure("Confusion matrix", ConfusionMatrix(final_preds, final_labels, self.params.num_accents), self.step)
            accuracy = Accuracy(final_preds, final_labels)*100
            f1_macro = F1_macro(final_preds, final_labels)*100
            f1_micro = F1_micro(final_preds, final_labels)*100
            loss = self.criterion(final_preds, final_labels)
            self.writer.add_scalar('Loss/val', loss , self.step)
            self.writer.add_scalar('Accuracy/val', accuracy  , self.step)
            self.writer.add_scalar('F1_macro/val', f1_macro, self.step)
            self.writer.add_scalar('F1_micro/val', f1_micro, self.step)
            print('--Validation Epoch:{epoch: d}, Updates:{Num_Batch: d}, Loss: {loss: 3.3f}, F1 Macro:{f1_macro: 3.3f}, F1 Micro:{f1_micro: 3.3f}, Accuracy:{acc: 3.3f}, elapse:{elapse: 3.3f} min'.format(epoch=self.epoch, Num_Batch=self.step, f1_macro=f1_macro, f1_micro=f1_micro, acc=accuracy, loss=loss, elapse=(time.time()-valid_time)/60))
            # early stopping and save the best model
            if f1_macro > self.best_f1_macro:
                self.best_f1_macro = f1_macro
                self.stopping = 0
                print('We found a better model!')
                chkptsave(params, self.net, self.optimizer, self.epoch, self.step)
            else:
                self.stopping += 1
                print('Better F1 macro is: {}. {} epochs of no improvement'.format(self.best_f1_macro, self.stopping))
            self.print_time = time.time()
            self.net.train()

    def __update(self):

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1

        if self.step % int(self.params.print_every) == 0:
            self.writer.add_scalar('Loss/train', self.train_loss/self.train_batch , self.step)
            self.writer.add_scalar('Accuracy/train', self.train_accuracy *100/self.train_batch  , self.step)
            self.writer.add_scalar('F1_macro/train', self.train_f1_macro *100/self.train_batch, self.step)
            self.writer.add_scalar('F1_micro/train', self.train_f1_micro *100/self.train_batch, self.step)
            print('Training Epoch:{epoch: d}, Updates:{Num_Batch: d} -----> xent:{xnet: .3f}, Accuracy:{acc: .2f}, F1 Macro:{f1_macro: .2f}, F1 Micro:{f1_micro: .2f}, elapse:{elapse: 3.3f} min'.format(epoch=self.epoch, Num_Batch=self.step, xnet=self.train_loss / self.train_batch, acc=self.train_accuracy *100/ self.train_batch, f1_macro=self.train_f1_macro *100/ self.train_batch, f1_micro=self.train_f1_micro *100/ self.train_batch, elapse=(time.time()-self.print_time)/60))
            self.__initialize_batch_variables()

        # validation
        if self.step % self.params.validate_every == 0:
            self.__validate()

    def __updateTrainningVariables(self):

        if (self.stopping+1)% 15 ==0:
            self.__update_optimizer()

    def __randomSlice(self, inputTensor):
        index = random.randrange(200,self.params.window_size*100)
        return inputTensor[:,:index,:]

    def train(self):

        print('Start Training')
        for self.epoch in range(self.starting_epoch, self.params.max_epochs):  # loop over the dataset multiple times
            self.net.train()
            self.__initialize_batch_variables()
            for input, label in tqdm(self.training_generator):
                input, label = input.float().to(self.device), label.long().to(self.device)
                input = self.__randomSlice(input) if self.params.randomSlicing else input 
                prediction, AMPrediction  = self.net(input, label=label, step=self.step)
                loss = self.criterion(AMPrediction, label)
                loss.backward()
                self.train_accuracy += Accuracy(prediction, label)
                self.train_f1_macro += F1_macro(prediction, label)
                self.train_f1_micro += F1_micro(prediction, label)
                self.train_loss += loss.item()
                
                self.train_batch += 1
                if self.train_batch % self.params.gradientAccumulation == 0:
                    self.__update()

            if self.stopping > self.params.early_stopping:
                print('--Best Model F1 Macro%%: %.2f' %(self.best_f1_macro))
                break
            
            self.__updateTrainningVariables()


        print('Finished Training')

def main(opt):

    torch.manual_seed(1234)
    np.random.seed(1234)
    
    print('Defining Device')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))

    print('Loading Trainer')
    trainer = Trainer(opt, device)
    trainer.train()

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Train a VGG based Speaker Embedding Extractor')

    parser.add_argument('--train_labels_path', type = str, default = './labels/no_limit/train_files.lst')
    parser.add_argument('--val_labels_path', type = str, default = './labels/no_limit/dev_files.lst')
    parser.add_argument('--features', type = str, default = 'new', choices=['old','new'])
    parser.add_argument('--padding', type = str, default = 'yes', choices=['yes','no'])
    parser.add_argument('--data_mode', type = str, default = 'normal', choices=['normal','window'])
    parser.add_argument('--data_mode_dev', type = str, default = 'normal', choices=['normal','window'])
    parser.add_argument('--out_dir', type=str, default='5_labels/limit_speaker', help='directory where data is saved')
    parser.add_argument('--model_name', type=str, default='CNN', help='Model associated to the model builded')
    parser.add_argument('--front_end', type=str, default='VGG4L', choices = ['VGG3L','VGG4L', 'ResNet34'], help='Kind of Front-end Used')
    parser.add_argument('--num_accents', type=int, default=5)

    # Network Parameteres
    parser.add_argument('--window_size', type=float, default=0.2, help='number of seconds per window')
    parser.add_argument('--randomSlicing',action='store_true')
    parser.add_argument('--normalization', type=str, default='cmn', choices=['cmn', 'cmvn'])
    parser.add_argument('--kernel_size', type=int, default=1024)
    parser.add_argument('--embedding_size', type=int, default=400)
    parser.add_argument('--heads_number', type=int, default=32)
    parser.add_argument('--pooling_method', type=str, default='DoubleMHA', choices=['Attention', 'MHA', 'DoubleMHA'], help='Type of pooling methods')
    parser.add_argument('--mask_prob', type=float, default=0.3, help='Masking Drop Probability. Only Used for Only Double MHA')
 
    # AMSoftmax Config
    parser.add_argument('--scalingFactor', type=float, default=30.0, help='')
    parser.add_argument('--marginFactor', type=float, default=0.4, help='')
    parser.add_argument('--annealing', action='store_true')

    # Optimization 
    parser.add_argument('--optimizer', type=str, choices=['Adam', 'SGD', 'RMSprop'], default='Adam')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='')
    parser.add_argument('--batch_size', type=int, default=64, help='number of sequences to train on in parallel')
    parser.add_argument('--gradientAccumulation', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=1000000, help='number of full passes through the trainning data')
    parser.add_argument('--early_stopping', type=int, default=50, help='-1 if not early stopping')
    parser.add_argument('--print_every', type = int, default = 1000)
    parser.add_argument('--requeue',action='store_true', help='restart from the last model for requeue on slurm')
    parser.add_argument('--validate_every', type = int, default = 1000)
    parser.add_argument('--num_workers', type = int, default = 8)
    
    # parse input params
    params=parser.parse_args()
    params.model_name = getModelName(params)
    # params.num_accents = getNumberOfAccents(params.train_labels_path) 
    print('{} Speaker Labels'.format(params.num_accents))

    if not os.path.exists('./models/' + params.out_dir):
        os.makedirs('./models/' + params.out_dir)

    with open( './models/' + params.out_dir + '/' + params.model_name + '_config.pkl', 'wb') as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    main(params)
