import argparse
import pickle
import torch
import numpy as np
import random
import os
import json
import time
import datetime

from model import SpeakerClassifier
from data import normalizeFeatures, featureReader
from utils import scoreCosineDistance, Score, getModelName


class ModelEvaluator:

    def __init__(self, input_params):

        self.input_params = input_params
        self.set_device()
        self.set_random_seed()
        self.evaluation_results = {}
        self.start_time = time.time()
        self.start_datetime = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d %H:%M:%S')

    
    def set_device(self):
    
        print('Setting device...')

        # Set device to GPU or CPU depending on what is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
        print(f"Running on {self.device} device.")
    
        if torch.cuda.device_count() > 1:
            print(f"{torch.cuda.device_count()} GPUs available.")
    
        print("Device setted.")

    
    def set_random_seed(self):

        print("Setting random seed...")

        # Set the seed for experimental reproduction
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)

        print("Random seed setted.")


    def load_checkpoint(self):

        # Load checkpoint
        checkpoint_path = self.input_params.model_checkpoint_path

        print(f"Loading checkpoint from {checkpoint_path}")

        self.checkpoint = torch.load(checkpoint_path, map_location = self.device)

        print(f"Checkpoint loaded.")
        
    
    def load_checkpoint_params(self):

        self.params = self.checkpoint['settings']


    def load_checkpoint_network(self):

        try:
            self.net.load_state_dict(self.checkpoint['model'])
        except RuntimeError:    
            self.net.module.load_state_dict(self.checkpoint['model'])


    def load_network(self):

        self.net = SpeakerClassifier(self.params, self.device)
        
        self.load_checkpoint_network()
        
        # Assign model to device
        self.net.to(self.device)

        if torch.cuda.device_count() > 1:
            logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
            self.net = nn.DataParallel(self.net)


    def __extractInputFromFeature(self, sline, data_dir):

        features1 = normalizeFeatures(
            featureReader(
                data_dir + '/' + sline[0] + '.pickle'), 
                normalization=self.params.normalization,
                )
        features2 = normalizeFeatures(
            featureReader(
                data_dir + '/' + sline[1] + '.pickle'), 
                normalization=self.params.normalization,
                )

        input1 = torch.FloatTensor(features1).to(self.device)
        input2 = torch.FloatTensor(features2).to(self.device)
        
        return input1.unsqueeze(0), input2.unsqueeze(0)


    def __extract_scores(self, trials, data_dir):

        scores = []
        for line in trials:
            sline = line[:-1].split()

            input1, input2 = self.__extractInputFromFeature(sline, data_dir)

            if torch.cuda.device_count() > 1:
                emb1, emb2 = self.net.module.getEmbedding(input1), self.net.module.getEmbedding(input2)
            else:
                emb1, emb2 = self.net.getEmbedding(input1), self.net.getEmbedding(input2)

            dist = scoreCosineDistance(emb1, emb2)
            scores.append(dist.item())
        
        return scores


    def __calculate_EER(self, CL, IM):

        thresholds = np.arange(-1,1,0.01)
        FRR, FAR = np.zeros(len(thresholds)), np.zeros(len(thresholds))
        for idx,th in enumerate(thresholds):
            FRR[idx] = Score(CL, th,'FRR')
            FAR[idx] = Score(IM, th,'FAR')

        EER_Idx = np.argwhere(np.diff(np.sign(FAR - FRR)) != 0).reshape(-1)
        if len(EER_Idx)>0:
            if len(EER_Idx)>1:
                EER_Idx = EER_Idx[0]
            EER = round((FAR[int(EER_Idx)] + FRR[int(EER_Idx)])/2,4)
        else:
            EER = 50.00

        return EER


    def evaluate(self, clients_labels, impostor_labels, data_dir):

        with torch.no_grad():

            # Switch torch to evaluation mode
            self.net.eval()

            print("Going to evaluate using these labels:")
            print(f"Clients: {clients_labels}")
            print(f"Impostors: {impostor_labels}")
            print(f"For each row in these labels where are using prefix {data_dir}")

            # EER Validation
            with open(clients_labels,'r') as clients_in, open(impostor_labels,'r') as impostors_in:
                # score clients
                CL = self.__extract_scores(clients_in, data_dir)
                IM = self.__extract_scores(impostors_in, data_dir)
            
            # Compute EER
            EER = self.__calculate_EER(CL, IM)

        return EER

    
    def evaluate_validation(self):

        print("Evaluating model on validation dataset...")
        
        result = self.evaluate(
            clients_labels = self.input_params.valid_clients,
            impostor_labels = self.input_params.valid_impostors, 
            data_dir = self.input_params.valid_data_dir,
            )

        self.evaluation_results['valid_result'] = result

        print(f"Model evaluated on validation dataset. EER: {result:.2f}")


    def evaluate_test(self):

        print("Evaluating model on test dataset...")
        
        result = self.evaluate(
            clients_labels = self.input_params.test_clients,
            impostor_labels = self.input_params.test_impostors, 
            data_dir = self.input_params.test_data_dir,
            )

        self.evaluation_results['test_result'] = result

        print(f"Model evaluated on test dataset. EER: {result:.2f}")


    def evaluate_valid_and_test(self):

        self.evaluate_validation()
        self.evaluate_test()


    def save_report(self):

        print("Creating report...")

        self.end_time = time.time()
        self.end_datetime = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d %H:%M:%S')
        self.elapsed_time_hours = (self.end_time - self.start_time) / 60 / 60
        model_name = getModelName(self.params)

        self.evaluation_results['start_datetime'] = self.start_datetime
        self.evaluation_results['end_datetime'] = self.end_datetime
        self.evaluation_results['elapsed_time_hours'] = self.elapsed_time_hours
        self.evaluation_results['model_name'] = model_name

        dump_folder = self.input_params.dump_folder
        if not os.path.exists(dump_folder):
            os.makedirs(dump_folder)

        dump_file_name = f"report_{model_name}.json"

        dump_path = os.path.join(dump_folder, dump_file_name)
        
        print(f"Saving file into {dump_path}")
        with open(dump_path, 'w', encoding = 'utf-8') as handle:
            json.dump(self.evaluation_results, handle, ensure_ascii = False, indent = 4)
        print("Saved.")


    def main(self):
        self.load_checkpoint()
        self.load_checkpoint_params()
        self.load_network()
        self.evaluate_valid_and_test()
        self.save_report()
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='score a trained model')

    parser.add_argument(
        '--model_checkpoint_path', 
        type = str, 
        default = '/home/usuaris/veu/federico.costa/git_repositories/DoubleAttentionSpeakerVerification/models/model1/CNN_VGG4L_3.5_128batchSize_0.0001lr_0.001weightDecay_1024kernel_400embSize_30.0s_0.4m_DoubleMHA_32_500.chkpt'
        ) 

    parser.add_argument(
        '--dump_folder', 
        type = str, 
        default = '/home/usuaris/veu/federico.costa/git_repositories/DoubleAttentionSpeakerVerification/scripts/models_results'
        )

    parser.add_argument(
        '--valid_clients', 
        type = str, 
        default = '/home/usuaris/veu/federico.costa/git_repositories/DoubleAttentionSpeakerVerification/scripts/labels/evaluation/valid/clients.ndx',
        help = 'Path of the file containing the validation clients pairs paths.',
        )

    parser.add_argument(
        '--valid_impostors', 
        type = str, 
        default = '/home/usuaris/veu/federico.costa/git_repositories/DoubleAttentionSpeakerVerification/scripts/labels/evaluation/valid/impostors.ndx',
        help = 'Path of the file containing the validation impostors pairs paths.',
        )

    parser.add_argument(
        '--valid_data_dir', 
        type = str, 
        default = '/home/usuaris/scratch/speaker_databases/VoxCeleb-1/wav', 
        help = 'Optional additional directory to prepend to valid_clients and valid_impostors paths.',
        )

    parser.add_argument(
        '--test_clients', 
        type = str, 
        default = '/home/usuaris/veu/federico.costa/git_repositories/DoubleAttentionSpeakerVerification/scripts/labels/evaluation/test/clients.ndx',
        )

    parser.add_argument(
        '--test_impostors', 
        type = str, 
        default = '/home/usuaris/veu/federico.costa/git_repositories/DoubleAttentionSpeakerVerification/scripts/labels/evaluation/test/impostors.ndx',
        )

    parser.add_argument(
        '--test_data_dir', 
        type = str, 
        default = '/home/usuaris/scratch/speaker_databases/VoxCeleb-1/wav', 
        )

    input_params = parser.parse_args()
        
    model_evaluator = ModelEvaluator(input_params)
    model_evaluator.main()