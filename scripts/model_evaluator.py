import argparse
import pickle
import torch
import numpy as np
import random


class ModelEvaluator:

    def __init__(self, input_params):

        self.input_params = input_params
        self.set_device()
        self.set_random_seed()

    
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


    def main(self):
        self.load_checkpoint()



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='score a trained model')

    parser.add_argument(
        '--model_checkpoint_path', 
        type = str, 
        default = '/home/usuaris/veu/federico.costa/git_repositories/DoubleAttentionSpeakerVerification/models/model1/CNN_VGG4L_3.5_128batchSize_0.0001lr_0.001weightDecay_1024kernel_400embSize_30.0s_0.4m_DoubleMHA_32_500.chkpt'
        ) 

    input_params = parser.parse_args()
        
    model_evaluator = ModelEvaluator(input_params)
    model_evaluator.main()