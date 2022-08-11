import argparse

parser = argparse.ArgumentParser(description='Train a VGG based Speaker Embedding Extractor')
   
parser.add_argument('--train_data_dir', type=str, default='/scratch/speaker_databases/', help='data directory.')
parser.add_argument('--valid_data_dir', type=str, default='/scratch/speaker_databases/VoxCeleb-1/wav', help='data directory.')
parser.add_argument('--train_labels_path', type = str, default = 'labels/Vox2.ndx')
parser.add_argument('--data_mode', type = str, default = 'normal', choices=['normal','window'])
parser.add_argument('--valid_clients', type = str, default='labels/clients.ndx')
parser.add_argument('--valid_impostors', type = str, default='labels/impostors.ndx')
parser.add_argument('--out_dir', type=str, default='./models/model1', help='directory where data is saved')
parser.add_argument('--model_name', type=str, default='CNN', help='Model associated to the model builded')
parser.add_argument('--front_end', type=str, default='VGG4L', choices = ['VGG3L','VGG4L'], help='Kind of Front-end Used')

# Network Parameteres
parser.add_argument('--window_size', type=float, default=3.5, help='number of seconds per window')
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
parser.add_argument('--early_stopping', type=int, default=25, help='-1 if not early stopping')
parser.add_argument('--print_every', type = int, default = 1000)
parser.add_argument('--requeue',action='store_true', help='restart from the last model for requeue on slurm')
parser.add_argument('--validate_every', type = int, default = 10000)
parser.add_argument('--num_workers', type = int, default = 2)

# parse input params
params = parser.parse_args()

print(vars(params))