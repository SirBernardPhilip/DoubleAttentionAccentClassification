import os
import warnings
import itertools
import random

# Setted variables
dev_dataset_path = "/home/usuaris/veu/federico.costa/datasets/voxceleb2/dev"
train_speakers_pctg = 0.8
labels_dump_path = "/home/usuaris/veu/federico.costa/git_repositories/DoubleAttentionSpeakerVerification/files_directories/labels/labels.ndx"
impostors_dump_path = "/home/usuaris/veu/federico.costa/git_repositories/DoubleAttentionSpeakerVerification/files_directories/labels/impostors.ndx"
clients_dump_path = "/home/usuaris/veu/federico.costa/git_repositories/DoubleAttentionSpeakerVerification/files_directories/labels/clients.ndx"
clients_lines_max = None
impostors_lines_max = None

def generate_speakers_dict(load_path):
    
    speakers_set = set()
    speakers_dict = {}
    
    for (dir_path, dir_names, file_names) in os.walk(load_path):
        
        # Directory should have some /id.../ part
        speaker_chunk = [chunk for chunk in dir_path.split("/") if chunk.startswith("id")]
    
        # Only consider directories with /id.../
        if len(speaker_chunk) > 0: 
        
            speaker_id = speaker_chunk[0]
            
            # If speaker_id is looped for the first time, initialize variables
            if speaker_id not in speakers_set:
                speakers_dict[speaker_id] = {}
                speakers_dict[speaker_id]["files_paths"] = set()
                
            # If it is a .pickle file, add the path to speakers_dict
            for file_name in file_names:
                if file_name.split(".")[-1] == "pickle":                
                    file_path = dir_path + "/" + file_name.replace(".pickle", "")
                    speakers_dict[speaker_id]["files_paths"].add(file_path)
            
            # Add speaker_id to set and continue with the loop
            speakers_set.add(speaker_id)

        # If there is some other "/id..." in the directory it should be looked
        if len(speaker_chunk) > 1:
            warnings.warn(f"Ambiguous directory path: {dir_path}")
            
    # Add 0 to total_speakers-1 labels
    speakers_list = list(speakers_set)
    speakers_list.sort()
    
    for i, speaker in enumerate(speakers_list):
        speakers_dict[speaker]["speaker_num"] = i
        
    # Sort in order of labels for better understanding
    speakers_dict = {k: v for k, v in sorted(speakers_dict.items(), key=lambda item: item[1]["speaker_num"])}
    
    return speakers_dict


def train_valid_split_dict(speakers_dict, train_speakers_pctg, labels_dump_path, impostors_dump_path):
    
    # We are going to randomly split speaker_id's
    
    num_speakers = len(speakers_dict)
    
    train_speakers_final_index = int(num_speakers * train_speakers_pctg)
    random_speaker_nums = list(range(num_speakers))
    random.shuffle(random_speaker_nums)
    
    for i, speaker in enumerate(speakers_dict.keys()):
        speakers_dict[speaker]["random_speaker_num"] = random_speaker_nums[i]
        
    train_speakers_dict = speakers_dict.copy()
    valid_speakers_dict = speakers_dict.copy()

    for speaker in speakers_dict.keys():

        random_speaker_num = speakers_dict[speaker]["random_speaker_num"]

        if random_speaker_num > train_speakers_final_index:
            del train_speakers_dict[speaker]
        else:
            del valid_speakers_dict[speaker]

    train_speakers_num = len(train_speakers_dict.keys())
    valid_speakers_num = len(valid_speakers_dict.keys())
    total_speakers_num = train_speakers_num + valid_speakers_num
    if total_speakers_num != num_speakers:
        raise Exception("total_speakers_num does not match total_speakers_num!")
    train_speakers_pctg = train_speakers_num * 100 / total_speakers_num
    valid_speakers_pctg = valid_speakers_num * 100 / total_speakers_num
    
    
    train_files_num = len(list(itertools.chain.from_iterable([value["files_paths"] for value in train_speakers_dict.values()])))
    valid_files_num = len(list(itertools.chain.from_iterable([value["files_paths"] for value in valid_speakers_dict.values()])))
    total_files_num = train_files_num + valid_files_num
    train_files_pctg = train_files_num * 100 / total_files_num
    valid_files_pctg = valid_files_num * 100 / total_files_num
    
    
    print(f"{train_speakers_num} speakers ({train_speakers_pctg:.1f}%) with a total of {train_files_num} files ({train_files_pctg:.1f}%) in training split.")
    print(f"{valid_speakers_num} speakers ({valid_speakers_pctg:.1f}%) with a total of {valid_files_num} files ({valid_files_pctg:.1f}%) in training split.")
    
    return train_speakers_dict, valid_speakers_dict


def generate_labels_file(dump_path, speakers_dict):
    
    with open(dump_path, 'w') as f:
        for key, value in speakers_dict.items():
            speaker_num = value["speaker_num"]
            for file_path in value["files_paths"]:
                line_to_write = f"{file_path} {speaker_num} -1"  
                f.write(line_to_write)
                f.write('\n')
        f.close()


def generate_clients_impostors_files(
    impostors_dump_path, clients_dump_path, 
    speakers_dict, 
    clients_lines_max = None, impostors_lines_max = None):
    
    clients_lines_to_write = []
    impostors_lines_to_write = []

    distinct_speakers = list(speakers_dict.keys())

    one_speaker_combinations = [(speaker, speaker) for speaker in distinct_speakers]
    two_speaker_combinations = list(itertools.combinations(distinct_speakers, 2))  
    speaker_combinations = one_speaker_combinations + two_speaker_combinations

    for speaker_1, speaker_2 in speaker_combinations:

        speaker_1_files = speakers_dict[speaker_1]["files_paths"]
        speaker_2_files = speakers_dict[speaker_2]["files_paths"]

        if speaker_1 == speaker_2:
            files_combinations = list(itertools.combinations(speaker_1_files, 2))
            for file_1, file_2 in files_combinations:
                line_to_write = file_1 + " " + file_2
                clients_lines_to_write.append(line_to_write)
        else:
            files_combinations = list(itertools.product(speaker_1_files, speaker_2_files))
            for file_1, file_2 in files_combinations:
                line_to_write = file_1 + " " + file_2
                impostors_lines_to_write.append(line_to_write)

    if clients_lines_max is not None:
        clients_lines_to_write = random.sample(clients_lines_to_write, clients_lines_max)
    if impostors_lines_max is not None:
        impostors_lines_to_write = random.sample(impostors_lines_to_write, impostors_lines_max)
    
    print(f"{len(clients_lines_to_write)} lines to write for clients.")
    print(f"{len(impostors_lines_to_write)} lines to write for impostors.")
    
    with open(clients_dump_path, 'w') as f:
        for line_to_write in clients_lines_to_write: 
            f.write(line_to_write)
            f.write('\n')
        f.close()

    with open(impostors_dump_path, 'w') as f:
        for line_to_write in impostors_lines_to_write: 
            f.write(line_to_write)
            f.write('\n')
        f.close()



print("Loading dev data...")
dev_speakers_dict = generate_speakers_dict(
    load_path = dev_dataset_path,
)

num_speakers = len(dev_speakers_dict)
print(f"Total number of distinct speakers loaded: {num_speakers}")

print(f"Spliting data into train and valid...")
train_speakers_dict, valid_speakers_dict = train_valid_split_dict(
    speakers_dict = dev_speakers_dict, 
    train_speakers_pctg = train_speakers_pctg, 
    labels_dump_path = None, 
    impostors_dump_path = None,
)
print(f"Data splited.")

print(f"Generating training labels...")
generate_labels_file(
    dump_path = labels_dump_path, 
    speakers_dict = train_speakers_dict,
)
print(f"Training labels generated.")

print(f"Generating valid clients and impostors trials...")
generate_clients_impostors_files(
    impostors_dump_path = impostors_dump_path, 
    clients_dump_path = clients_dump_path, 
    speakers_dict = valid_speakers_dict, 
    clients_lines_max = None, 
    impostors_lines_max = None,
)
print(f"Valid clients and impostors trials generated.")