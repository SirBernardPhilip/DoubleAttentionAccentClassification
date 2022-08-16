import os
import warnings

# were to look for files to make the feature extraction
load_path = "/home/usuaris/veu/federico.costa/datasets/voxceleb2/dev"

# output file path
dump_path = "/home/usuaris/veu/federico.costa/git_repositories/DoubleAttentionSpeakerVerification/files_directories/labels/train/labels.ndx"

speakers_set = set()
speakers_dict = {}
for (dir_path, dir_names, file_names) in os.walk(load_path):
    speaker_chunk = [chunk for chunk in dir_path.split("/") if chunk.startswith("id")]
    
    if len(speaker_chunk) > 0: 
        
        speaker_id = speaker_chunk[0]
        if speaker_id not in speakers_set:
            speakers_dict[speaker_id] = {}
            speakers_dict[speaker_id]["files_paths"] = set()
        speakers_set.add(speaker_id)
        
        for file_name in file_names:
            if file_name.split(".")[-1] == "pickle":                
                
                file_path = dir_path + "/" + file_name.replace(".pickle", "")
                speakers_dict[speaker_id]["files_paths"].add(file_path)
        
        if len(speaker_chunk) > 1:
            warnings.warn("Ambiguous directory path!")

speakers_list = list(speakers_set)
speakers_list.sort()
num_speakers = len(speakers_list)
for i, speaker in enumerate(speakers_list):
    speakers_dict[speaker]["speaker_num"] = i
    
speakers_dict = {k: v for k, v in sorted(speakers_dict.items(), key=lambda item: item[1]["speaker_num"])}
    
with open(dump_path, 'w') as f:
    for key, value in speakers_dict.items():
        speaker_num = value["speaker_num"]
        for file_path in value["files_paths"]:
            line_to_write = f"{file_path} {speaker_num} -1"  
            f.write(line_to_write)
            f.write('\n')