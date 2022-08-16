import os

# were to look for files to make the feature extraction
load_path = "/home/usuaris/veu/federico.costa/datasets/voxceleb2/dev"

# output file path
dump_path = "/home/usuaris/veu/federico.costa/git_repositories/DoubleAttentionSpeakerVerification/files_directories/feature_extractor/feature_extractor_paths.lst"

valid_formats = [".wav", ".m4a"]

number_of_files = 0

with open(dump_path, 'w') as f:

    print(f"Searching {valid_formats} files in {load_path}")
    print("-"*50)

    for (dir_path, dir_names, file_names) in os.walk(load_path):

        print(f"Searching in {dir_path}")

        for file_name in file_names:
            if file_name[-4:] in valid_formats:
                path_to_write = f"{dir_path}/{file_name}"
                f.write(path_to_write)
                f.write('\n')
                number_of_files = number_of_files + 1

    print("-"*50)
    print(f"{number_of_files} files paths dumped in {dump_path}")

