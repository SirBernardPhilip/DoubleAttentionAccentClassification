import argparse
import os

from settings import paths_generator_default_settings

class PathsGenerator:

    def __init__(self, params):
        self.params = params
    
    
    def search_files(self):

        self.lines_to_write = []

        print(f"Searching {self.params.valid_audio_formats} files in {self.params.load_data_dir}")
        
        for (dir_path, dir_names, file_names) in os.walk(self.params.load_data_dir):

                if self.params.verbose: print(f"Searching in {dir_path}")

                for file_name in file_names:
        
                    if file_name.split(".")[-1] in self.params.valid_audio_formats:
                        
                        path_to_write = f"{dir_path}/{file_name}"
                        self.lines_to_write.append(path_to_write)

        print(f"{len(self.lines_to_write)} files founded in {self.params.load_data_dir}")

    
    def dump_paths(self):

        if not os.path.exists(self.params.dump_data_dir):
            os.makedirs(self.params.dump_data_dir)
        
        dump_path = self.params.dump_data_dir + self.params.dump_file_name
        with open(dump_path, 'w') as file:

            for line_to_write in self.lines_to_write:
                file.write(line_to_write)
                file.write('\n')

        print(f"{len(self.lines_to_write)} files paths dumped in {self.params.dump_data_dir}")

    
    def main(self):

        self.search_files()
        self.dump_paths()
        

class ArgsParser:

    def __init__(self):
        
        self.initialize_parser()

    
    def initialize_parser(self):

        self.parser = argparse.ArgumentParser(
            description = 'Generate audio paths file as input for the feature exctractor script. \
                It searches audio files in a directory and dumps their paths into a .lst file.',
            )


    def add_parser_args(self):

        self.parser.add_argument(
            'load_data_dir',
            type = str, 
            help = 'Data directory containing the audio files we want to extract features from.',
            )

        self.parser.add_argument(
            '--dump_file_name', 
            type = str, 
            default = paths_generator_default_settings['dump_file_name'], 
            help = 'Name of the .lst file we want to dump paths into.',
            )

        self.parser.add_argument(
            '--dump_data_dir', 
            type = str, 
            default = paths_generator_default_settings['dump_data_dir'], 
            help = 'Data directory where we want to dump the .lst file.',
            )
        
        self.parser.add_argument(
            '--valid_audio_formats', 
            action = 'append',
            default = paths_generator_default_settings['valid_audio_formats'],
            help = 'Audio files extension to search for.',
            )

        self.parser.add_argument(
            "--verbose", 
            action = "store_true",
            help = "Increase output verbosity.",
            )


    def main(self):

        self.add_parser_args()
        self.arguments = self.parser.parse_args()


if __name__=="__main__":

    args_parser = ArgsParser()
    args_parser.main()
    parameters = args_parser.arguments

    paths_generator = PathsGenerator(parameters)
    paths_generator.main()
    