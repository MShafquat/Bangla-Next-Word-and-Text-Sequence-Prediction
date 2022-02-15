from pathlib import Path
import pandas as pd
import re

class Preprocess():
    def __init__(self, raw_files_dir, processed_file_path):
        self.raw_files_dir = raw_files_dir
        self.processed_file_path = processed_file_path
        self.files = Path(self.raw_files_dir).glob('**/*.txt')
    
    def process_file(self, file_path):
        self.data = pd.read_csv(file_path, sep='\r\n', header=None, engine='python')
        self.data = self.data[self.data.apply(self.__is_bangla_sentence, axis=1)]
        self.data = self.data.apply(self.__replace_punctuations, axis=1)
        self.data = self.data.dropna()
        self.data = self.data.drop_duplicates()

    def __is_bangla_sentence(self, row):
        """
        Checks if a string is a pure Bangla sentence
        containing only Bangla letters, digits, and punctuations
        """
        pattern = u'^[\u0020|\u0980-\u09FF]|[\u2000-\u206F]+$'
        return bool(re.match(pattern, row[0]))

    def __replace_punctuations(self, row):
        """
        Replaces all punctuations with space
        """
        pattern = u'[\u2000-\u206F]+'
        return re.sub(pattern, ' ', row[0])
