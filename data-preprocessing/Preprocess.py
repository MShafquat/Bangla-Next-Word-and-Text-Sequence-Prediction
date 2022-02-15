import os
from pathlib import Path
import pandas as pd
import re

class Preprocess():
    def __init__(self, raw_files_dir, processed_file_dir):
        self.raw_files_dir = raw_files_dir
        self.processed_file_path = processed_file_dir
        os.makedirs(processed_file_dir, exist_ok=True)
        self.raw_files = Path(self.raw_files_dir).glob('**/*.txt')
        self.preprocess_and_save_files()

    def preprocess_and_save_files(self):
        """
        Performs preprocessing on all files in the raw_files_dir
        and saves the processed dataframes to the processed_file_dir
        """
        for file in self.raw_files:
            file_name = file.name
            file_path = self.processed_file_path / file_name
            df = self.preprocess_file(file)
            self.save_dataframe_to_file(df, file_path)
    
    def preprocess_file(self, file_path):
        """
        Performs preprocessing on a text file and returns a dataframe
        """
        data = pd.read_csv(file_path, sep="\r\n", engine='python', header=None)[0] # reads text file as dataframe
        data = data.str.split('।|\.|\?|!').explode() # splits long text to separate sentences
        data = data.apply(self.__replace_unknowns) # removes unknown characters
        data = data[data.apply(self.__is_bangla_sentence)] # removes non-bangla sentences
        data = data.dropna() # removes NaN values
        data = data.drop_duplicates() # removes duplicate sentences
        data = data.reset_index(drop=True) # resets index
        return data

    def save_dataframe_to_file(self, df, file_path):
        """
        Writes the processed dataframe to a file
        """
        df.to_csv(file_path, sep='\n', index=False, header=False)

    def __replace_unknowns(self, row):
        """
        Replaces everything that is not any English or Bangla
        alphanumeric character with an space
        """
        row = re.sub(u'[^A-Za-z0-9\u0980-\u09FF ]',' ', row)
        row = re.sub(' +',' ', row)
        row = row.strip()
        return row

    def __is_bangla_sentence(self, row):
        """
        Checks if a string is a pure Bangla sentence
        containing only Bangla letters, digits, and spaces
        """
        pattern = u'^(\u0020|[\u0980-\u09FF])+$'
        return bool(re.match(pattern, row)) and row != ' '

if __name__ == '__main__':
    project_root = Path(__file__).parent.parent
    preprocess = Preprocess(project_root / 'data/', project_root / 'processed_data/')