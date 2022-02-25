import os
from pathlib import Path
import shutil
import pandas as pd
import re

class Preprocess():
    def __init__(self, unprocessed_files_dir, processed_files_dir):
        self.unprocessed_files_dir = unprocessed_files_dir
        self.processed_file_path = processed_files_dir
        os.makedirs(processed_files_dir, exist_ok=True)
        self.unprocessed_files = Path(self.unprocessed_files_dir).glob('**/*.txt')
        self.preprocess_and_save_files()

    def preprocess_and_save_files(self):
        """
        Performs preprocessing on all files in the unprocessed_files_dir
        and saves the processed dataframes to the processed_files_dir
        """
        for file in self.unprocessed_files:
            try:
                file_name = file.name
                file_path = Path(self.processed_file_path) / file_name
                df = self.preprocess_file(file)
                self.save_dataframe_to_file(df, file_path)
            except Exception as e:
                pass

    def preprocess_file(self, file_path):
        """
        Performs preprocessing on a text file and returns a dataframe
        """
        data = pd.read_csv(file_path, sep="\r\n", engine='python', encoding="utf-8", header=None)[0] # reads text file as dataframe
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
        shutil.rmtree(file_path, ignore_errors=True)
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
    project_root = Path(__file__).resolve().parents[1]
    preprocess = Preprocess(project_root / 'data/', project_root / 'processed_data/')
