# Bangla Next Word Prediction

Different next-word prediction models are implemented and trained using [Bengali OSCAR Corpus](https://www.kaggle.com/tapash39/bengali-oscar-corpus).

## Necessary steps to setup the project
1. Download and extract the dataset, `bn_dedup.txt` and moved to `data/` directory.
2. Create a virtual environment: `python3 -m venv venv`.
3. Install required packages: `pip install -r requirements.txt`
4. Activate the virtual environment: `source venv/bin/activate`

## Project Structure
### Data Preprocessing
1. [Preprocess.py](./data-preprocessing/Preprocess.py) reads files from `data` directory in the project root,
and after preprocessing stores the preprocessed files to the `processed_data` directory in the project root.
2. Additionally, the `Preprocess` class can be used from [Preprocess.py](./data-preprocessing/Preprocess.py)
file which takes two string parameters as constructor: `unprocessed_files_dir` which contains unprocessed text
files and `processed_files_dir` where the processed files will be stored. The processed files will have the
same name as the original files.
