# Bangla Next Word Prediction

Different next-word prediction models are implemented and trained using Bangla books dataset.

## Necessary steps to setup the project
1. Download and extract the dataset, `bn_dedup.txt` and moved to `data/` directory.
2. Create a virtual environment: `python3 -m venv venv`.
3. Install required packages: `pip install -r requirements.txt`
4. Activate the virtual environment: `source venv/bin/activate`

## Project Structure

### Data Collection
1. We are using Bangla books dataset. The dataset is made from scraping 104453 pages from
[ebanglalibrary.com](https://www.ebanglalibrary.com/). The script and details can be found in
[data-collection](./data-collection/) directory. The directory has extra dependencies specified
in [data-collection/requirements.txt](./data-collection/requirements.txt) file.

### Data Preprocessing
1. [Preprocess.py](./data-preprocessing/Preprocess.py) reads files from `data` directory in the project root,
and after preprocessing stores the preprocessed files to the `processed_data` directory in the project root.
2. Alternatively, the `Preprocess` class can be used from [Preprocess.py](./data-preprocessing/Preprocess.py)
file which takes two string parameters as constructor: `unprocessed_files_dir` which contains unprocessed text
files and `processed_files_dir` where the processed files will be stored. The processed files will have the
same name as the original files.

### Model Training
1. We are training LSTM models with and without attention mechanism, and fine-tuning a
[Bangla GPT2 pretrained model](https://huggingface.co/flax-community/gpt2-bengali).
The training scripts are available in [model-training](./model-training/).
