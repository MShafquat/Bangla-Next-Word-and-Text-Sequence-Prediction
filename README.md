# Bangla Next Word and Text Sequence Prediction

Training LSTM and GPT-2 models for Bangla word and text sequence prediction.

## Necessary steps to setup the project
1. Create a virtual environment: `python3 -m venv venv`.
2. Install required packages: `pip install -r requirements.txt`
3. Activate the virtual environment: `source venv/bin/activate`

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
3. Books from renowned authors are manually picked and stored in `processed_data` and used without preprocessing
since they are already clean.

### Model Training
1. We are training an LSTM model, and fine-tuning a [Bangla GPT2 pretrained model](https://huggingface.co/flax-community/gpt2-bengali).
The training notebooks are available in [model-training](./model-training/). Models are generated locally in `models` directory. Our trained model is also released in github.
