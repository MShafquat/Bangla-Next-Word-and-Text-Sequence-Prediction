## Model Training

We have trained an LSTM model and fine-tuned a GPT-2 model on our scraped Bangla book dataset.
There are two notebooks available in this directory for training the two models:
1. [LSTM Training](./LSTM%20Training.ipynb)
2. [GPT2 Training](./GPT-2%20Training.ipynb)
The notebooks contain all the details.

### TODO

We have used only 100 files in total for training and testing due to memory limitations.
We need to use `generator` functions to create dataset to train on all the files without exceeding the memory.
This will increase the efficiency of the models, but will take more time to train the model.

We haven't preprocessed the data to train the model. So although the model gives generated texts that are more realistic,
it gives punctuations and other symbols which may not be much useful for keyboard's suggestions.

So we need to use generator functions and preprocess the dataset for further improvement of the models.
