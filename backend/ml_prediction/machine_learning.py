import numpy as np
import tensorflow as tf
from pathlib import Path
from transformers import AutoTokenizer, TFGPT2LMHeadModel
import pickle

project_root = Path(__file__).resolve().parent
gpt2_model_dir = project_root / 'mlmodels/bn_gpt2'
latm_model_dir = project_root / 'mlmodels/bn_lstm'

MAX_WORDS = 10000
MAX_SEQUENCE_LENGTH = 200

# tokenizer = AutoTokenizer.from_pretrained(gpt2_model_dir)
# model = TFGPT2LMHeadModel.from_pretrained(str(gpt2_model_dir))
# # model = tf.keras.models.load_model(gpt2_model_dir)

# def generate_text(text, model, tokenizer):
#     input_ids = tokenizer.encode(text, return_tensors='tf')
#     outputs = model.predict(input_ids).logits

#     print("Next most probable tokens:\n" + 100 * '-')
#     for i in range(outputs.shape[1]):
#         pred_id = np.argmax(outputs[:, i, :]).item()
#         print(tokenizer.decode(pred_id))
    
#     beam_outputs = model.generate(
#         input_ids,
#         max_length=100,
#         num_beams=5,
#         no_repeat_ngram_size=2,
#         num_return_sequences=5,
#         early_stopping=True,
#         do_sample=True,
#         top_k=50,
#         top_p=0.95,
#     )

#     print("Beam Output:\n" + 100 * '-')
#     for i, beam_output in enumerate(beam_outputs):
#         print("{}: {}".format(i, tokenizer.decode(
#             beam_output, skip_special_tokens=True)))

# text = input("Enter text: ")
# generate_text(text, model, tokenizer)


def predict_lstm(text: str):
    model = tf.keras.models.load_model(str(latm_model_dir))
    with open(str(latm_model_dir / f'tokenizer_{MAX_WORDS}_words.pickle'), 'rb') as f:
        tokenizer = pickle.load(f)

    # tokenize the text
    tokenized_sequence = tokenizer.texts_to_sequences([text])
    # pre-pad with 0's to make it of size MAX_SEQUENCE_LENGTH - 1
    input_sequences = tf.keras.preprocessing.sequence.pad_sequences(tokenized_sequence,
                                                                maxlen=MAX_SEQUENCE_LENGTH - 1,
                                                                padding='pre')
    # predict using model
    predictions = model.predict(input_sequences)
    ids = np.argsort(predictions, axis=1)[:,-5:] # indices of the top 5 predictions
    # print next word with score

    res = []

    for id in ids[0]:
        # print(tokenizer.index_word[id], "->", predictions[:, id].squeeze())
        res.append(tokenizer.index_word[id])

    return res