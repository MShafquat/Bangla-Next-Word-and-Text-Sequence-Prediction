import numpy as np
import tensorflow as tf
from pathlib import Path
from transformers import AutoTokenizer, TFGPT2LMHeadModel

project_root = Path(__file__).resolve().parent
gpt2_model_dir = project_root / 'mlmodels/bn_gpt2'

tokenizer = AutoTokenizer.from_pretrained(gpt2_model_dir)
model = TFGPT2LMHeadModel.from_pretrained(str(gpt2_model_dir))
# model = tf.keras.models.load_model(gpt2_model_dir)

def generate_text(text, model, tokenizer):
    input_ids = tokenizer.encode(text, return_tensors='tf')
    outputs = model.predict(input_ids).logits

    print("Next most probable tokens:\n" + 100 * '-')
    for i in range(outputs.shape[1]):
        pred_id = np.argmax(outputs[:, i, :]).item()
        print(tokenizer.decode(pred_id))
    
    beam_outputs = model.generate(
        input_ids,
        max_length=100,
        num_beams=5,
        no_repeat_ngram_size=2,
        num_return_sequences=5,
        early_stopping=True,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )

    print("Beam Output:\n" + 100 * '-')
    for i, beam_output in enumerate(beam_outputs):
        print("{}: {}".format(i, tokenizer.decode(
            beam_output, skip_special_tokens=True)))

text = input("Enter text: ")
generate_text(text, model, tokenizer)