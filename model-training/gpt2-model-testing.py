from pathlib import Path
from transformers import AutoTokenizer, TFGPT2LMHeadModel
import numpy as np

# get data directory
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / 'processed_data/'
files = [str(file) for file in Path(data_dir).glob('**/*.txt')]
model_dir = project_root / 'models/bn_gpt2/'

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = TFGPT2LMHeadModel.from_pretrained(str(model_dir))

text = "শেখ হাসিনা"
# encoding the input text
input_ids = tokenizer.encode(text, return_tensors='tf')
print(input_ids)
# getting out output
outputs = model.predict(input_ids).logits[:, -3:, :]

for i in range(3):
    pred_id = np.argmax(outputs[:, -i, :]).item()
    print(tokenizer.decode(pred_id))
# print(tokenizer.decode(output))

beam_outputs = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,
    no_repeat_ngram_size=2,
    num_return_sequences=5,
    early_stopping=True
)

print("Output:\n" + 100 * '-')
for i, beam_output in enumerate(beam_outputs):
    print("{}: {}".format(i, tokenizer.decode(
        beam_output, skip_special_tokens=True)))
