from DeBERTa import deberta
import torch
from torch import nn
from torch import LongTensor
from torch import tensor
from torch import optim
import numpy as np

import random


class MyModel(torch.nn.Module):
	def __init__(self):
		super().__init__()
    # Your existing model code
		self.deberta = deberta.DeBERTa(pre_trained='xlarge-v2') # Or 'large' 'base-mnli' 'large-mnli' 'xlarge' 'xlarge-mnli' 'xlarge-v2' 'xxlarge-v2'
    # Your existing model code
    # do inilization as before
    #
		self.output_layer = nn.Linear(1536,1)
		self.deberta.apply_state() # Apply the pre-trained model of DeBERTa at the end of the constructor
    #
	def forward(self, input_ids,token_type_ids):
    # The inputs to DeBERTa forward are
    # `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] with the word token indices in the vocabulary
    # `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token types indices selected in [0, 1]. 
    #    Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
    # `attention_mask`: an optional parameter for input mask or attention mask. 
    #   - If it's an input mask, then it will be torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1]. 
    #      It's a mask to be used if the input sequence length is smaller than the max input sequence length in the current batch. 
    #      It's the mask that we typically use for attention when a batch has varying length sentences.
    #   - If it's an attention mask then if will be torch.LongTensor of shape [batch_size, sequence_length, sequence_length]. 
    #      In this case, it's a mask indicate which tokens in the sequence should be attended by other tokens in the sequence. 
    # `output_all_encoded_layers`: whether to output results of all encoder layers, default, True

		encoding = self.deberta(input_ids,token_type_ids = token_type_ids) #  -1 hidden state 0 CLS

		encoding = encoding[0].squeeze(0)

		output = self.output_layer(encoding)
	# linear layer hidden_layerx1
		return output.squeeze(1)

# 2. Change your tokenizer with the the tokenizer built in DeBERta
from DeBERTa import deberta
vocab_path, vocab_type = deberta.load_vocab(pretrained_id='xlarge-v2')
tokenizer = deberta.tokenizers[vocab_type](vocab_path)
# We apply the same schema of special tokens as BERT, e.g. [CLS], [SEP], [MASK]
max_seq_len = 32
tokens = tokenizer.tokenize('Examples input text of DeBERTa')
# Truncate long sequence
tokens = tokens[:max_seq_len -2]
# Add special tokens to the `tokens`
tokens = ['[CLS]'] + tokens + ['[SEP]']
input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_mask = [1]*len(input_ids)
# padding
paddings = max_seq_len-len(input_ids)
input_ids = input_ids + [0]*paddings
input_mask = input_mask + [0]*paddings
features = {
'input_ids': torch.tensor(input_ids, dtype=torch.int),
'input_mask': torch.tensor(input_mask, dtype=torch.int)
}


from json import loads

def load_data(file_name):
	data = []
	with open(file_name) as file:
		for line in file.readlines():
			line = loads(line)
			# change punctuation or case maybe
			data.append(line)
	return data



def sen2list(s):
	return re.findall(r"[\w']+|[.,!?;]",s.lowers())

def data2token(data_point):
	s1 = data_point['sentence1']
	s2 = data_point['sentence2']
	word = data_point['word']

	t1 = tokenizer.tokenize(s1)
	t2 = tokenizer.tokenize(s2)
	word = [tokenizer.tokenize(word)[0]]

	tokens = ["[CLS]"] + word
	tokens =  word

	input_ids = tokenizer.convert_tokens_to_ids(tokens)
	token_type_ids = tokenizer.convert_tokens_to_ids(t1+t2)


	token_type_ids = [token_type_ids,[0]*len(t1)+[1]*len(t2)]

	return LongTensor([input_ids]), LongTensor([token_type_ids])


train_data = load_data('train.jsonl')
test_data = load_data('test.jsonl')
val_data = load_data('val.jsonl')

model = MyModel()

ce = nn.CrossEntropyLoss()
ce = nn.BCELoss()
softmax = nn.Softmax(dim=0)
optimizer = optim.SGD(model.parameters(), lr=0.02)

for param in model.parameters():
	if 1 not in param.shape:
		param.requires_grad = False

sig = nn.Sigmoid()

epochs = 1000
for epoch in range(epochs):

	model.train()

	total_loss = 0
	yep = len(train_data)
	count = 0
	for point in train_data:
		print(str(count)+"/"+str(yep))
		count += 1
		optimizer.zero_grad()

		# a) calculate probs / get an output
		input_ids,token_type_ids = data2token(point)
		y_raw = model(input_ids,token_type_ids)

		#y_hat = softmax(y_raw)

		y = tensor(float(point["label"]))
		# b) compute loss
		y_raw = sig(y_raw)
		loss = ce(y_raw,y.unsqueeze(0))
		total_loss += loss

		# c) get the gradient
		loss.backward()

		# d) update the weights
		optimizer.step()

		print(total_loss/len(train_data))


		model.eval()

		score = 0
		for point in train_data:
			print(score)
			input,token_ids = data2token(point)
			output = model(input,token_ids)
			result = True if output >= 0 else False
			if bool(result) == point["label"]:
				score += 1

		print("epoch:",epoch," train data ",score/len(train_data))

		score = 0
		for point in val_data:
			input,token_ids = data2token(point)
			output = model(input,token_ids)
			result = True if output >= 0 else False
			if bool(result) == point["label"]:
				score += 1

		print("epoch:",epoch," val data ",score/len(val_data))


