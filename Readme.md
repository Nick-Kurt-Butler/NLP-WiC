Project: WiC
Teammates: Nick Butler, Alexa Rinard, Emma King

WiC: the word-in-context dataset
Do two occurrences of a word correspond to the same meaning or not? 

Grading Scale
-----------------
50.0%-53.0% — D

53.1%-60% — C

60%-70% — B

+70% — A

Goal: WiC is used to determine whether two sentences with two different occurrences of the same word correspond to the same meaning or not. 

Accuracy: 66.8%
<p align="center">
  <img align="center" src="https://github.com/Nick-Kurt-Butler/NLP-WiC/blob/main/epoch 2.png"/>
</p>

To Run Code: 

Use the RoBerta.ipynb file to run the finalized code.



Our end model was heavily based upon the code shown by resource 1. We used the code shown by resource 1 and proceeded to change parameters and use our own training, testing, and value data. We went through step-by-step in the code and made alterations to what we believed could be improved, overall focusing on learning how a WiC model can be implemented and how the parameters can be changed to affect the output. 

Example 1:

“Mike wrapped his present up with a pretty bow.”

“Bill proceeded to bow when he saw the king.”

Same Meaning: NO

Example 2:

"We hiked through the Andes mountain range."

"The plains lay just beyond the mountain range."

Same Meaning: YES

# Approaches:

## First Approach - Embedding Layer
- Attempted to build our own version of a network based off of the assigned reading for WiC
- The system has two LSTM layers with 50 units, one for each context side, which concatenates the outputs and passes that to a feedforward layer with 64 neurons, followed by a dropout layer at rate 0.5, and a final one-neuron output layer of sigmoid activation.

## Second Approach - GloVe
- Attempted to use GloVe - researched more into that and realized that it does not work well with WiC
- GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

## Third Approach - DeBerta
- Attempted to use DeBerta 
- DeBerta vs RoBerta: The first is the disentangled attention mechanism, where each word is represented using two vectors that encode its content and position, respectively, and the attention weights among words are computed using disentangled matrices on their contents and relative positions. Second, an enhanced mask decoder is used to incorporate absolute positions in the decoding layer to predict the masked tokens in model pre-training. We show that these two techniques significantly improve the efficiency of model pre-training and performance of downstream tasks. Compared to RoBERTa-Large, a DeBERTa model trained on half of the training data performs consistently better on a wide range of NLP tasks
- Unfortunately, the pre-trained data ‘xlarge-v2’ was too big for our computers to handle and the ‘base’ data refused to load due to an error that is common in users but still unsolved

## Fourth Approach - RoBerta
- Used RoBerta
- While using RoBerta, we have had issues such as a lack of RAM and CPU - resulting in slow running speeds for our epochs and not enough space to run our training 
- Found a resource that showed us how to train our data based off of a WiC model

Resources:

1.) https://github.com/llightts/CSI5138_Project

2.) https://github.com/pytorch/fairseq

3.) https://mccormickml.com/2019/07/22/BERT-fine-tuning/

4.) https://arxiv.org/pdf/1808.09121.pdf

