#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# In[3]:


get_ipython().system('pip install bs4')
get_ipython().system('pip install contractions')


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


reviews = pd.read_csv('../input/amazon-fine-food-reviews/Reviews.csv')


# In[6]:


# reviews = reviews.iloc[:20000]


# In[7]:


reviews.head()


# In[8]:


reviews.iloc[10].Text


# In[9]:


reviews.iloc[10].Summary


# In[10]:


reviews.info()


# ## Get Total unique Products

# In[11]:


reviews.nunique()


# In[12]:


reviews.isnull().sum()
# ProfileName doesnt matter since UserId and ProfileName is same. So If UserId is present then we should not care about ProfileName


# In[13]:


# users who didnt provide Summary
reviews[reviews.Summary.isnull()].UserId.value_counts()


# In[14]:


# How many users provided, respective ratings
reviews.Score.value_counts()


# In[15]:


reviews.Score.value_counts().plot.bar()


# In[16]:


# Most of the reviews are positive, So ppl do like food , Average of the Score is 4.1
reviews.Score.mean()


# In[17]:


import datetime
lol = lambda x: datetime.datetime.fromtimestamp(x)
reviews['Date'] = reviews.Time.apply(lol)


# In[18]:


reviews['weekDay_Time'] = reviews.Date.dt.weekday
reviews['month'] = reviews.Date.dt.month
reviews['quarter'] = reviews.Date.dt.quarter


# In[19]:


reviews.head()


# In[20]:


sns.countplot(data=reviews, x='weekDay_Time')


# In[21]:


sns.countplot(data=reviews, x='month')


# In[22]:


sns.countplot(data=reviews, x='quarter')


# In[23]:


reviews = reviews[~reviews.Summary.isnull()]
reviews=reviews.reset_index(drop=True)


# ## Lets do Some EDA on the Text Data

# ### Lets preprocess convert everything to Lower Case

# In[24]:


reviews.Text = reviews.Text.str.lower()
reviews.Summary = reviews.Summary.str.lower()


# In[25]:


reviews.head()


# In[26]:


Text_data = reviews.Text.values
Summary = reviews.Summary.values


# ## do we have similar Summariaes?

# In[27]:


reviews[reviews.Text.duplicated()]


# In[28]:



Total_len = len(Summary)
Total_distinct_Summary = len(set(Summary))
print(Total_len)
print(Total_distinct_Summary)


# ## do we have similar Text Data?

# In[29]:



Total_len = len(Text_data)
Total_distinct_Text_data = len(set(Text_data))
print(Total_len)
print(Total_distinct_Text_data)


# ## ok lets drop the similar Text Data

# In[30]:


reviews = reviews.drop_duplicates('Text')
reviews = reviews.reset_index(drop=True)


# In[31]:


Text_data = reviews.Text.values
Summary = reviews.Summary.values


# In[32]:


from wordcloud import WordCloud, STOPWORDS
summaryWordCloud = ' '.join(Summary).lower()
wordcloud2 = WordCloud().generate(summaryWordCloud)
plt.imshow(wordcloud2)


# In[33]:


# TextWordCloud = ' '.join(Text_data).lower()
# wordcloud2 = WordCloud().generate(TextWordCloud)
# plt.imshow(wordcloud2)


# ## Word Cloud for our Data. Looks COOL

# ## Remove HTML Tags

# In[34]:


from bs4 import BeautifulSoup
soup = lambda text:BeautifulSoup(text)
passText = lambda text:soup(text).get_text()

reviews.Text = reviews.Text.apply(passText)


# ## Histogram for the length of reviews and summary

# In[35]:


reviews.Summary.apply(lambda x:len(x.split(' '))).plot(kind='hist')


# In[36]:


reviews.Text.apply(lambda x:len(x.split(' '))).plot(kind='hist')


# In[37]:


reviews.Text.apply(lambda x:len(x.split(' '))).quantile(0.95)


# In[38]:


input_characters = set()
target_characters = set()


# In[39]:


import re


# In[40]:


from nltk.tokenize import TweetTokenizer
tweet = TweetTokenizer()


# In[41]:


import contractions


# In[42]:


sampleReview = reviews.iloc[:20000].copy()

#Fix Contradiction
sampleReview.Text = sampleReview.Text.apply(lambda x: contractions.fix(x))
sampleReview.Summary = sampleReview.Summary.apply(lambda x: contractions.fix(x))

sampleReview.Text = sampleReview.Text.apply(lambda x: tweet.tokenize(x))
sampleReview.Summary = sampleReview.Summary.apply(lambda x: tweet.tokenize(x))


# In[43]:


sampleReview.head()


# In[44]:


sampleReview.Text = sampleReview.Text.apply(lambda tokens: [word for word in tokens if word.isalpha()])
sampleReview.Summary = sampleReview.Summary.apply(lambda tokens: [word for word in tokens if word.isalpha()])


# In[45]:


from nltk.corpus import stopwords
stop_words = stopwords.words()
stop_words = set(stop_words)


# In[46]:


len(sampleReview.Text)


# In[47]:


get_ipython().run_cell_magic('time', '', 'sampleReview.Text = sampleReview.Text.apply(lambda tokens: [word for word in tokens if word not in stop_words][:100])')


# In[48]:


sampleReview.head()


# In[49]:


# from nltk.corpus import wordnet
# sampleReview.Text = sampleReview.Text.apply(lambda tokens: [word for word in tokens if wordnet.synsets(word)])
# sampleReview.Summary = sampleReview.Summary.apply(lambda tokens: [word for word in tokens if wordnet.synsets(word)])


# In[50]:


sampleReview.Summary = sampleReview.Summary.apply(lambda x:['<BOS>']+x+['<EOS>'])


# In[51]:


sampleReview.Text = sampleReview.Text.apply(lambda x:' '.join(x))
sampleReview.Summary = sampleReview.Summary.apply(lambda x:' '.join(x))


# In[52]:


sampleReview.Summary


# In[53]:


input_texts = sampleReview.Text.values
target_texts = list(sampleReview.Summary.values)


# In[54]:


sampleReview.to_csv('filtered_data.csv', index=False)


# In[55]:


from keras.preprocessing.text import Tokenizer
VOCAB_SIZE = 50000

tokenizerText = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizerText.fit_on_texts(input_texts)

tokenizerSummary = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizerSummary.fit_on_texts(target_texts)


def text2seq(encoder_text, decoder_text, VOCAB_SIZE):
  encoder_sequences = tokenizerText.texts_to_sequences(encoder_text)
  decoder_sequences = tokenizerSummary.texts_to_sequences(decoder_text)
  
  return encoder_sequences, decoder_sequences

encoder_sequences, decoder_sequences = text2seq(input_texts, target_texts, VOCAB_SIZE) 


# In[56]:


textVocabSize = len(tokenizerText.word_index)
summaryVocabSize = len(tokenizerSummary.word_index)
textVocabSize, summaryVocabSize


# In[57]:


def vocab_creater(text_lists, VOCAB_SIZE, tokenizer):

  dictionary = tokenizer.word_index
  
  word2idx = {}
  idx2word = {}
  for k, v in dictionary.items():
      if v < VOCAB_SIZE:
          word2idx[k] = v
          idx2word[v] = k
      if v >= VOCAB_SIZE-1:
          continue
          
  return word2idx, idx2word

word2idxText, idx2wordText = vocab_creater(input_texts, textVocabSize, tokenizerText)
word2idxSummary, idx2wordSummary = vocab_creater(target_texts, summaryVocabSize, tokenizerSummary)


# In[58]:


EMBEDDING_DIM=100
maxLenText=100
maxLenSummary=40


# In[59]:


from keras.preprocessing.sequence import pad_sequences

def padding(encoder_sequences, decoder_sequences, maxLenText, maxLenSummary):
  
  encoder_input_data = pad_sequences(encoder_sequences, maxlen=maxLenText, dtype='int32', padding='post', truncating='post')
  decoder_input_data = pad_sequences(decoder_sequences, maxlen=maxLenSummary, dtype='int32', padding='post', truncating='post')
  
  return encoder_input_data, decoder_input_data

encoder_input_data, decoder_input_data = padding(encoder_sequences, decoder_sequences,maxLenText, maxLenSummary)


# In[60]:


encoder_input_data.shape


# In[61]:


def glove_100d_dictionary():
  embeddings_index = {}
  f = open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt')
  for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
  f.close()
  return embeddings_index


# In[62]:


embeddings_index = glove_100d_dictionary()


# In[63]:


def embedding_matrix_creater(embedding_dimention, tokenizer):
  embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dimention))
  for word, i in tokenizer.word_index.items():
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
          # words not found in embedding index will be all-zeros.
          embedding_matrix[i] = embedding_vector
  return embedding_matrix


# In[64]:


embeddingMatrixText = embedding_matrix_creater(100, tokenizerText)
embeddingMatrixSummary = embedding_matrix_creater(100, tokenizerSummary)


# In[65]:


embeddingMatrixText.shape


# In[66]:


from keras.layers import Embedding
encoder_embedding_layer = Embedding(input_dim = textVocabSize+1, 
                                    output_dim = 100,
                                    input_length = maxLenText,
                                    mask_zero=True,
                                    weights = [embeddingMatrixText],
                                    trainable = False)
decoder_embedding_layer = Embedding(input_dim = summaryVocabSize+1, 
                                    output_dim = 100,
                                    input_length = maxLenSummary,
                                    mask_zero=True,
                                    weights = [embeddingMatrixSummary],
                                    trainable = False)


# In[67]:


# sampleReview.Text = sampleReview.Text.apply(lambda tokens: [contractions[word] for word in tokens if word.isalpha()])
# sampleReview.Summary = sampleReview.Summary.apply(lambda tokens: [contractions[word] for word in tokens if word.isalpha()])


# In[68]:



from numpy.random import seed
seed(1)

from sklearn.model_selection import train_test_split
import logging

import matplotlib.pyplot as plt
import pandas as pd
import pydot


import keras
from keras import backend as k
k.set_learning_phase(1)
from keras.preprocessing.text import Tokenizer
from keras import initializers
from keras.optimizers import RMSprop,Adam
from keras.models import Sequential,Model
from keras.layers import Dense,LSTM,Dropout,Input,Activation,Add,concatenate, Embedding, RepeatVector
from keras.layers.advanced_activations import LeakyReLU,PReLU
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam

from keras.layers import TimeDistributed


# In[69]:


MAX_LEN = 100
EMBEDDING_DIM = 100
HIDDEN_UNITS = 300
textVocabSize = textVocabSize+1
summaryVocabSize = summaryVocabSize+1 
LEARNING_RATE = 0.002
BATCH_SIZE = 8
EPOCHS = 5


# In[70]:


# input_characters = sorted(list(input_characters))+[' ']
# target_characters = sorted(list(target_characters))+[' ']
# num_encoder_tokens = len(encoder_input_data)
# num_decoder_tokens = len(decoder_input_data)
num_encoder_tokens = textVocabSize
num_decoder_tokens = summaryVocabSize

max_encoder_seq_length = max([len(txt) for txt in encoder_input_data])
max_decoder_seq_length = max([len(txt) for txt in decoder_input_data])

print("Number of samples:", len(input_texts))
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)


# In[71]:


# encoder_input_data = np.zeros(
#     (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
# )
# decoder_input_data = np.zeros(
#     (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
# )
# decoder_target_data = np.zeros(
#     (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
# )


# In[72]:


# for i, seqs in enumerate(encoder_input_data):
#     for j, seq in enumerate(seqs):
#         decoder_target_data[i, j, seq] = 1.0


# In[73]:


from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional


# In[74]:



"""
Chatbot Inspired Encoder-Decoder-seq2seq
"""
encoder_inputs = Input(shape=(maxLenText, ), dtype='int32',)
encoder_embedding = encoder_embedding_layer(encoder_inputs)
encoder_LSTM = LSTM(HIDDEN_UNITS, return_state=True,return_sequences=True)
encoder_outputs1, state_h, state_c = encoder_LSTM(encoder_embedding)

encoder_lstm2 = LSTM(HIDDEN_UNITS,return_sequences=True,return_state=True) 
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_outputs1) 

encoder_lstm3= LSTM(HIDDEN_UNITS, return_state=True) 
encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2) 

decoder_inputs = Input(shape=(maxLenSummary, ), dtype='int32',)
decoder_embedding = decoder_embedding_layer(decoder_inputs)

decoder_LSTM = LSTM(HIDDEN_UNITS, return_state=True, return_sequences=True)
decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])

# attn_out, attn_states = tf.keras.layers.Attention()([encoder_outputs, decoder_outputs]) 

# decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])


# dense_layer = Dense(VOCAB_SIZE, activation='softmax')
decoder_time = TimeDistributed(Dense(summaryVocabSize, activation='softmax'))
outputs = decoder_time(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], outputs)


# In[75]:


rmsprop = RMSprop(lr=0.01, clipnorm=1.)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=["accuracy"])


# In[76]:


model.summary()


# In[77]:


import numpy as np1
num_samples = len(decoder_sequences)
decoder_output_data = np.zeros((num_samples, maxLenSummary, summaryVocabSize), dtype="int32")


# In[78]:


num_samples


# In[79]:


for i, seqs in enumerate(decoder_sequences):
    for j, seq in enumerate(seqs):
        if j > 0:
            decoder_output_data[i][j-1][seq] = 1


# In[80]:


art_train, art_test, sum_train, sum_test = train_test_split(encoder_input_data, decoder_input_data, test_size=0.2)
train_num = art_train.shape[0]
target_train = decoder_output_data[:train_num]
target_test = decoder_output_data[train_num:]


# In[81]:


import tensorflow as tf


# In[82]:


class My_Custom_Generator(keras.utils.Sequence) :
    def __init__(self, art_train, sum_train, decoder_output, batch_size) :
        self.art = art_train
        self.sum = sum_train
        self.decoder = decoder_output
        self.batch_size = batch_size

    def __len__(self) :
        return (np.ceil(len(self.art) / float(self.batch_size))).astype(np.int)
  
    def __getitem__(self, idx) :
        batch_x1 = self.art[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_x2 = self.sum[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.decoder[idx * self.batch_size : (idx+1) * self.batch_size]
        return [np.array(batch_x1),np.array(batch_x2)],np.array(batch_y)


# In[83]:


batch_size = 64

my_training_batch_generator = My_Custom_Generator(art_train, sum_train, target_train, batch_size)
my_validation_batch_generator = My_Custom_Generator(art_test, sum_test, target_test,batch_size)


# In[84]:


callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=4)
model.fit_generator(generator=my_training_batch_generator,
                   steps_per_epoch = int(16000 // batch_size),
                   epochs = 100,
                   verbose = 1,callbacks=[callback])


# In[85]:


model.save_weights('nmt_weights_100epochs.h5')


# In[86]:


model.load_weights('nmt_weights_100epochs.h5')


# In[87]:


encoder_states = [state_h, state_c]
encoder_model = Model(encoder_inputs, encoder_states)

thought_input = [Input(shape=(HIDDEN_UNITS, )), Input(shape=(HIDDEN_UNITS, ))]

decoder_embedding = decoder_embedding_layer(decoder_inputs)

decoder_outputss, state_h, state_c = decoder_LSTM(decoder_embedding, initial_state=thought_input)
decoder_states = [state_h, state_c]

decoder_outputs = decoder_time(decoder_outputss)

decoder_model = Model(inputs=[decoder_inputs]+thought_input, outputs=[decoder_outputs]+decoder_states)


# In[88]:


# encoder_states = [encoder_outputs, state_h, state_c]
# encoder_model = Model(encoder_inputs, encoder_states)


# decoder_embedding = decoder_embedding_layer(decoder_inputs)
# decoder_outputss, state_h, state_c = decoder_LSTM(decoder_embedding, initial_state=thought_input)
# decoder_states = [state_h, state_c]

# decoder_outputs = decoder_time(decoder_outputss)

# decoder_model = Model(inputs=[decoder_inputs]+thought_input, outputs=[decoder_outputs]+decoder_states)


# In[89]:


decoder_model.summary()


# In[90]:


def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((100,1))
    target_seq[0, 0] = word2idxSummary['bos']
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word =idx2wordSummary[sampled_token_index]
        decoded_sentence += ' '+ sampled_word

        if (sampled_word == 'eos' or
           len(decoded_sentence) > 40):
            stop_condition = True

        target_seq = np.zeros((100,1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]
    return decoded_sentence


# In[91]:


print("Text->",tokenizerText.sequences_to_texts([art_train[10]]))
print("\n\n\n")
print("Summary->",tokenizerSummary.sequences_to_texts([art_test[10]]))
print("\n\n\n")
print("using model->",decode_sequence([art_test[10]]))

