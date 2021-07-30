from flask import Flask, request
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow import keras as tf_keras 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import common

app = Flask(__name__)
app.debug = True

# This is all set up work
app.logger.info("model is being loaded")
model = tf_keras.models.load_model("jira_open_data_classifier.h5")

app.logger.info("dataset being loaded")
df = pd.read_csv('./JIRA_OPEN_DATA_LARGESET_PROCESSED.csv')

app.logger.info("data being prepared")
df['labels'] = df['priority'].map({'Optional': 0, 'Trivial': 1, 'Minor': 2, 'Major': 3, 'Blocker': 4, 'Critical': 5})
Y = df['labels'].values
df_train = df['features']
Ytrain = Y

app.logger.info("converting sentances to sequences")
MAX_VOCAB_SIZE = 20000
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(df_train)
sequences_train = tokenizer.texts_to_sequences(df_train)

app.logger.info("get word -> integer mapping")
word2idx = tokenizer.word_index
V = len(word2idx)
app.logger.info('Found %s unique tokens.' % V)

app.logger.info("pad sequences so that we get a N x T matrix")
data_train = pad_sequences(sequences_train)
app.logger.info('Shape of data train tensor: %s', str(data_train.shape))

app.logger.info("get sequence length")
T = data_train.shape[1]

# http://localhost:5000/api_predict
@app.route('/api_predict', methods=["GET", "POST"])
def api_predict():
    if request.method == "GET":
        return "Please send Post Request"
    elif request.method == "POST":
        data = request.get_json()
        app.logger.info("%s was obtained", str(data))
        
        title = data['title']
        description = data['description'] 
        app.logger.info("description = %s", str(description))
        app.logger.info("title = %s", str(title))
        
        sentence = str(title) + " " + str(description)
        app.logger.info("sentence = %s", sentence)
        test_seq = tokenizer.texts_to_sequences(sentence)
        test_padded = pad_sequences(test_seq, maxlen=T)
        
        predictclass = model.predict_classes(test_padded)
        prediction = common.name_labels[predictclass[0]]
        app.logger.info("predictclass = %s  prediction = %s", predictclass[0], prediction)
        
        return str(prediction)

# use this to test on localhost:5000
#app.run()

# use this to run in a docker container
app.run(host='0.0.0.0', port='33')
