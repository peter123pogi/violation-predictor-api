import tensorflow as tf
from gensim.models import Word2Vec as w2v
import numpy as np
import re

def tokenized(text):
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    return text.split()

class Model:
    def __init__(self, x_tokenized, y):
        self.tokenized_texts = x_tokenized
        self.y = np.array(y).reshape(-1, 1)
        self.vector_size = 100
        self.word2vec_model = self.train_word2vec()
        self.x = self.vectorize_texts()
        self.model = self.train_logistic_regression()

    def train_word2vec(self):
        return w2v(sentences=self.tokenized_texts, vector_size=self.vector_size, window=5, min_count=1, workers=4)

    def vectorize_texts(self):
        def vectorize(sentence):
            vectors = [self.word2vec_model.wv[word] for word in sentence if word in self.word2vec_model.wv]
            return np.mean(vectors, axis=0) if vectors else np.zeros(self.vector_size)
        return np.array([vectorize(s) for s in self.tokenized_texts])

    def train_logistic_regression(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.vector_size,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(self.x, self.y, epochs=10, batch_size=32, verbose=0)
        return model

    def get_accuracy(self):
        _, acc = self.model.evaluate(self.x, self.y, verbose=0)
        return acc

    def predict(self, text):
        tokens = tokenized(text)
        vec = self.vectorize_texts_for_predict(tokens)
        pred = self.model.predict(np.array([vec]), verbose=0)[0][0]
        return pred
    
    def predict_top_offenses(self, text, top_k=3):
        tokens = tokenized(text)
        vec = self.vectorize_texts_for_predict(tokens)
        probs = self.model.predict(np.array([vec]), verbose=0)[0]
        top_indices = probs.argsort()[-top_k:][::-1]

        # Reverse map to offense_id
        index_to_offense_id = {v: k for k, v in self.offense_id_index_map.items()}
        return [(index_to_offense_id[i], float(probs[i])) for i in top_indices]

    def vectorize_texts_for_predict(self, tokens):
        vectors = [self.word2vec_model.wv[word] for word in tokens if word in self.word2vec_model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.vector_size)
