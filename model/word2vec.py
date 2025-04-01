import torch
import numpy as np
import torch.nn as nn
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

class MLPClassifier(nn.Module):
    def __init__(self, input_size,hidden_units=256,output_size=4):
        super().__init__()
        self.fc1 = nn.Linear(input_size,hidden_units)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units,hidden_units//2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_units//2,output_size)
        
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
class Word2VecMLP:
    def __init__(self,vector_size=300,min_count=1,window=5,word2vec_epochs=10,mlp_epochs=10,lr=0.001):
        self.vector_size = vector_size
        self.min_count = min_count
        self.window = window
        self.word2vec = None
        self.word2vec_epochs = word2vec_epochs
        self.mlp_epochs = mlp_epochs
        
        # MLP Classifier
        self.model = MLPClassifier(input_size=vector_size,output_size=4)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),lr=self.lr)
        
    def train_word2vec(self,sentences):
        print("Training Word2Vec for", self.word2vec_epochs, "epochs...")
        self.word2vec = Word2Vec(sentences,vector_size=self.vector_size,window=5,min_count=1,workers=4,epochs=self.word2vec_epochs)
        
    def compute_tfidf_weights(self,corpus):
        vectorizer = TfidfVectorizer(tokenizer=lambda x :x,lowercase=False,preprocessor=lambda x : x)
        vectorizer.fit(corpus)
        return {word:vectorizer.idf_[i] for word, i in vectorizer.vocabulary_.items()}
    
    def tweet2vec(self,tweet,tfidf_weights):
        words = [word for word in tweet if word in self.word2vec.wv]
        if not words:
            return np.zeros(self.vector_size)
        
        # Apply TF-IDF
        word_vectors = np.array([self.word2vec.wv[word]*tfidf_weights.get(word) for word in words])
        return np.mean(word_vectors,axis=0) # size = vector_size
    
    
    
    
        
    
        

        
        
