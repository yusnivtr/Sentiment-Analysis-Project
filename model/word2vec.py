import torch
import numpy as np
import torch.nn as nn
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report,accuracy_score
from tqdm.auto import tqdm

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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # MLP Classifier
        self.model = MLPClassifier(input_size=vector_size,output_size=4)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),lr=0.001)
        
    def train_word2vec(self,sentences):
        print("Training Word2Vec for", self.word2vec_epochs, "epochs...")
        self.word2vec = Word2Vec(sentences,vector_size=self.vector_size,window=5,min_count=1,workers=4,epochs=self.word2vec_epochs)
        
    def compute_tfidf_weights(self,corpus):
        vectorizer = TfidfVectorizer(tokenizer=lambda x :x,lowercase=False,preprocessor=lambda x : x)
        vectorizer.fit(corpus)
        return {word:vectorizer.idf_[i] for word, i in vectorizer.vocabulary_.items()}
    
    # def tweet2vec(self,tweet,tfidf_weights):
    #     words = [word for word in tweet if word in self.word2vec.wv]
    #     if not words:
    #         return np.zeros(self.vector_size)
        
    #     # Apply TF-IDF
    #     word_vectors = np.array([self.word2vec.wv[word]*tfidf_weights.get(word,1) for word in words])
    #     return np.mean(word_vectors,axis=0).astype(np.float32) # size = vector_size
    
    # def train(self,X_train,y_train,tfidf_weights):
    #     print("Traning MLP")
    #     tmp = np.array([self.tweet2vec(tweet,tfidf_weights) for tweet in X_train],dtype=np.float32)
    #     print(tmp.dtype)
    #     X_train_tensor = torch.tensor(tmp,dtype=torch.float64).to(self.device)
    #     y_train_tensor = torch.tensor(y_train,dtype=torch.long).to(self.device)
        
    #     self.model.train()
    #     for epoch in tqdm(range(self.mlp_epochs)):
    #         outputs = self.model(X_train_tensor)
    #         print(outputs.dtypes)
    #         loss = self.criterion(outputs,y_train_tensor)
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
            
    #         if (epoch%10)==0:
    #             print(f"Epoch [{epoch+1}/{self.mlp_epochs}], Loss: {loss.item():.4f}")
    
    def tweet2vec(self, tweet, tfidf_weights):
        words = [word for word in tweet if word in self.word2vec.wv]
        if not words:
            return np.zeros(self.vector_size, dtype=np.float32)  # Specify float32
        
        # Apply TF-IDF and ensure float32
        word_vectors = []
        for word in words:
            vec = self.word2vec.wv[word].astype(np.float32)  # Convert to float32
            weight = tfidf_weights.get(word, 0.0)
            weighted_vec = vec * weight
            word_vectors.append(weighted_vec)
        
        word_vectors = np.array(word_vectors,dtype=np.float32)
        
        return np.mean(word_vectors, axis=0).astype(np.float32)  # Cast to float32

    def train(self, X_train, y_train, tfidf_weights):
        print("Training MLP")
        tmp = np.array([self.tweet2vec(tweet, tfidf_weights) for tweet in X_train], dtype=np.float32)  # Ensure float32 array
        X_train_tensor = torch.tensor(tmp, dtype=torch.float32).to(self.device)  # Use float32 tensor
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)
        
        self.model.train()
        for epoch in tqdm(range(self.mlp_epochs)):
            outputs = self.model(X_train_tensor)
            loss = self.criterion(outputs, y_train_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (epoch % 10) == 0:
                print(f"Epoch [{epoch+1}/{self.mlp_epochs}], Loss: {loss.item():.4f}")


    def evaluate(self, X_test, y_test, tfidf_weights):
        X_test_vectors = np.array([self.tweet2vec(tweet, tfidf_weights) for tweet in X_test])
        X_test_tensor = torch.tensor(X_test_vectors, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

        print("TF-IDF Weighted Word2Vec + MLP Performance:")
        print(classification_report(y_test, y_pred))
        return accuracy_score(y_test, y_pred)
    
        
    
        

        
        
