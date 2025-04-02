import numpy as np
import pandas as pd
from dataLoader import TwitterDataset
from processor import TextProcessor
from model.tfidf_model import TfIdfModel
from model.word2vec import Word2VecMLP
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


if __name__ == '__main__':
    
    train_loader = TwitterDataset('data/twitter_training.csv')
    train_loader.load_data()
    train_loader.prepocess_label()
    df_train = train_loader.df

    test_loader = TwitterDataset('data/twitter_validation.csv')
    test_loader.load_data()
    test_loader.prepocess_label()
    df_test = test_loader.df


    preprocessor = TextProcessor()
    df_train,df_test = preprocessor.preproces_dataframe(df_train),preprocessor.preproces_dataframe(df_test)

    # TFIDF - Model
    tfidf_model = TfIdfModel()
    tfidf_model.train(df_train['clean text'],df_train['label'])
    tfidf_model.evaluate(df_test['clean text'],df_test['label'])

    # Word2Vec MLP
    df_train["tokenized"] = df_train["clean text"].apply(lambda x: word_tokenize(x.lower()))
    df_test['tokenized'] = df_test['clean text'].apply(lambda x: word_tokenize(x.lower()))

    X_train,y_train = df_train['tokenized'].tolist(),df_train['label'].tolist()
    X_test,y_test = df_test['tokenized'].tolist(),df_test['label'].tolist()


    word2vec_mlp = Word2VecMLP(vector_size=400,mlp_epochs=100,word2vec_epochs=50)
    word2vec_mlp.train_word2vec(X_train)
    tfidf_weights = word2vec_mlp.compute_tfidf_weights(X_train)

    word2vec_mlp.train(X_train,y_train,tfidf_weights)
    word2vec_mlp.evaluate(X_test,y_test,tfidf_weights)

