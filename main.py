import numpy as np
import pandas as pd
from dataLoader import TwitterDataset
from processor import TextProcessor
from model.tfidf_model import TfIdfModel

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
