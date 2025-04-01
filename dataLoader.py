import kagglehub
import nltk
import re
import os
import pandas as pd

class TwitterDataset:
    def __init__(self,file_path):
        self.file_path = file_path
        self.df = None
        
    def load_data(self):
        data = pd.read_csv(os.path.join(os.getcwd(),self.file_path))
        header = ['label','text']
        self.df = pd.DataFrame(data.iloc[:,2:4])
        self.df.columns = header
        
    def prepocess_label(self):
        label_mapping = {"Negative":0,"Positive":1,"Neutral":2,"Irrelevant":3}
        self.df['label'] = self.df['label'].map(label_mapping)
        self.df.dropna(inplace=True)
        self.df.drop_duplicates(inplace=True)
        
        