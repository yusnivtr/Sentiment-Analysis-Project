import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

class TextProcessor():
    def __init__(self):
        self.stop_words = stopwords.words('english')
    
    def clean_text(self,text):
        text = text.lower()        
        text = re.sub(r'http\S+|https\S+|www\S+','',text)
        text = re.sub(r'\W'," ",text)
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        return " ".join(tokens)
    
    def preproces_dataframe(self,df):
        df['clean text'] = df['text'].apply(self.clean_text)
        return df
    

    

        
        
        