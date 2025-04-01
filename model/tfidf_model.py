from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

class TfIdfModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(max_iter=500)
        
    def train(self,X_train,y_train):
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_tfidf,y_train)
        
    def evaluate(self,X_test,y_test):
        X_test_tfidf = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_tfidf)
        print("Tf-Idf + Logistic Regression Model: ")
        print(classification_report(y_true=y_test,y_pred=y_pred))
        return accuracy_score(y_pred=y_pred,y_true=y_test)
    
