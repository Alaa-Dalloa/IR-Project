import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

class Indexing:

    @staticmethod
    def create_corpus(df,col): 
        # Defining the documents (corpus) dictionary
        corpus = {}
        for i, doc in enumerate(df[col], start=1):
            corpus[f"doc_{i}"] = " ".join(doc)

        df = pd.DataFrame(corpus, index=["Document"])
        df
        return corpus

    @staticmethod
    def vectorizer_docs(corpus):
        documents = list(corpus.values())

        vectorizer = TfidfVectorizer()

        # Fit the vectorizer to the documents
        tfidf_matrix = vectorizer.fit_transform(documents)
        print("Indexing: ")
        print(tfidf_matrix)

        # Store tfidf_matrix in a binary file
        with open('tfidf_matrix_Recreation.bin', 'wb') as file:
            pickle.dump(tfidf_matrix, file)

        # Save the model
        with open('model_Recreation.pkl', 'wb') as file:
            pickle.dump(vectorizer, file)

        # Load tfidf_matrix from the binary file
        with open('tfidf_matrix_Recreation.bin', 'rb') as file:
            tfidf_matrix = pickle.load(file)

        # Load the model
        with open('model_Recreation.pkl', 'rb') as file:
            vectorizer = pickle.load(file)