from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
 # Represent the data using the Vector Space Model (VSM)
from sklearn.feature_extraction.text import TfidfVectorizer

class DataRepresentation:

    @staticmethod
    def count_vectorizer(df,col):
            
        # Create an instance of CountVectorizer
        vectorizer = CountVectorizer()

        # Fit the vectorizer to your document strings
        document_strings = [' '.join(doc) for doc in df[col]]
        document_vectors = vectorizer.fit_transform(document_strings)

        print("count_vectorizer: ")
        print(document_vectors)

    # VSM Representation
    @staticmethod
    def VSM_representation(df,col):

        # Convert the preprocessed documents into a list of strings
        document_strings = []
        for document in df[col]:
            if isinstance(document, list):
                document_strings.append(' '.join(document))
            elif isinstance(document, str):
                document_strings.append(document)

        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer()

        # Fit the vectorizer on the document strings and transform them into vectors
        document_vectors = vectorizer.fit_transform(document_strings)

        # Print the document vectors
        print("VSM_representation: ")
        print(document_vectors)
         