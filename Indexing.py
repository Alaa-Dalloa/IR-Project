# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# import pickle

# class Indexing:

#     @staticmethod
#     def create_corpus(df,col): 
#         # Defining the documents (corpus) dictionary
#         corpus = {}
#         for i, doc in enumerate(df[col], start=1):
#             corpus[f"doc_{i}"] = " ".join(doc)

#         df = pd.DataFrame(corpus, index=["Document"])
#         df
#         return corpus

#     @staticmethod
#     def vectorizer_docs(corpus):
#         documents = list(corpus.values())

#         vectorizer = TfidfVectorizer()

#         # Fit the vectorizer to the documents
#         tfidf_matrix = vectorizer.fit_transform(documents)
#         print("Indexing: ")
#         print(tfidf_matrix)

#         # Store tfidf_matrix in a binary file
#         with open('tfidf_matrix.bin', 'wb') as file:
#             pickle.dump(tfidf_matrix, file)

#         # Save the model
#         with open('model.pkl', 'wb') as file:
#             pickle.dump(vectorizer, file)

#         # Load tfidf_matrix from the binary file
#         with open('tfidf_matrix.bin', 'rb') as file:
#             tfidf_matrix = pickle.load(file)

#         # Load the model
#         with open('model.pkl', 'rb') as file:
#             vectorizer = pickle.load(file)


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

class Indexing:

    @staticmethod
    def create_corpus(df, col):
        # Defining the documents (corpus) dictionary
        corpus = {}
        for i, doc in enumerate(df[col], start=1):
            corpus[f"doc_{i}"] = " ".join(doc)

        df = pd.DataFrame(corpus, index=["Document"])
        return corpus

    @staticmethod
    def vectorizer_docs(corpus, file_name):
        documents = list(corpus.values())

        vectorizer = TfidfVectorizer()

        # Fit the vectorizer to the documents
        tfidf_matrix = vectorizer.fit_transform(documents)
        print("Indexing: ")
        print(tfidf_matrix)

        # Store tfidf_matrix in a binary file
        tfidf_matrix_file = f"{file_name}_tfidf_matrix.bin"
        with open(tfidf_matrix_file, 'wb') as file:
            pickle.dump(tfidf_matrix, file)

        # Save the model
        model_file = f"{file_name}_model.pkl"
        with open(model_file, 'wb') as file:
            pickle.dump(vectorizer, file)

        return tfidf_matrix_file, model_file

    @staticmethod
    def create_corpus_fromcsv(file_path, col_name):
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Defining the documents (corpus) dictionary
        corpus = {}
        for i, doc in enumerate(df[col_name], start=1):
            corpus[f"doc_{i}"] = doc

        print(corpus)
        return corpus

    @staticmethod
    def vectorizer_docs_fromcsv(corpus, file_name):
        documents = list(corpus.values())

        vectorizer = TfidfVectorizer()

        # Fit the vectorizer to the documents
        tfidf_matrix = vectorizer.fit_transform(documents)
        print("Indexing: ")
        print(tfidf_matrix)

        # Store tfidf_matrix in a binary file
        tfidf_matrix_file = f"{file_name}_tfidf_matrix.bin"
        with open(tfidf_matrix_file, 'wb') as file:
            pickle.dump(tfidf_matrix, file)

        # Save the model
        model_file = f"{file_name}_model.pkl"
        with open(model_file, 'wb') as file:
            pickle.dump(vectorizer, file)

        return tfidf_matrix_file, model_file

