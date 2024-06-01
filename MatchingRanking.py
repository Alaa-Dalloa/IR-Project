import pickle
from QueryProcessing import QueryProcessing
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class MatchingRanking:

    # Convert the documents into a list of strings
    df = pd.read_csv('antique-collection.csv',encoding='latin-1')
    document_strings = []
    for document in df['doc']:
        document_strings.append(document)
    # print(f"The size of the document_strings is: {len(document_strings)}")


    document_IDs = []
    for document_id in df['num']:
        document_IDs.append(document_id)
    # print(f"The size of the document_IDs is: {len(document_IDs)}")

    @staticmethod
    def matching_and_ranking(query_text):
        # Load tfidf_matrix from the binary file
        with open('tfidf_matrix.bin', 'rb') as file:
            tfidf_matrix = pickle.load(file)
            # print("tfidf_matrix: ")
            # print(tfidf_matrix.shape)

        # Load the model
        with open('model.pkl', 'rb') as file:
            vectorizer = pickle.load(file)
            
        # Preprocess the query
        queryProcessing_instance = QueryProcessing()
        query = queryProcessing_instance.query_processing(query_text) # Pass the query to preprocess_query func. in section preprocess query
        # print("query: ")
        # print(query)

        # Transform the query into a vector
        query_string = ' '.join(query)  # Convert the list of tokens back into a single string

        # Convert the query document to a TF-IDF vector
        query_vector = vectorizer.transform([query_string])

        # print("query_vector: ")
        # print(query_vector)

        # Calculate cosine similarity between the query vector and document vectors
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        # for similarity in cosine_similarities[0:4000]:
        #     print(similarity)

        # Rank the results
        document_indices = np.argsort(cosine_similarities)[::-1]
        # print(f"The size of the document_indices is: {len(document_indices)}") # 403666
        k = 10
        ranked_documents = document_indices[:k]  # Choose the top-k most similar documents
        # print(f"The size of the ranked_documents is: {len(ranked_documents)}") # 10

        ranked_document_ids = []
        ranked_document_strings = []

        # Print the ranked documents
        for idx in ranked_documents:
            # print("---------- idx: " , idx , " --------------")
            # print("Document id:", MatchingRanking.document_IDs[idx])
            # print("Document:", MatchingRanking.document_strings[idx])
            # print("Cosine Similarity:", cosine_similarities[idx])
            # print("----------------------------")
            ranked_document_ids.append(MatchingRanking.document_IDs[idx])
            ranked_document_strings.append(MatchingRanking.document_strings[idx])

        # return ranked_document_ids, ranked_document_strings # in Api
        
        return ranked_document_ids # in evaluation 

# Call the matching_and_ranking() method to execute the code
# MatchingRanking.matching_and_ranking("small group of ")