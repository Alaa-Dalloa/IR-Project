from ConvertToSCV import ConvertToCSV
from DataProcessing import DataProcessing
from DataRepresentation import DataRepresentation
from Indexing import Indexing
from QueryProcessing import QueryProcessing
from MatchingRanking import MatchingRanking

# NOTE:
# on file APIs there is API for each Service

# first data:
# Convert Data File From txt to CSV
# toCSV = ConvertToCSV()
# toCSV.convert_to_csv("antique-collection.txt")

# second data:
# Convert Data File From tsv to CSV
# fromTSVtoCSV = ConvertToCSV()
# fromTSVtoCSV.convert_from_tsv("C:\\Users\\DELL\\Desktop\\recreation\\dev\\collection.tsv")

# __________________________________________ 1 __________________________________________
# Data Processing

# dataProcessing_instance = DataProcessing()
# df = dataProcessing_instance.load_dataset("try.csv")
# col_name = 'doc'
# df = dataProcessing_instance.text_preprocessing(df, col_name)
# dataProcessing_instance.save_processed_data(df, "try.csv")

# __________________________________________ 2 __________________________________________
# Data Representation

# dataRepresentation_instance = DataRepresentation()
# dataRepresentation_instance.count_vectorizer(df,col_name)
# dataRepresentation_instance.VSM_representation(df,col_name)

# __________________________________________ 3 __________________________________________
# Indexing

# indexing_instance = Indexing()
# corpus = indexing_instance.create_corpus(df,col_name)
# indexing_instance.vectorizer_docs(corpus, "try")

# file_path = 'C:\\Users\\DELL\\Desktop\\New folder IR\\New IR Project\\try-processed.csv'
# col_name = 'doc'
# indexing_instance = Indexing()
# corpus = indexing_instance.create_corpus_fromcsv(file_path, col_name)
# tfidf_matrix_file, model_file = indexing_instance.vectorizer_docs_fromcsv(corpus, "try-processed")

# __________________________________________ 4 __________________________________________
# Query Processing

# query_text = "samll group to do play"
# queryProcessing_instance = QueryProcessing()
# query = queryProcessing_instance.query_processing(query_text) 
# print("query: ")
# print(query)

# __________________________________________ 5 __________________________________________
# Matching & Ranking 

# matching_ranking_instance = MatchingRanking('recreation-collection.csv', 'tfidf_matrix_recreation.bin', 'model_recreation.pkl')
# matching_ranking_instance = MatchingRanking('antique-collection.csv', 'tfidf_matrix.bin', 'model.pkl')
# ranked_document_ids, ranked_document_strings = matching_ranking_instance.matching_and_ranking("real world we dont use most of it")
# print("Ranked Document IDs:", ranked_document_ids)
# print("Ranked Document Strings:", ranked_document_strings)