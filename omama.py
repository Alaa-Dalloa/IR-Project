from ConvertToSCV import ConvertToCSV
from DataProcessing import DataProcessing
from DataRepresentation import DataRepresentation
from Indexing import Indexing

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
dataProcessing_instance = DataProcessing()
df = dataProcessing_instance.load_dataset("C:\\Users\\Asus\\Desktop\\New folder IR\\New folder IR\\recreation-collection.csv")
col_name = 'doc'
df = dataProcessing_instance.text_preprocessing(df, col_name)

# __________________________________________ 2 __________________________________________
# Data Representation
# dataRepresentation_instance = DataRepresentation()
# dataRepresentation_instance.count_vectorizer(df,col_name)
# dataRepresentation_instance.VSM_representation(df,col_name)

# __________________________________________ 3 __________________________________________
# Indexing
indexing_instance = Indexing()
corpus = indexing_instance.create_corpus(df,col_name)
indexing_instance.vectorizer_docs(corpus)