# # Import Necessary Libraries
# # pandas for data manipulation, 
# # nltk for natural language processing, 
# # and sklearn for machine learning tasks.

# import pandas as pd
# import re #to remove stop words
# import nltk
# from nltk.stem import PorterStemmer
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
# from nltk import pos_tag
# from nltk.corpus import wordnet
# from nltk.corpus import stopwords
# import string
# import numpy as np
# import inflect

# class DataProcessing:

#     documents = []  # Initialize an empty list to store the processed documents
#     stemmer = PorterStemmer()
#     lemmatizer = WordNetLemmatizer()
#     inflect_engine = inflect.engine()

#     file = open("common_words", "r")
#     fileData = file.read()
#     file.close()
#     stopwords_from_file = re.findall("\S+", fileData)
#     stop_words = set(stopwords_from_file)

#     # Lemmatization func.
#     @staticmethod
#     def get_wordnet_pos(tag_parameter):
#         tag = tag_parameter[0].upper()
#         tag_dict = {"J": wordnet.ADJ,
#                     "N": wordnet.NOUN,
#                     "V": wordnet.VERB,
#                     "R": wordnet.ADV}    
#         return tag_dict.get(tag, wordnet.NOUN)


#     # Load Dataset func.
#     @staticmethod
#     def load_dataset(name):
#         df = pd.read_csv(name,encoding='latin-1')
#         # Create DataFrame
#         dataFrameAntique = pd.DataFrame(df)
#         print(dataFrameAntique)
#         return df

#     # Text Preprocessing

#     def text_preprocessing(self, df):
#         i=0

#         for document in df['doc']:
            
#             print(i)
#             i = i+1
#             # Cleaning
#             document = re.sub(r'\W', ' ', str(document)) # Replace non-word characters with a space
#             document = re.sub(r'\s+', ' ', document)  # Remove extra spaces
#             document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) # Remove caret (^) followed by a letter at the beginning of a word

#             # Normalization
#             # Convert the entire document to lowercase
#             document = document.lower() 

#             # Remove Punctuation إزالة علامات الترقيم     
#             document = document.translate(str.maketrans('', '', string.punctuation))
#             document = re.sub(r'\s+', ' ', document)  # Remove extra spaces
            
#             # Tokenize into words
#             words = word_tokenize(document)

#             # Lemmatization
#             # POS tagging
#             pos_tags = pos_tag(words)
#             lemmatized_words = [self.lemmatizer.lemmatize(word, pos=self.get_wordnet_pos(tag)) for word, tag in pos_tags]

#             # Stemming      
#             stemmed_words = [self.stemmer.stem(word) for word in lemmatized_words] # Apply stemming to each word
        
#             # Remove stopwords from the text 
#             without_Stop_words = [word for word in stemmed_words if word not in stopwords.words('English') and word.isalpha()]
#             # Remove more stopwords from file 
#             without_Stop_words = [w for w in without_Stop_words if not w in self.stop_words]

#             # Number To Word 
#             converted_words = []
#             for word in without_Stop_words:
#                 if word.isdigit():
#                     try:
#                         if int(word) > 999999999: 
#                             converted_words.append(word)
#                         else:
#                             converted_word = self.inflect.engine.number_to_words(word)
#                             converted_words.append(converted_word)
#                     except Exception as e:
#                         print(f"Error converting number: {word}, {e}")
#                         converted_words.append(word)
#                 else:
#                     converted_words.append(word)

#             # Add the processed document to the list of documents    
#             self.documents.append(converted_words) 
            
#         # Update the 'doc' column in the DataFrame with the processed documents
#         df['doc'] = self.documents[:len(df)]

#         # Print the updated DataFrame, See resault
#         print("Data After Processing: ")
#         print(df['doc'])
#         return df

###########################################    2    ###############################################
import pandas as pd
import re #to remove stop words
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import numpy as np

class DataProcessing:

    documents = []  # Initialize an empty list to store the processed documents
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Load Dataset func.
    @staticmethod
    def load_dataset(name):
        df = pd.read_csv(name,encoding='latin-1')
        # Create DataFrame
        dataFrameAntique = pd.DataFrame(df)
        print(dataFrameAntique)
        return df

    # Text Preprocessing

    def text_preprocessing(self, df, col_name):

        for document in df[col_name]:

            # Cleaning
            document = re.sub(r'\W', ' ', str(document)) # Replace non-word characters with a space
            document = re.sub(r'\s+', ' ', document)  # Remove extra spaces
         
            # Normalization
            # Convert the entire document to lowercase
            document = document.lower() 

            # Tokenize into words
            words = word_tokenize(document)

            # Stemming      
            stemmed_words = [self.stemmer.stem(word) for word in words] # Apply stemming to each word
            
            # Lemmatization
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in stemmed_words]

            # Remove stopwords from the text 
            without_stop_words = [word for word in lemmatized_words if word not in stopwords.words('English')]
        
            # Remove punctuation
            removed_punctuation_words = [word for word in without_stop_words if word.isalnum()]

            # Add the processed document to the list of documents    
            self.documents.append(removed_punctuation_words) 
            
        # Update the 'doc' column in the DataFrame with the processed documents
        df[col_name] = self.documents[:len(df)]

        # Print the updated DataFrame, See resault
        print("Data After Processing: ")
        print(df[col_name])
        return df