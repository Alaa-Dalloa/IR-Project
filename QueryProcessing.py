# Spell checking
# pip install pyspellchecker 

from spellchecker import SpellChecker
from nltk.tokenize import word_tokenize
from typing import List  # Import the List type from the typing module
import re #to remove stop words
# import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# from nltk import pos_tag
from nltk.corpus import wordnet
# import inflect

class QueryProcessing:
        
    documents = []  # Initialize an empty list to store the processed documents
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # file = open("common_words", "r")
    # fileData = file.read()
    # file.close()
    # stopwords_from_file = re.findall("\S+", fileData)
    # stop_words = set(stopwords_from_file)

    # Lemmatization func.
    # @staticmethod
    # def get_wordnet_pos(tag_parameter):
    #     tag = tag_parameter[0].upper()
    #     tag_dict = {"J": wordnet.ADJ,
    #                 "N": wordnet.NOUN,
    #                 "V": wordnet.VERB,
    #                 "R": wordnet.ADV}    
    #     return tag_dict.get(tag, wordnet.NOUN)

    @staticmethod
    def correct_sentence_spelling(tokens: List[str]) -> List[str]:
        spell = SpellChecker()
        misspelled = spell.unknown(tokens)
        for i, token in enumerate(tokens):
            if token in misspelled:
                corrected = spell.correction(token)
                if corrected is not None:
                    tokens[i] = corrected
        return tokens

    @staticmethod
    def query_processing(query_text):

    # Normalization
        query_text = re.sub(r'\W', ' ', str(query_text)) # Replace non-word characters with a space
        query_text = re.sub(r'\s+', ' ', query_text)  # Remove extra spaces
        # query_text = re.sub(r'\^[a-zA-Z]\s+', ' ', query_text) # Remove caret (^) followed by a letter at the beginning of a word

        # Convert the entire document to lowercase
        query_text = query_text.lower() 

        # Remove Punctuation إزالة علامات الترقيم
        # query_text = query_text.translate(str.maketrans('', '', string.punctuation))
        # query_text = re.sub(r'\s+', ' ', query_text)  # Remove extra spaces

        # Tokenize into words
        query_text_words = word_tokenize(query_text)
        query_text_words = QueryProcessing.correct_sentence_spelling(query_text_words)

        # Lemmatization
        # POS tagging
        # pos_tags = pos_tag(query_text_words)
        # Lemmatization
        # query_text_Lemmatized_words = [QueryProcessing.lemmatizer.lemmatize(word, pos=QueryProcessing.get_wordnet_pos(tag)) for word, tag in pos_tags]
       
        # Stemming      
        query_text_stemmed_words = [QueryProcessing.stemmer.stem(word) for word in query_text_words] # Apply stemming to each word
        
        query_text_Lemmatized_words = [QueryProcessing.lemmatizer.lemmatize(word) for word in query_text_stemmed_words]
       
    # Normalization
        # Remove stopwords from the text 
        query_text_words = [word for word in query_text_Lemmatized_words if word not in stopwords.words('English')]
        # Remove more stopwords from file 
        # query_text_words = [w for w in query_text_words if not w in QueryProcessing.stop_words]

        removed_punctuation_query = [word for word in query_text_words if word.isalnum()]

         # Number To Word 
        # converted_words = []
        # for word in query_text_words:
        #     if word.isdigit():
        #         try:
        #             if int(word) > 999999999: 
        #                 converted_words.append(word)
        #             else:
        #                 converted_word = self.inflect.engine.number_to_words(word)
        #                 converted_words.append(converted_word)
        #         except Exception as e:
        #             print(f"Error converting number: {word}, {e}")
        #             converted_words.append(word)
        #     else:
        #         converted_words.append(word)

        return removed_punctuation_query

