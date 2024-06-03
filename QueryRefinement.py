from spellchecker import SpellChecker
import csv
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List


class QueryRefinement:

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
    def compare_with_history(query, history_file="history.csv"):
        """Compares query with past queries, preserving word order."""

        # Load history queries from CSV
        history_queries = []
        with open(history_file, "r") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header row
            for row in reader:
                history_queries.append(row[0])  # Store entire query string

        # Convert query to lowercase words
        query_words = query.lower().split()

        # Find matching queries (preserve order)
        similar_queries = []
        for history_query in history_queries:
            if any(word in query_words for word in history_query.lower().split()):  # Check for at least one match
                similar_queries.append(history_query)

        return similar_queries

    def query_refinement(self, query):
        """Refines the query by performing spell checking and comparing with history."""

        list_queries_refinement = []

        # 1. Spell check the query
        corrected_query = self.correct_sentence_spelling(query.split())
        corrected_query = " ".join(corrected_query)
        list_queries_refinement.append(corrected_query)

        # 2. Compare with query history
        similar_queries = self.compare_with_history(corrected_query)
        list_queries_refinement.extend(similar_queries)

        return list_queries_refinement

# refiner = QueryRefinement()
# query = "helllo world"
# refined_queries = refiner.query_refinement(query)
# print(refined_queries)
