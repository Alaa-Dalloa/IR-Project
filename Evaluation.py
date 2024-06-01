import random
from collections import defaultdict
from MatchingRanking import MatchingRanking
from itertools import chain
import pandas as pd

class Evaluation:

    @staticmethod
    def load_qrels(file_path):
        qrels = {}
        with open(file_path, 'r') as f:
            for line in f:
                query_id, _, doc_id, relevance = line.strip().split()
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = int(relevance)
        return qrels

    @staticmethod
    def load_queries(file_path):
        queries = {}
        with open(file_path, 'r') as f:
            for line in f:
                query_id, query_text = line.strip().split('\t')
                queries[query_id] = query_text
        return queries
    
    @staticmethod
    def calculate_precision_at_k(qrels, queries, k=10):
        precision_at_k = defaultdict(list)

        for query_id, query_text in queries.items():
            if query_id not in qrels:
                print(f"No relevance judgments found for query {query_id}.")
                continue

            ranked_docs = MatchingRanking.matching_and_ranking(query_text)[:k]
            relevant_docs = 0

            for doc_id in ranked_docs:
                # print("doc_id: " , doc_id)
                if doc_id in qrels[query_id] and qrels[query_id][doc_id] >= 1:
                    relevant_docs += 1

            p_at_k = relevant_docs / k
            precision_at_k[query_id].append(p_at_k)
            print("p_at_k: " , p_at_k)

        avg_precision_at_k = sum(chain(*precision_at_k.values())) / len(precision_at_k)
        return avg_precision_at_k

    @staticmethod
    def calculate_recall_at_k(qrels, queries, k=10):
        recall_at_k = defaultdict(list)

        for query_id, query_text in queries.items():
            if query_id not in qrels:
                print(f"No relevance judgments found for query {query_id}.")
                continue

            ranked_docs = MatchingRanking.matching_and_ranking(query_text)[:k]
            relevant_docs = 0
            total_relevant = sum(qrels[query_id].values())

            for doc_id, relevance in qrels[query_id].items():
                if doc_id in ranked_docs and relevance >= 1:
                    relevant_docs += 1

            r_at_k = relevant_docs / total_relevant
            recall_at_k[query_id].append(r_at_k)

        avg_recall_at_k = sum(chain(*recall_at_k.values())) / len(recall_at_k)
        return avg_recall_at_k
    
    @staticmethod
    def calculate_mean_average_precision(qrels, queries):
        average_precisions = []

        for query_id, query_text in queries.items():
            if query_id not in qrels:
                print(f"No relevance judgments found for query {query_id}.")
                continue

            ranked_docs = MatchingRanking.matching_and_ranking(query_text)
            relevant_docs = 0
            precision_values = []

            for i, doc_id in enumerate(ranked_docs):
                if doc_id in qrels[query_id] and qrels[query_id][doc_id] >= 1:
                    relevant_docs += 1
                    precision = relevant_docs / (i + 1)
                    precision_values.append(precision)

            if relevant_docs > 0:
                average_precision = sum(precision_values) / relevant_docs
                average_precisions.append(average_precision)

        return sum(average_precisions) / len(average_precisions)

    @staticmethod
    def calculate_mean_reciprocal_rank(qrels, queries):
        reciprocal_ranks = []

        for query_id, query_text in queries.items():
            if query_id not in qrels:
                print(f"No relevance judgments found for query {query_id}.")
                continue

            ranked_docs = MatchingRanking.matching_and_ranking(query_text)
            found_relevant = False
            rank = 1

            for doc_id in ranked_docs:
                if doc_id in qrels[query_id] and qrels[query_id][doc_id] >= 1:
                    reciprocal_rank = 1 / rank
                    reciprocal_ranks.append(reciprocal_rank)
                    found_relevant = True
                    break
                rank += 1

            if not found_relevant:
                reciprocal_ranks.append(0.0)

        return sum(reciprocal_ranks) / len(reciprocal_ranks)

qrels = Evaluation.load_qrels("C:\\Users\\DELL\\Desktop\\antique DataSet\\antique-test.qrel")
queries = Evaluation.load_queries("C:\\Users\\DELL\\Desktop\\antique DataSet\\antique-test-queries.txt")

avg_p_at_10 = Evaluation.calculate_precision_at_k(qrels, queries, k=10)
print(f"Average Precision@10: {avg_p_at_10:.4f}")

avg_recall_at_10 = Evaluation.calculate_recall_at_k(qrels, queries, k=10)
print(f"Average Recall@10: {avg_recall_at_10:.4f}")

map_score = Evaluation.calculate_mean_average_precision(qrels, queries)
print(f"Mean Average Precision (MAP): {map_score:.4f}")

mrr_score = Evaluation.calculate_mean_reciprocal_rank(qrels, queries)
print(f"Mean Reciprocal Rank (MRR): {mrr_score:.4f}")