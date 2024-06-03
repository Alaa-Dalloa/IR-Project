from MatchingRanking import MatchingRanking
import json

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
    def load_qrels_recreation(file_path):
        qrels = {}
        with open(file_path, 'r') as f:
            for line in f: 
                try:
                    # Attempt to parse each line as JSON
                    data = json.loads(line.strip())  # Parse each line as separate JSON
                    query_id =  str(data['qid'])
                    for doc_id in data['answer_pids']:
                        relevance = 1  # Assuming relevance is always 1 in this format
                        if query_id not in qrels:
                            qrels[query_id] = {}
                        qrels[query_id][doc_id] = relevance
                    relevance = 1  # Assuming relevance is always 1 in this format
                except json.JSONDecodeError:
                    # Handle potential non-JSON lines
                    continue  # Skip to the next line

                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = relevance
        return qrels

    @staticmethod
    def load_queries_recreation(file_path):
        queries = {}
        with open(file_path, 'r') as f:
            for line in f:
                query_id, query_text = line.strip().split('\t')
                queries[query_id] = query_text
        return queries
    
    
    @staticmethod
    def calculate_precision_at_k(qrels, queries, ranked_doc_ids, k=10):
        avg_precision_at_k = 0
        num_queries = 0

        for query_id, query_text in queries.items():
            if query_id not in qrels:
                print(f"No relevance judgments found for query {query_id}.")
                continue

            relevant_docs_at_k = 0
            for i, doc_id in enumerate(ranked_doc_ids[query_id]):
                if i >= k:
                    break
                if doc_id in qrels[query_id] and qrels[query_id][doc_id] >= 1:
                    relevant_docs_at_k += 1

            precision_at_k = relevant_docs_at_k / k
            avg_precision_at_k += precision_at_k
            num_queries += 1

        if num_queries > 0:
            avg_precision_at_k /= num_queries
        return avg_precision_at_k



    @staticmethod
    def calculate_recall_at_k(qrels, queries, ranked_doc_ids, k=10, relevance_threshold=8):
        avg_recall_at_k = 0
        num_queries = 0

        for query_id, query_text in queries.items():
            if query_id not in qrels:
                print(f"No relevance judgments found for query {query_id}.")
                continue

            relevant_docs_at_k = 0
            for i, doc_id in enumerate(ranked_doc_ids[query_id]):
                if i >= k:
                    break
                if doc_id in qrels[query_id] and qrels[query_id][doc_id] >= 1:
                    relevant_docs_at_k += 1

            recall_at_k = relevant_docs_at_k / relevance_threshold
            avg_recall_at_k += recall_at_k
            num_queries += 1

        if num_queries > 0:
            avg_recall_at_k /= num_queries
        return avg_recall_at_k


    @staticmethod
    def calculate_mean_average_precision(qrels, queries, ranked_doc_ids):
        average_precisions = []

        for query_id, query_text in queries.items():
            if query_id not in qrels:
                print(f"No relevance judgments found for query {query_id}.")
                continue

            relevant_docs = 0
            precision_values = []

            for i, doc_id in enumerate(ranked_doc_ids[query_id]):
                if doc_id in qrels[query_id] and qrels[query_id][doc_id] >= 1:
                    relevant_docs += 1
                    precision = relevant_docs / (i + 1)
                    precision_values.append(precision)

            if relevant_docs > 0:
                average_precision = sum(precision_values) / relevant_docs
                average_precisions.append(average_precision)

        return sum(average_precisions) / len(average_precisions)

    @staticmethod
    def calculate_mean_reciprocal_rank(qrels, queries, ranked_doc_ids):
        reciprocal_ranks = []

        for query_id, query_text in queries.items():
            if query_id not in qrels:
                print(f"No relevance judgments found for query {query_id}.")
                continue

            found_relevant = False
            rank = 1

            for doc_id in ranked_doc_ids[query_id]:
                if doc_id in qrels[query_id] and qrels[query_id][doc_id] >= 1:
                    reciprocal_rank = 1 / rank
                    reciprocal_ranks.append(reciprocal_rank)
                    found_relevant = True
                    break
                rank += 1

            if not found_relevant:
                reciprocal_ranks.append(0.0)

        return sum(reciprocal_ranks) / len(reciprocal_ranks)
    




# first data
qrels = Evaluation.load_qrels("C:\\Users\\DELL\\Desktop\\antique DataSet\\antique-test.qrel")
queries = Evaluation.load_queries("C:\\Users\\DELL\\Desktop\\antique DataSet\\antique-test-queries.txt")

ranked_doc_ids = {}
for query_id, query_text in queries.items():
    ranked_doc_ids[query_id] = MatchingRanking.matching_and_ranking(query_text)

avg_p_at_10 = Evaluation.calculate_precision_at_k(qrels, queries, ranked_doc_ids, k=10)
print(f"Average Precision@10: {avg_p_at_10:.4f}")

avg_recall_at_10 = Evaluation.calculate_recall_at_k(qrels, queries, ranked_doc_ids, k=10)
print(f"Average Recall@10: {avg_recall_at_10:.4f}")

map_score = Evaluation.calculate_mean_average_precision(qrels, queries, ranked_doc_ids)
print(f"Mean Average Precision (MAP): {map_score:.4f}")

mrr_score = Evaluation.calculate_mean_reciprocal_rank(qrels, queries, ranked_doc_ids)
print(f"Mean Reciprocal Rank (MRR): {mrr_score:.4f}")



#  Second data
qrels_recreation = Evaluation.load_qrels_recreation("C:\\Users\\DELL\\Desktop\\recreation\\dev\\qas.search.jsonl")
queries_recreation = Evaluation.load_queries_recreation("C:\\Users\\DELL\\Desktop\\recreation\\dev\\questions.search.txt")

ranked_doc_ids_recreation = {}
for query_id, query_text in queries_recreation.items():
    ranked_doc_ids_recreation[query_id] = MatchingRanking.matching_and_ranking(query_text)

avg_p_at_10_recreation = Evaluation.calculate_precision_at_k(qrels_recreation, queries_recreation, ranked_doc_ids_recreation, k=10)
print(f"Average Precision@10 recreation: {avg_p_at_10_recreation:.4f}")

avg_recall_at_10_recreation = Evaluation.calculate_recall_at_k(qrels_recreation, queries_recreation, ranked_doc_ids_recreation, k=10)
print(f"Average Recall@10 recreation: {avg_recall_at_10_recreation:.4f}")

map_score_recreation = Evaluation.calculate_mean_average_precision(qrels_recreation, queries_recreation, ranked_doc_ids_recreation)
print(f"Mean Average Precision (MAP) recreation: {map_score_recreation:.4f}")

mrr_score_recreation = Evaluation.calculate_mean_reciprocal_rank(qrels_recreation, queries_recreation, ranked_doc_ids_recreation)
print(f"Mean Reciprocal Rank (MRR) recreation: {mrr_score_recreation:.4f}")