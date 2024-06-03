# pip install fastapi
# pip install uvicorn
# pip install python-multipart
from fastapi import FastAPI, File, UploadFile
import uvicorn
from dataclasses import dataclass
from MatchingRanking import MatchingRanking
import pandas as pd
from DataProcessing import DataProcessing
from Indexing import Indexing
from QueryProcessing import QueryProcessing
from io import BytesIO
import os
import csv
from QueryRefinement import QueryRefinement 
# CORS configuration
from fastapi.middleware.cors import CORSMiddleware

# Define allowed origins (adjust for production)
origins = ["*"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

@dataclass
class SearchQueryRequest:
    query: str

class ApiToUI:
    def __init__(self):
        self.app = app

        # Service APIs For SOA:
        # __________________________________________ 1 __________________________________________
        # Data Processing
        @self.app.post("/data-process")
        async def process_csv(file: UploadFile = File(...)):
            try:
                df = pd.read_csv(file.file)

                # Process the DataFrame
                dataProcessing_instance = DataProcessing()
                df = dataProcessing_instance.text_preprocessing(df, 'doc')
                dataProcessing_instance.save_processed_data(df, file.filename)

                # Return the processed DataFrame
                return {"processed_data": df.to_csv(index=False)}

            except Exception as e:
                # Handle the exception and return an appropriate response
                return {"error": str(e)}, 500

        # __________________________________________ 2 __________________________________________
        # Indexing
        @self.app.post("/indexing")
        async def indexing_csv(file: UploadFile = File(...)):
        
            df = pd.read_csv(file.file)

            file_path = file.filename
            col_name = 'doc'
            indexing_instance = Indexing()
            corpus = indexing_instance.create_corpus_fromcsv(file_path, col_name)
            tfidf_matrix_file, model_file = indexing_instance.vectorizer_docs_fromcsv(corpus, file.filename.split('.')[0])

            response = {
                "tfidf_matrix_file": tfidf_matrix_file,
                "model_file": model_file,
            }

            return response

        # __________________________________________ 3 __________________________________________
        # Query Processing
        @self.app.post("/query-processing")
        async def queryprocessing(request: SearchQueryRequest):
            
            queryProcessing_instance = QueryProcessing()
            query = queryProcessing_instance.query_processing(request.query) 

            return query

        # __________________________________________ 4 __________________________________________
        # Matching And Ranking
        @self.app.post("/matching-ranking")
        async def matching_and_ranking(request: dict):
            try:
                # Extract the required parameters from the request
                csv_file_name = request.get("csv_file_name")
                tfidf_matrix_file = request.get("tfidf_matrix_file")
                model_file = request.get("model_file")
                query = request.get("query")

                # Initialize the MatchingRanking instance
                matching_ranking_instance = MatchingRanking(csv_file_name, tfidf_matrix_file, model_file)

                # Perform matching and ranking
                ranked_doc_ids, ranked_doc_strings = matching_ranking_instance.matching_and_ranking(query)

                # Create history.csv if it doesn't exist
                if not os.path.exists("history.csv"):
                    with open("history.csv", "w", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(["Query"])  # Write header row

                # Append query to history.csv
                with open("history.csv", "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([query])

                return {"ranked_document_ids": ranked_doc_ids, "ranked_document_strings": ranked_doc_strings}

            except Exception as e:
                # Handle the exception and return an appropriate response
                return {"error": str(e)}, 500

        # __________________________________________ * __________________________________________
        # Refine Query            
        @self.app.post("/refine-query")
        async def refine_query(request: SearchQueryRequest):
            # Refines the provided query using spell checking and history comparison.

            query_refinement_instance = QueryRefinement()
            temp_query = request.query
            refined_queries = query_refinement_instance.query_refinement(temp_query)

            return {"refined_queries": refined_queries}


    def run(self):
        print("Done")
        uvicorn.run(self.app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    api_to_ui = ApiToUI()
    api_to_ui.run()