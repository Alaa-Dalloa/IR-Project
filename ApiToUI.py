# pip install fastapi
# pip install uvicorn
from fastapi import FastAPI
import uvicorn 
from pydantic import BaseModel
from MatchingRanking import MatchingRanking

class QueryRequest(BaseModel):
    query: str

class ApiToUI:
    def __init__(self):
        self.app = FastAPI()

        # first Data Set : antique-collection
        @self.app.post("/send-query")
        async def search(request: QueryRequest):
            ranked_doc_ids, ranked_doc_strings = MatchingRanking.matching_and_ranking(request.query)
            return {"ranked_document_ids": ranked_doc_ids, "ranked_document_strings": ranked_doc_strings}
        
        # Second Data Set : TREC-TOT



        # Service APIs For SOA:
        # __________________________________________ 1 __________________________________________
        # Data Processing
        

        # __________________________________________ 2 __________________________________________
        # Indexing

        # __________________________________________ 3 __________________________________________
        # Matching And Ranking
        
    def run(self):
        print("Done")
        uvicorn.run(self.app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    api_to_ui = ApiToUI()
    api_to_ui.run()