import json
import os
from http.client import responses

from botocore.exceptions import ClientError

from vector_db import VectorDB
from utils import create_bedrock_client


def get_response(query):
    # Get Context
    vector_db = VectorDB(pinecone_api_key=os.getenv("PINECONE_API_KEY"), pinecone_env=os.getenv("PINECONE_ENV"),
                         index_name=os.getenv("PINECONE_INDEX_NAME"), dimension=int(os.getenv("DIMENSION")), metric=os.getenv("METRIC"), cloud=os.getenv("PINECONE_CLOUD"))
    search_results = vector_db(query, top_k=5)
    # Prepare Prompt
    prompt = f"""
    Question: {query}
    Context: {search_results}
    """
    body_content = {
        "prompt": prompt
    }
    # Get Response
    try:
        client = create_bedrock_client()
        response = client.invoke_model(
            modelId="meta.llama3-70b-instruct-v1:0",
            body=json.dumps(body_content),
        )
        return response['body'].read()
    except ClientError as e:
        print(f"Error: {e}")
        return None


if __name__ == '__main__':
    query = "I eat a lot do I have cancer ?"
    responses = get_response(query)
    print(responses)
