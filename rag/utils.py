import os

import boto3
from dotenv import load_dotenv


def get_bedrock_key():
    load_dotenv()
    return os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"), os.getenv("AWS_SESSION_TOKEN")


def create_bedrock_client():
    aws_access_key_id, aws_secret_access_key, aws_session_token = get_bedrock_key()
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token
    )
    return session.client("bedrock-runtime", region_name="us-east-1")
