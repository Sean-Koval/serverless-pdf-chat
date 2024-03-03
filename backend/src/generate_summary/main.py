import os
import json
import boto3
import backoff
import ratelimit
import warnings
from logging import Logger
import PyPDF2
from pathlib import Path as p
import pandas as pd
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from tqdm import tqdm
from langchain.llms import Bedrock # Assuming this is the correct import for Bedrock
#from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings # Update this import based on actual usage
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import Bedrock  # Adjust based on actual LLMs used
from langchain.chains import ConversationalRetrievalChain, MapReduceDocumentsChain, LLMChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate  # For custom prompt templates
from langchain.text_splitter import CharacterTextSplitter  # If text splitting is required
from langchain.chains.summarize import load_summarize_chain


DOCUMENT_TABLE = os.environ["DOCUMENT_TABLE"]
BUCKET = os.environ["BUCKET"]

s3 = boto3.client("s3")
ddb = boto3.resource("dynamodb")
document_table = ddb.Table(DOCUMENT_TABLE)
logger = Logger()


def set_doc_status(user_id, document_id, status):
    document_table.update_item(
        Key={"userid": user_id, "documentid": document_id},
        UpdateExpression="SET docstatus = :docstatus",
        ExpressionAttributeValues={":docstatus": status},
    )


@logger.inject_lambda_context(log_event=True)
def lambda_handler(event, context):
    event_body = json.loads(event["Records"][0]["body"])
    document_id = event_body["documentid"]
    user_id = event_body["user"]
    key = event_body["key"]
    file_name_full = key.split("/")[-1]

    set_doc_status(user_id, document_id, "Generating Summary")

    s3.download_file(BUCKET, key, f"/tmp/{file_name_full}")


    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
    )
        
    llm = Bedrock(
        model_id="anthropic.claude-v2", client=bedrock_runtime, region_name="us-east-1"
    )

    """
    Load and process file
    """
    loader = PyPDFLoader(f"/tmp/{file_name_full}")
    pages = loader.load_and_split()


    """
    Prompt template creation
    """
    map_prompt_template = """
                      Write a summary of this chunk of text that includes the main points and any important details.
                      {text}
                      """
    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    combine_prompt_template = """
                      Write a concise summary of the following text delimited by triple backquotes.
                      Return your response in bullet points which covers the key points of the text.
                      ```{text}```
                      BULLET POINT SUMMARY:
                      """
    
    combine_prompt = PromptTemplate(
        template=combine_prompt_template, input_variables=["text"]
    )

    """
    Create chains - processing logic
    """
    map_reduce_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        return_intermediate_steps=True,
    )
    # TODO: do I want to just have this upload the document and then a sep lambda for returning responses or
    # do I want this to return the response, but first check if the document with the summary exists
    # TODO: do I want this to exist as a sep feature outside of the chat ask, or just when asking for a summary of the doc this is produced?
    """
    Invoke the summarization chain(s)
    """
    map_reduce_output = map_reduce_chain({"input_documents": pages})

    s3.upload_file("/tmp/document_summary.txt", BUCKET, f"{user_id}/{file_name_full}/document_summary.txt")

    set_doc_status(user_id, document_id, "Summary READY")

    logger.info(map_reduce_output)

