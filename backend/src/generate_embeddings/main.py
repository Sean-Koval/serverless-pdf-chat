import os, json
import boto3
from aws_lambda_powertools import Logger
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS

from pypdf import PdfReader
import re
import time
from typing import Optional, Union
from langchain.tools import BaseTool

from langchain.llms.bedrock import Bedrock
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter  # If text splitting is required

from langchain.prompts import PromptTemplate  # For custom prompt templates
#from langchain.document_loaders import PyPDFLoader
#from langchain_community.document_loaders import PyPDFLoader


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

class SymDocumentChatBot:

    def __init__(self, embeddings=None, llm=None, index=None, memory=None, user=None, file_name=None):
        self.embeddings = embeddings
        self.llm = llm
        self.index = index
        self.memory = memory
        self.user = user
        self.file_name = file_name

    def summarization(self):

        """
        Summarize a document using a map-reduce approach with LLM.
        """
        start_time = time.time()  # Start timing

        if not self.llm:
            return "LLM not initialized."

        try:
            # Assuming s3, PyPDFLoader, and necessary prompts are defined outside this snippet.
            document_path = f"/tmp/{self.file_name}"
            #s3_start = time.time()  # Start timing S3 download
            #s3.download_file(BUCKET, f"{self.user}/{self.file_name}/{self.file_name}", document_path)
            #s3_end = time.time()  # End timing S3 download
            #logger.info(f"S3 Download Time: {s3_end - s3_start} seconds")
            
            load_start = time.time()  # Start timing PDF load and split
            pages = self._load_and_split_pdf(document_path)
            load_end = time.time()  # End timing PDF load and split
            logger.info(f"Load and Split PDF Time: {load_end - load_start} seconds")
            
            map_reduce_start = time.time()
            # map reduce and return final step in intermediate steps (output)
            map_reduce_output = self._perform_map_reduce_summarization(pages)
            map_reduce_end = time.time()
            logger.info(f"Map Reduce Summarization Time: {map_reduce_end - map_reduce_start} seconds")

            final_summary = map_reduce_output["intermediate_steps"][-1]  # Assuming final summary is in the last step
            end_time = time.time()
            logger.info(f"Total Summarization Time: {end_time - start_time} seconds")

            return final_summary
        except Exception as e:
            return f"Error during summarization: {str(e)}"
        

    def _load_and_split_pdf(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        return loader.load_and_split()

    def _perform_map_reduce_summarization(self, pages):
        map_prompt_template = """
                        Write a summary of this chunk of text that includes the main points and any important details.
                        {text}
                        """
        map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

        # combine_prompt_template = """
        #                 Write a concise summary of the following text delimited by triple backquotes.
        #                 Return your response in bullet points which covers the key points of the text.
        #                 ```{text}```
        #                 BULLET POINT SUMMARY:
        #                """
        
        combine_prompt_template = """
                        Write a concise summary of the following text delimited by triple backquotes.
                        Return your response in paragraph format which covers the key points of the text.
                        ```{text}```
                        SUMMARY:
                        """

        combine_prompt = PromptTemplate(
            template=combine_prompt_template, input_variables=["text"]
        )

        map_reduce_chain = load_summarize_chain(
            llm=self.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            return_intermediate_steps=True,
        )

        return map_reduce_chain({"input_documents": pages})


@logger.inject_lambda_context(log_event=True)
def lambda_handler(event, context):
    event_body = json.loads(event["Records"][0]["body"])
    document_id = event_body["documentid"]
    user_id = event_body["user"]
    key = event_body["key"]
    file_name_full = key.split("/")[-1]

    set_doc_status(user_id, document_id, "PROCESSING")

    s3.download_file(BUCKET, key, f"/tmp/{file_name_full}")

    loader = PyPDFLoader(f"/tmp/{file_name_full}")

    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
    )

    embeddings = BedrockEmbeddings(
        model_id="cohere.embed-english-v3",
        client=bedrock_runtime,
        region_name="us-east-1",
    )

    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=embeddings,
    )

    index_from_loader = index_creator.from_loaders([loader])

    index_from_loader.vectorstore.save_local("/tmp")

    s3.upload_file(
        "/tmp/index.faiss", BUCKET, f"{user_id}/{file_name_full}/index.faiss"
    )
    s3.upload_file("/tmp/index.pkl", BUCKET, f"{user_id}/{file_name_full}/index.pkl")

    """
    SUMMARIZATION LOGIC IF LENGTH(DOC)<3 PAGES:
    """ 

    # store data in pdfReader 
    pdfreader = PdfReader(f"/tmp/{file_name_full}") 
    number_of_pages = len(pdfreader.pages)
    if number_of_pages < 5:
        llm = Bedrock(
            model_id="anthropic.claude-v2", client=bedrock_runtime, region_name="us-east-1"
        )

        chat_bot = SymDocumentChatBot(
                                    embeddings=embeddings, 
                                    llm=llm, 
                                    user=user_id,
                                    file_name=file_name_full,
        )
        summary_response = chat_bot.summarization()

        summary_file_path = "/tmp/summary.txt"  # Temporary file path
        with open(summary_file_path, 'w') as file:
            file.write(summary_response)  # Write the summary string to the file

        # NOTE: added in as temp solution for gateway summary generation (timeout)    
        s3.upload_file("/tmp/summary.txt", BUCKET, f"{user_id}/{file_name_full}/summary.txt")

    set_doc_status(user_id, document_id, "READY")
