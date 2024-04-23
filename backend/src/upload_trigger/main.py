import os, json
from datetime import datetime
import boto3
import PyPDF2
import shortuuid
import urllib
from docx import Document

from aws_lambda_powertools import Logger

DOCUMENT_TABLE = os.environ["DOCUMENT_TABLE"]
MEMORY_TABLE = os.environ["MEMORY_TABLE"]
QUEUE = os.environ["QUEUE"]
BUCKET = os.environ["BUCKET"]


ddb = boto3.resource("dynamodb")
document_table = ddb.Table(DOCUMENT_TABLE)
memory_table = ddb.Table(MEMORY_TABLE)
sqs = boto3.client("sqs")
s3 = boto3.client("s3")
logger = Logger()


@logger.inject_lambda_context(log_event=True)
def lambda_handler(event, context):
    key = urllib.parse.unquote_plus(event["Records"][0]["s3"]["object"]["key"])
    if key.startswith("summaries/"):
        logger.info(f"skipping processing for summary text file: {key}")
        return {
            "statusCode": 200,
            "body": json.dumps(f"Skpped file in summaries dir: {key}")
        }
        
    split = key.split("/")
    user_id = split[0]
    file_name = split[1]

    document_id = shortuuid.uuid()

    s3.download_file(BUCKET, key, f"/tmp/{file_name}")

    # with open(f"/tmp/{file_name}", "rb") as f:
    #     reader = PyPDF2.PdfReader(f)
    #     pages = str(len(reader.pages))
    _, file_extension = os.path.splitext(file_name)

    try:
        if file_extension.lower() == '.pdf':
            with open(f"/tmp/{file_name}", "rb") as file:
                reader = PyPDF2.PdfReader(file)
                pages = str(len(reader.pages))
                #page_count = str(pages)
        elif file_extension.lower() == '.txt':
            with open(f"/tmp/{file_name}", "r") as file:
                content = file.readlines()
                pages = str(len(content))
                #page_count = str(pages)
        elif file_extension.lower() == '.docx':
            doc = Document(f"/tmp/{file_name}")
            num_paragraphs = len(doc.paragraphs)
            pages = str(num_paragraphs) # calculating number of paragraphs not pages

        else:
            raise ValueError("Unsupported file type")

        conversation_id = shortuuid.uuid()

        timestamp = datetime.utcnow()
        timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        document = {
            "userid": user_id,
            "documentid": document_id,
            "filename": file_name,
            "created": timestamp_str,
            "pages": pages,
            "filesize": str(event["Records"][0]["s3"]["object"]["size"]),
            "docstatus": "UPLOADED",
            "conversations": [],
        }

        conversation = {"conversationid": conversation_id, "created": timestamp_str}
        document["conversations"].append(conversation)

        document_table.put_item(Item=document)

        conversation = {"SessionId": conversation_id, "History": []}
        memory_table.put_item(Item=conversation)

        message = {
            "documentid": document_id,
            "key": key,
            "user": user_id,
        }
        sqs.send_message(QueueUrl=QUEUE, MessageBody=json.dumps(message))
        return {
            'statusCode': 200,
            'body': f"Document processed successfully with {pages} pages."
        }

    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        return {
            'statusCode': 500,
            'body': f"Error uploading file: {str(e)}"
        }

