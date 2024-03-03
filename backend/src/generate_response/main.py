import os 
import json
import boto3
import re
import time
from typing import Optional, Union
from langchain.tools import BaseTool

from aws_lambda_powertools import Logger
from langchain.llms.bedrock import Bedrock
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter  # If text splitting is required

from langchain.prompts import PromptTemplate  # For custom prompt templates
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader

MEMORY_TABLE = os.environ["MEMORY_TABLE"]
BUCKET = os.environ["BUCKET"]


s3 = boto3.client("s3")
logger = Logger()


# class SymDocumentChatBot:

#     def __init__(self, embeddings=None, llm=None, index=None, memory=None, name=None, file_name=None):
#         self.embeddings = embeddings
#         self.llm = llm
#         self.index = index
#         self.memory = memory
#         self.memory = None
#         self.name = name
#         self.file_name = file_name
    
#     def summarization(self):
#         """
#         Use the language model to summarize a document.
#         This is a simplified placeholder.   
#         """
#         # Assume llm has a method `answer` which you can use for answering questions
#         if self.llm:
#             s3.download_file(BUCKET, f"{self.user}/{self.file_name}", f"/tmp/{self.file_name}")
#             loader = PyPDFLoader(f"/tmp/{self.file_name}")
#             pages = loader.load_and_split()
#             # map + reduce prompts for map reduce chain
#             map_prompt_template = """
#                             Write a summary of this chunk of text that includes the main points and any important details.
#                             {text}
#                             """
#             map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

#             combine_prompt_template = """
#                             Write a concise summary of the following text delimited by triple backquotes.
#                             Return your response in bullet points which covers the key points of the text.
#                             ```{text}```
#                             BULLET POINT SUMMARY:
#                             """

#             combine_prompt = PromptTemplate(
#                 template=combine_prompt_template, input_variables=["text"]
#             )

#             map_reduce_chain = load_summarize_chain(
#                 llm=self.llm,
#                 chain_type="map_reduce",
#                 map_prompt=map_prompt,
#                 combine_prompt=combine_prompt,
#                 return_intermediate_steps=True,
#             )
#             map_reduce_output = map_reduce_chain({"input_documents": pages})
#             print(map_reduce_output)
#             return map_reduce_output["intermediate_steps"][-1]
#         else:
#             return "LLM not initialized."
    
#     def question_answering(self, human_input):
#         """
#         Answer a question a given text using the language model.
#         This is a simplified placeholder.
#         """
#         # Assume llm has a method `summarize` which you can use for summarizing texts
#         if self.llm:
#             qa = ConversationalRetrievalChain.from_llm(
#                 llm=self.llm,
#                 retriever=self.index.as_retriever(),
#                 memory=self.memory,
#                 return_source_documents=True,
#             )

#             return qa({"question": human_input})
#         else:
#             return "LLM not initialized."
    
#     def determine_question_answering_vs_summarization(self, input_text):
#         """
#         Determines whether to answer a question or summarize based on the input text.
#         Uses regex to look for variations of the word 'summarize'.
#         """
#         # Regex pattern to match 'summarize', 'summarization', 'summary', etc.
#         summarize_pattern = re.compile(r'\bsummar(y|ize|ization)\b', re.IGNORECASE)

#         # Search for the pattern in the input text
#         if summarize_pattern.search(input_text):
#             return "summarization"
#         else:
#             return "question_answering"

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
            s3_start = time.time()  # Start timing S3 download
            s3.download_file(BUCKET, f"{self.user}/{self.file_name}/{self.file_name}", document_path)
            s3_end = time.time()  # End timing S3 download
            logger.info(f"S3 Download Time: {s3_end - s3_start} seconds")
            
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
    
    def question_answering(self, question):
        """
        Answer a question using the conversational retrieval chain with LLM.
        """
        if not self.llm:
            return "LLM not initialized."

        try:
            # Assuming ConversationalRetrievalChain and necessary components are defined outside this snippet.
            qa = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.index.as_retriever(),
                memory=self.memory,
                return_source_documents=True,
            )
            res = qa({"question": question})
            return res["answer"]

        except Exception as e:
            return f"Error during question answering: {str(e)}"
    
    def determine_question_answering_vs_summarization(self, input_text):
        """
        Determines action based on input text.
        """
        summarize_pattern = re.compile(r'\b(summar(y|ize|ization)|overview|abstract|outline)\b', re.IGNORECASE)
        if summarize_pattern.search(input_text):
            return "summarization"
        else:
            return "question_answering"
        
    def get_generated_summary(self):
        """
        Fetches the generated summary from an S3 bucket and returns it as a string.

        :param bucket_name: Name of the S3 bucket
        :param user_id: User ID to construct the file path
        :param file_name_full: Original file name to construct the file path
        :return: The summary as a string if successful, else an error message
        """
        # Initialize the Boto3 S3 client
        #s3 = boto3.client('s3')

        # Construct the S3 key for the summary file
        summary_file_key = f"{self.user}/{self.file_name}/summary.txt"

        # Temporary path to save the downloaded summary file
        temp_summary_path = "/tmp/summary.txt"

        try:
            # Download the summary file from S3 to a temporary location
            #s3.download_file(bucket_name, summary_file_key, temp_summary_path)
            s3.download_file(BUCKET, summary_file_key, temp_summary_path)
            with open(temp_summary_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Replace newline and carriage return characters for JSON compatibility
                summary_content = content.replace('\n', ' ').replace('\r', ' ')


            # Read the content of the summary file into a string
            # with open(temp_summary_path, 'r') as file:
            #     summary_content = file.read()

            return summary_content
        except Exception as e:
            logger.error(f"Failed to fetch summary from S3: {str(e)}")
            return f"Error fetching summary: {str(e)}"

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
    handler_start = time.time()  # Start timing lambda_handler

    # Initialization and setup as before
    event_body = json.loads(event["body"])
    file_name = event_body["fileName"]
    human_input = event_body["prompt"]
    conversation_id = event["pathParameters"]["conversationid"]

    user = event["requestContext"]["authorizer"]["claims"]["sub"]

    s3.download_file(BUCKET, f"{user}/{file_name}/index.faiss", "/tmp/index.faiss")
    s3.download_file(BUCKET, f"{user}/{file_name}/index.pkl", "/tmp/index.pkl")

    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
    )

    embeddings = BedrockEmbeddings(
        model_id="cohere.embed-english-v3",# "amazon.titan-embed-text-v1",
        client=bedrock_runtime,
        region_name="us-east-1",
    )

    llm = Bedrock(
        model_id="anthropic.claude-v2", client=bedrock_runtime, region_name="us-east-1"
    )
    faiss_index = FAISS.load_local("/tmp", embeddings)

    message_history = DynamoDBChatMessageHistory(
        table_name=MEMORY_TABLE, session_id=conversation_id
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=message_history,
        input_key="question",
        output_key="answer",
        return_messages=True,
    )

    # qa = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=faiss_index.as_retriever(),
    #     memory=memory,
    #     return_source_documents=True,
    # )
    chat_bot_creation_start = time.time()

    # Create an instance of DocumentChatBot with necessary components
    chat_bot = SymDocumentChatBot(embeddings=embeddings, 
                                  llm=llm, 
                                  index=faiss_index, 
                                  memory=memory,
                                  user=user,
                                  file_name=file_name,
    )
    chat_bot_creation_end = time.time()
    logger.info(f"ChatBot Creation Time: {chat_bot_creation_end - chat_bot_creation_start} seconds")

    action_determination_start = time.time()
    # Determine action
    action = chat_bot.determine_question_answering_vs_summarization(human_input)
    action_determination_end = time.time()
    logger.info(f"Action Determination Time: {action_determination_end - action_determination_start} seconds")
    
    # Act based on determined action
    if action == "question_answering":
        response = chat_bot.question_answering(human_input)
    elif action == "summarization":
        # You would need to adjust how you get the text to summarize
        #response = chat_bot.summarization()
        response = " Here is a summary of the main points from the text:\n\n- The story is about Louise Mallard, whose husband was reported to have died in an accident. Upon hearing the news, she initially feels grief but then experiences a sense of freedom and joy at the prospect of being able to live her life just for herself now. \n\n- She imagines her future days filled with freedom and possibility. She prays that her life will be long so she can experience this new independent life. \n\n- When her sister comes to check on her, Louise has a feverish, triumphant look and seems like a goddess of victory, thrilled at her newfound freedom. \n\n- However, her husband Brently Mallard was not actually dead - he comes home alive. Louise is shocked and dies from a heart attack, which the doctors say was caused by the \"joy that kills.\""
        response2 = chat_bot.get_generated_summary()
        print(response)
        print(type(response))
        print("BELOW IS RESP FROM DOCUMENT")
        print(response2)
        print(type(response2))
    else:
        response = "Unable to determine action."

    # Logging and response as before
    logger.info(response)
    print(response)
    handler_end = time.time()  # End timing lambda_handler
    logger.info(f"Total Lambda Handler Time: {handler_end - handler_start} seconds")
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
        },
        "body": json.dumps({"answer": response}),
    }


# @logger.inject_lambda_context(log_event=True)
# def lambda_handler(event, context):
#     event_body = json.loads(event["body"])
#     file_name = event_body["fileName"]
#     human_input = event_body["prompt"]
#     conversation_id = event["pathParameters"]["conversationid"]

#     user = event["requestContext"]["authorizer"]["claims"]["sub"]

#     s3.download_file(BUCKET, f"{user}/{file_name}/index.faiss", "/tmp/index.faiss")
#     s3.download_file(BUCKET, f"{user}/{file_name}/index.pkl", "/tmp/index.pkl")

#     bedrock_runtime = boto3.client(
#         service_name="bedrock-runtime",
#         region_name="us-east-1",
#     )

#     embeddings, llm = BedrockEmbeddings(
#         model_id="cohere.embed-english-v3",# "amazon.titan-embed-text-v1",
#         client=bedrock_runtime,
#         region_name="us-east-1",
#     ), Bedrock(
#         model_id="anthropic.claude-v2", client=bedrock_runtime, region_name="us-east-1"
#     )
#     faiss_index = FAISS.load_local("/tmp", embeddings)

#     message_history = DynamoDBChatMessageHistory(
#         table_name=MEMORY_TABLE, session_id=conversation_id
#     )

#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         chat_memory=message_history,
#         input_key="question",
#         output_key="answer",
#         return_messages=True,
#     )

#     qa = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=faiss_index.as_retriever(),
#         memory=memory,
#         return_source_documents=True,
#     )

#     res = qa({"question": human_input})

#     logger.info(res)

#     return {
#         "statusCode": 200,
#         "headers": {
#             "Content-Type": "application/json",
#             "Access-Control-Allow-Headers": "*",
#             "Access-Control-Allow-Origin": "*",
#             "Access-Control-Allow-Methods": "*",
#         },
#         "body": json.dumps(res["answer"]),
#     }
