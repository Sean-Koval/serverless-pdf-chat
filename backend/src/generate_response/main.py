import os 
import json
import boto3
import re
import time
from typing import Optional, Union, List, Dict, Any
from langchain.tools import BaseTool
from botocore.exceptions import ClientError
from langchain.chains import ConversationChain
from langchain.docstore.document import Document

from aws_lambda_powertools import Logger
from langchain.llms.bedrock import Bedrock
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter  # If text splitting is required
from langchain.memory import ConversationBufferWindowMemory


from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate  # For custom prompt templates
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader

MEMORY_TABLE = os.environ["MEMORY_TABLE"]
BUCKET = os.environ["BUCKET"]


s3 = boto3.client("s3")
logger = Logger()


    
class SymDocumentChatBot:

    def __init__(self, embeddings=None, llm=None, index=None, message_history=None, memory=None, user=None, file_name=None, conversation_id=None):
        self.embeddings = embeddings
        self.llm = llm
        self.index = index
        self.message_history = message_history
        self.memory = memory
        self.user = user
        self.file_name = file_name
        self.conversation_id = conversation_id
        self._general_chat = False

    def set_general_conversation(self):
        """Sets General conversational interface for users"""
        if self.file_name == "general_chat.pdf":
            self._general_chat = True
    
    def general_chat(self, input):
        """General conversational interface for users"""
        if self._general_chat:
            # self.memory = ConversationBufferMemory(
            #         memory_key="chat_history",
            #         chat_memory=self.message_history,
            #         output_key="answer",
            #         return_messages=True,
            #     )

            prompt_template = """   
                    As an AI developed with the Claude v2 model, your capabilities include understanding complex queries, summarizing information, and engaging in detailed conversations. You excel at providing concise answers, generating insights, and maintaining a conversational tone that's approachable and informative. When responding, consider the user's perspective, aim to add value with your responses, and ensure clarity and relevance in your summaries.
                    
                    Given this background, here's the current task: {input}

                    Remember:
                    - Aim for accuracy and helpfulness in your response.
                    - If the task involves a question, provide a clear and concise answer.
                    - If it involves summarizing, ensure your summary captures the key points effectively.
                    - Maintain a conversational tone that's friendly and engaging.

                    """
        
            prompt = PromptTemplate(
            template=prompt_template, input_variables=["input"]
            )
            llm_chain = LLMChain(llm=self.llm, 
                                prompt=prompt)

            # load the entire doc into a prompt and ask summary question
            # conversation_chain = ConversationChain(
            #     llm=self.llm,
            #     prompt=prompt,
            #     memory=self.memory
            # )
            #conversation_response = conversation_chain.predict(input)            
            conversation_response = llm_chain.predict(input=input)
            logger.info(f"conversation response: {conversation_response}")
            self.memory.chat_memory.add_user_message(input)
            self.memory.chat_memory.add_ai_message(conversation_response)

            return conversation_response
    
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
    # TEMP SOLUTION
    def summarize_stuff(self, input):
        """
        Summarizes doucument by loading the entire document into a single prompt to take advantage of broad summary questions.
        """

        try:
            document_path = f"/tmp/{self.file_name}"
            s3_start = time.time()  # Start timing S3 download
            print(f"{self.user}/{self.file_name}/{self.file_name}")
            s3.download_file(BUCKET, f"{self.user}/{self.file_name}/{self.file_name}", document_path)
            s3_end = time.time()  # End timing S3 download
            logger.info(f"S3 Download Time: {s3_end - s3_start} seconds")

            pages = self._load_and_split_pdf(document_path)
            large_doc_str = self.format_docs(pages)

            prompt_template = """Here is the task: {input}
                                
                                Here is the document:
                                {context}
                                """
            prompt = PromptTemplate(
            template=prompt_template, input_variables=["input", "context"]
            )

            # load the entire doc into a prompt and ask summary question
            llm_chain = LLMChain(
                llm=self.llm,
                prompt=prompt
            )
            summary_content = llm_chain.predict(input=input, context=large_doc_str)
            #self.memory.chat_memory.add_user_message(input)
            #self.memory.chat_memory.add_ai_message(summary_content)

            return summary_content

        except Exception as e:
            logger.error(f"There was an error with 'stuff' summarization: {e}")
    
    def format_docs(self, docs: List[Document]) -> str:
        """Format the docs."""
        return ", ".join([doc.page_content for doc in docs])
    
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
        
    def get_generated_summary(self, question):
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

        # Check if the summary file exists in the S3 bucket
        response = s3.list_objects_v2(Bucket=BUCKET, Prefix=summary_file_key)

        if 'Contents' in response:
            # File exists, download it
            try:
                self.s3.download_file(BUCKET, summary_file_key, temp_summary_path)
                with open(temp_summary_path, 'r', encoding='utf-8') as file:
                    summary_content = file.read()
            except ClientError as e:
                print(f"Error downloading file: {e}")
                return "Error handling summary file."
        else:
            # File does not exist, generate summary
            #summary_content = self.summarization()
            #print(summary_content)
            summary_content = self.summarize_stuff(question)

        
        # Update chat memory regardless of whether summary was downloaded or generated
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(summary_content)

        return summary_content


    def _load_and_split_pdf(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        return loader.load_and_split()

    def _perform_map_reduce_summarization(self, pages):
        # map_prompt_template = """
        #                 Write a summary of this chunk of text that includes the main points and any important details.
        #                 {text}
        #                 """
        # map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

        # # combine_prompt_template = """
        # #                 Write a concise summary of the following text delimited by triple backquotes.
        # #                 Return your response in bullet points which covers the key points of the text.
        # #                 ```{text}```
        # #                 BULLET POINT SUMMARY:
        # #                """
        
        # combine_prompt_template = """
        #                 Write a concise summary of the following text delimited by triple backquotes.
        #                 Return your response in paragraph format which covers the key points of the text.
        #                 ```{text}```
        #                 SUMMARY:
        #                 """

        # combine_prompt = PromptTemplate(
        #     template=combine_prompt_template, input_variables=["text"]
        # )

        question_prompt_template = """
                  Please provide a summary of the following text.
                  TEXT: {text}
                  SUMMARY:
                  """
        # REFINE PROMPT SECTION
        question_prompt = PromptTemplate(
            template=question_prompt_template, input_variables=["text"]
        )

        refine_prompt_template = """
                    Write a concise summary of the following text delimited by triple backquotes.
                    Return your response in bullet points which covers the key points of the text.
                    ```{text}```
                    BULLET POINT SUMMARY:
                    """

        refine_prompt = PromptTemplate(
            template=refine_prompt_template, input_variables=["text"]
        )

        # map_reduce_chain = load_summarize_chain(
        #     llm=self.llm,
        #     chain_type="map_reduce",
        #     map_prompt=map_prompt,
        #     combine_prompt=combine_prompt,
        #     return_intermediate_steps=True,
        # )

        map_reduce_chain = load_summarize_chain(
            llm=self.llm,
            chain_type="refine",
            question_prompt=question_prompt,
            refine_prompt=refine_prompt,
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
    print("memory")
    logger.info(memory)

    chat_bot_creation_start = time.time()

    # Create an instance of DocumentChatBot with necessary components
    chat_bot = SymDocumentChatBot(embeddings=embeddings, 
                                  llm=llm, 
                                  index=faiss_index, 
                                  memory=memory,
                                  user=user,
                                  file_name=file_name,
                                  conversation_id=conversation_id,
                                  message_history=message_history,
    )

    # initialize conversation chain
    chat_bot.set_general_conversation()

    chat_bot_creation_end = time.time()
    logger.info(f"ChatBot Creation Time: {chat_bot_creation_end - chat_bot_creation_start} seconds")

    action_determination_start = time.time()
    # Determine action
    action = chat_bot.determine_question_answering_vs_summarization(human_input)
    action_determination_end = time.time()
    logger.info(f"Action Determination Time: {action_determination_end - action_determination_start} seconds")
    
    # Act based on determined action
    if action == "question_answering" and not chat_bot._general_chat:
        response = chat_bot.question_answering(human_input)
        print("question answering response")

    elif action == "summarization" and not chat_bot._general_chat:
        # You would need to adjust how you get the text to summarize
        response = chat_bot.get_generated_summary(human_input)

        print(response)

    else:
        response = chat_bot.general_chat(human_input)
        print(response)
        #"Unable to determine action."

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
