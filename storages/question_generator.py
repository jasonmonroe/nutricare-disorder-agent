# storage/question_generator.py

import random
import time
from langchain.chains.query_constructor.base import AttributeInfo
from models.chroma import ChromaModel
from models.openai import OpenAIModel
from src.config import DOCUMENT_CHUNK_TEXT_BATCH_SIZE, DOCUMENT_DIR, AI_ROLE, PROMPT_INSTR, EMPTY_RESP
from src.doc_handler import DocHandler
from src.utils import handle_rate_limit_error, show_timer


def _prompt() -> str:
    return """
        You are an AI {AI_ROLE} specialized in generating precise, clinically relevant questions for information retrieval.
        Your task is to analyze the provided TEXT CHUNK and generate a list of exactly three hypothetical questions for which the chunk contains the complete answer.
        
        **QUESTION STYLE REQUIREMENTS:**
        1. Questions must be factual and directly address **diagnostic criteria, treatment dosages, clinical findings, or defining concepts** mentioned in the TEXT CHUNK.
        2. Phrasing must be natural and sound like a question a {ROLE} would actually ask.
        
        TEXT CHUNK:
        {docs}
        
        {PROMPT_INSTR}
        """

class QuestionGenerator(ChromaModel):
    def __init__(self, dataset: dict):
        super().__init__(dataset)

        self.doc_handle = dataset['doc_handle']
        #self.collection_name = dataset['collection_name']
        #self.metadata = []
        self.metadata_info = self._get_metadata_info()
        self.title = ''
        self.batch_size = DOCUMENT_CHUNK_TEXT_BATCH_SIZE


        #self.document_content_description = "Text Semantic Chunks for " + DOCUMENT_DIR + " published by the Global Nutritional Health Organization"
        self.document_content_description = dataset['document_content_description']
        self.prompt = _prompt()
        self.llm = dataset['llm']



    def get_hypothetical_questions(self, semantic_chunks):
        start_time = time.time()
        rate_limit_hit = False
        sleep_time = random.randint(25, 45)
        #semantic_chunks = dataset['semantic_chunks']
        batch_size = DOCUMENT_CHUNK_TEXT_BATCH_SIZE
        hypothetical_questions_prompt = _prompt()

        openai_model = OpenAIModel()

        hypothetical_questions = []

        for batch_start in range(0, len(semantic_chunks), batch_size):
            batch = semantic_chunks[batch_start: batch_start + batch_size]

            # List to store documents with hypothetical questions
            batched_hypothetical_questions = []

            for i, document in enumerate(batch, start=batch_start):
                try:
                    # Invoke the LLM to generate questions based on the chunk content
                    formatted_response = hypothetical_questions_prompt.format(
                        AI_ROLE=AI_ROLE,
                        PROMPT_INSTR=PROMPT_INSTR,
                        docs=document.page_content
                    )

                    questions = openai_model.filter_response(self.llm.invoke(formatted_response), i)

                except Exception as e:
                    handle_rate_limit_error(e, self.collection_name, sleep_time)
                    questions = EMPTY_RESP # Formerly "NA"

                    sleep_time, rate_limit_hit = handle_rate_limit_error(e, self.collection_name, sleep_time)

                if rate_limit_hit:
                    break

                # Only append metadata for successful responses
                if questions and questions != EMPTY_RESP:

                    # Create metadata for the generated question
                    questions_metadata = {
                        'original_content': document.page_content, # Store the original chunk content
                        'source': document.metadata['source'],     # Source document of the chunk
                        'page': document.metadata['page'],         # Page number where the chunk appears
                        'doc_type': self.collection_name,               # Indicate the content type
                    }

                    # Create and store the document containing generated questions
                    batched_hypothetical_questions.append(
                        self.doc_handle._create(
                            questions,
                            questions_metadata
                        )
                    )

            # Store each chunk into the master list of documents with hypothetical questions
            hypothetical_questions.extend(batched_hypothetical_questions)

            # ** Wait for 1 minute before processing the next batch **
            print(f"Processed {batch_start + batch_size} / {len(semantic_chunks)} chunks. Waiting {sleep_time} seconds...")
            time.sleep(sleep_time)

        show_timer(start_time)

        print(hypothetical_questions)

        return hypothetical_questions


    def _get_metadata_info(self):
        return [
            AttributeInfo(
                name="page",  # Fill in the metadata field name (e.g., "Category")
                description="page number of document",  # Describe what this field represents
                type="integer"  # Fill in the data type (e.g., "string", "integer")
            ),
            AttributeInfo(
                name="source",
                description="file path of document",
                type="string"
            ),
            AttributeInfo(
                name="page_content",
                description="raw text of (sectional) document",
                type="string"
            )
        ]



    """
    collection name: hypothetical_questions, table_hypothetical_questions, nutritional
    metadata_info
    metadata
    document_content_description
    batch_size: DOCUMENT_CHUNK_BATCH_SIZE, DOCUMENT_CHUNK_TEXT_BATCH_SIZE
    title: Hypothetical Questions,
    prompt (hypothetical_questions_prompt)

    """

