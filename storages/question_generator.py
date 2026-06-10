# storage/question_generator.py

# +-------------------+
# |     QUESTIONS     |
# +-------------------+

# Python Libraries
import time

# Vendor Libraries

# Local Libraries
from models.chroma import ChromaModel
from models.openai import OpenAIModel

from src.config import AI_ROLE, I_INFO, PROMPT_INSTR, EMPTY_RESP, I_QUES, RATE_LIMIT_TIME
from src.utils import handle_rate_limit_error, show_timer


class QuestionGenerator(ChromaModel):
    def __init__(self, dataset: dict):
        super().__init__(dataset)

        self.batch_size = None
        self.doc_handle = None
        self.document_content_description = None
        self.llm = None
        self.prompt = self._prompt().strip()
        self.title = 'Hypothetical Questions'

        self._set_attrs(dataset)

    @staticmethod
    def _prompt() -> str:
        return """
            You are an AI {AI_ROLE} specialized in generating precise, clinically relevant questions for information retrieval.
            Your task is to analyze the provided TEXT CHUNK and generate a list of exactly three hypothetical questions for which the chunk contains the complete answer.
            
            **QUESTION STYLE REQUIREMENTS:**
            1. Questions must be factual and directly address **diagnostic criteria, treatment dosages, clinical findings, or defining concepts** mentioned in the TEXT CHUNK.
            2. Phrasing must be natural and sound like a question a {AI_ROLE} would actually ask.
            
            TEXT CHUNK:
            {docs}
            
            {PROMPT_INSTR}
            """

    def get_hypothetical_questions(self, semantic_chunks) -> list:
        print(f'\n# --- {I_QUES} Getting Hypothetical Questions {I_QUES} --- #')

        start_time = time.time()
        rate_limit_hit = False
        sleep_time = RATE_LIMIT_TIME
        hypothetical_questions = []
        hypothetical_questions_prompt = self._prompt().strip()

        print(f'{I_INFO} Rate limit sleep timer: {sleep_time}\n')
        for batch_start in range(0, len(semantic_chunks), self.batch_size):
            batch = semantic_chunks[batch_start: batch_start + self.batch_size]

            # List to store documents with hypothetical questions.
            batched_hypothetical_questions = []

            for i, document in enumerate(batch, start=batch_start):
                try:
                    # Invoke the LLM to generate questions based on the chunk content
                    formatted_response = hypothetical_questions_prompt.format(
                        AI_ROLE=AI_ROLE,
                        PROMPT_INSTR=PROMPT_INSTR,
                        docs=document.page_content
                    )

                    questions = OpenAIModel.filter_response(self.llm.invoke(formatted_response), i)

                except Exception as e:
                    handle_rate_limit_error(e, self.collection_name, sleep_time)
                    questions = EMPTY_RESP # Formerly "NA"
                    sleep_time, rate_limit_hit = handle_rate_limit_error(e, self.collection_name, sleep_time, i)

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
                        self.doc_handle.create(
                            questions,
                            questions_metadata
                        )
                    )

            # Store each chunk into the master list of documents with hypothetical questions
            hypothetical_questions.extend(batched_hypothetical_questions)

            # ** Wait for 1 minute before processing the next batch **
            chunk_cnt = (batch_start + self.batch_size) / len(semantic_chunks)
            print(f"\nProcessed {chunk_cnt} chunks.  Waiting {sleep_time} seconds...")
            time.sleep(sleep_time)

        show_timer(start_time)

        print(hypothetical_questions)
        return hypothetical_questions
