# storage/question_generator.py

# Python Libraries
import time
import random

# Vendor Libraries

# Local Libraries
from models.chroma import ChromaModel
from models.openai import OpenAIModel

from src.config import AI_ROLE, I_INFO, PROMPT_INSTR, EMPTY_RESP, I_QUES, RATE_LIMIT_TIME, I_WARNING
from src.utils import handle_rate_limit_error, show_timer


class QuestionGenerator(ChromaModel):
    def __init__(self, dataset: dict):
        self.batch_size = 0
        self.doc_handle = None
        #self.document_content_description = ''
        #self.llm = None

        super().__init__(dataset)
        self.prompt = self._prompt().strip()
        self.title = 'Hypothetical Questions'

        #self._set_attrs(dataset)

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
        hypothetical_questions = []
        hypothetical_questions_prompt = self._prompt().strip()

        # Track total items for progress logging
        total_chunks = len(semantic_chunks)
        print(f'{I_INFO} Processing {total_chunks} chunks using batch size {self.batch_size}.\n')

        for batch_start in range(0, total_chunks, self.batch_size):
            batch = semantic_chunks[batch_start: batch_start + self.batch_size]
            batched_hypothetical_questions = []

            for i, document in enumerate(batch, start=batch_start):
                rate_limit_hit = False
                
                # Dynamic per-request jittered sleep to keep the API gateway happy
                current_sleep_time = self.get_new_sleep_time()

                try:
                    formatted_response = hypothetical_questions_prompt.format(
                        AI_ROLE=AI_ROLE,
                        PROMPT_INSTR=PROMPT_INSTR,
                        docs=document.page_content
                    )

                    questions = OpenAIModel.filter_response(self.llm.invoke(formatted_response), i)

                except Exception as e:
                    questions = EMPTY_RESP
                    # Single call to cleanly parse the exception payload
                    current_sleep_time, rate_limit_hit = handle_rate_limit_error(
                        e, self.collection_name, current_sleep_time, i
                    )

                if rate_limit_hit:
                    print(f"{I_WARNING}️ Skipping chunk {i} due to rate limit threshold.\n")
                    continue

                if questions and questions != EMPTY_RESP:
                    questions_metadata = {
                        'original_content': document.page_content,
                        'source': document.metadata['source'],
                        'page': document.metadata['page'],
                        'doc_type': self.collection_name,
                    }

                    batched_hypothetical_questions.append(
                        self.doc_handle.create(questions, questions_metadata)
                    )

                # --- ⏳ PER-REQUEST THROTTLING ⏳ ---
                # We cool down immediately AFTER the execution inside the loop,
                # rather than dumping a massive burst and sleeping at the end of the batch.
                print(f"Chunk {i+1}/{total_chunks} completed. Throttling for {current_sleep_time:.2f}s...")
                time.sleep(current_sleep_time)

            hypothetical_questions.extend(batched_hypothetical_questions)

            # Optional: Keep a high-level batch update log
            processed_count = min(batch_start + self.batch_size, total_chunks)
            print(f"📊 Batch completed. Total progress: {processed_count}/{total_chunks} chunks written.")

        show_timer(start_time)
        return hypothetical_questions
