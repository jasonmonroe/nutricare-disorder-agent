# storage/question_generator.py

# Python Libraries
import time

# Vendor Libraries

# Local Libraries
from models.chroma import ChromaModel
from models.openai import OpenAIModel

from src.constants import (
    AGENT_EMPTY_RESP,
    AI_ROLE,
    I_DB,
    I_INFO,
    I_PEN,
    I_QUES,
    I_WARNING,
    PROMPT_INSTR
)
from src.utils import premium_model_tier, get_new_sleep_time, handle_rate_limit_error, show_timer
from storages.data_generator import DataGenerator


class QuestionGenerator(DataGenerator):
    def __init__(self, dataset: dict, chroma_db: ChromaModel):
        super().__init__(dataset, chroma_db)
        

    def get_hypothetical_questions(self, semantic_chunks) -> list:

        # Check tier status, if we're using the expensive models run this function instead and return
        if premium_model_tier():
            return self.get_hypothetical_questions_with_high_tier_models(semantic_chunks)


        # free tier version
        print(f'\n# --- {I_QUES} Getting {self.title} {I_QUES} --- #')
        print(f'{I_INFO} USING FREE TIER MODELS')

        start_time = time.time()
        hypothetical_questions = []

        total_chunks = len(semantic_chunks)
        print(f'{I_INFO}  Processing {total_chunks} chunks using prompt batch size {self.batch_size}.\n')

        # The loop now properly processes the slices as unified payloads
        for batch_start in range(0, total_chunks, self.batch_size):
            batch = semantic_chunks[batch_start: batch_start + self.batch_size]
            
            # --- 1. COMPACT THE BATCH INTO XML BLOCKS ---
            compacted_docs_list = []
            for idx, doc in enumerate(batch, start=batch_start):
                compacted_docs_list.append(
                    f'<chunk id="{idx}">\n{doc.page_content}\n</chunk>'
                )
            compacted_docs_str = "\n".join(compacted_docs_list)

            # --- 2. SINGLE API TRANSACTION PER BATCH ---
            rate_limit_hit = False
            current_sleep_time = get_new_sleep_time()

            try:
                # Format prompt passing ALL chunks in this batch at once
                formatted_response = self.prompt.format(
                    AI_ROLE=AI_ROLE,
                    PROMPT_INSTR=PROMPT_INSTR,
                    docs=compacted_docs_str  # Injected as a single unified variable
                )

                # Invoke the LLM once for the entire batch
                response = self.llm.invoke(formatted_response)
                
                # Expects a JSON back: {"chunk_id": ["q1", "q2", "q3"]}
                batch_questions_dict = OpenAIModel.filter_response(response, batch_start)

            except Exception as e:
                batch_questions_dict = {}
                current_sleep_time, rate_limit_hit = handle_rate_limit_error(
                    e, self.collection_name, int(current_sleep_time), batch_start
                )

            if rate_limit_hit:
                print(f"{I_WARNING}️ Skipping batch starting at chunk {batch_start} due to rate limits.\n")
                continue

            # --- 3. PARSE AND MAP THE GENERATED QUESTIONS ---
            if batch_questions_dict and isinstance(batch_questions_dict, dict):
                for idx, document in enumerate(batch, start=batch_start):
                    # Match the key back to the specific chunk index from the JSON payload
                    questions_for_chunk = batch_questions_dict.get(str(idx)) or batch_questions_dict.get(idx)
                    
                    if questions_for_chunk:
                        questions_metadata = {
                            'original_content': document.page_content,
                            'source': document.metadata['source'],
                            'page': document.metadata['page'],
                            'doc_type': self.collection_name,
                        }

                        hypothetical_questions.append(
                            self.doc_handle.create(questions_for_chunk, questions_metadata)
                        )

            # --- ⏳ PER-BATCH THROTTLING ⏳ ---
            # Throttling happens exactly ONCE per batch block instead of once per chunk file!
            processed_count = min(batch_start + self.batch_size, total_chunks)
            print(f"{I_PEN}  Batch progress: {processed_count}/{total_chunks} completed. Throttling for {current_sleep_time:.2f}s...")
            time.sleep(current_sleep_time)

        show_timer(start_time)
        return hypothetical_questions

    # --- Old version --- #
    """
    Use this method when the models are:
        LLAMA_MODEL=meta-llama/llama-guard-4-12b
        OPENAI_EMBEDDING_MODEL=text-embedding-3-small
        OPENAI_MODEL=gpt-4o-mini
    """
    def get_hypothetical_questions_with_high_tier_models(self, semantic_chunks) -> list:
        print(f'\n# --- {I_QUES} Getting {self.title} {I_QUES} --- #')
        print(f'{I_INFO} USING PREMIUM TIER MODELS')


        start_time = time.time()
        hypothetical_questions = []

        # Track total items for progress logging
        total_chunks = len(semantic_chunks)
        print(f'{I_INFO}  Processing {total_chunks} chunks using batch size {self.batch_size}.\n')

        for batch_start in range(0, total_chunks, self.batch_size):
            batch = semantic_chunks[batch_start: batch_start + self.batch_size]
            batched_hypothetical_questions = []

            for i, document in enumerate(batch, start=batch_start):
                rate_limit_hit = False
                
                # Dynamic per-request jittered sleep to keep the API gateway happy
                current_sleep_time = get_new_sleep_time()

                try:
                    formatted_response = self.prompt.format(
                        AI_ROLE=AI_ROLE,
                        PROMPT_INSTR=PROMPT_INSTR,
                        docs=document.page_content
                    )

                    questions = OpenAIModel.filter_response(self.llm.invoke(formatted_response), i)

                except Exception as e:
                    questions = AGENT_EMPTY_RESP
                    # Single call to cleanly parse the exception payload
                    current_sleep_time, rate_limit_hit = handle_rate_limit_error(
                        e, self.collection_name, int(current_sleep_time), i
                    )

                if rate_limit_hit:
                    print(f"{I_WARNING}️ Skipping chunk {i} due to rate limit threshold.\n")
                    continue

                if questions and questions != AGENT_EMPTY_RESP:
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
                # We cool down immediately AFTER the execution inside the loop, rather than dumping a massive burst and
                # sleeping at the end of the batch.
                print(f"{I_PEN}  Chunk {i+1}/{total_chunks} completed. Throttling for {current_sleep_time:.2f}s...")
                time.sleep(current_sleep_time)

            hypothetical_questions.extend(batched_hypothetical_questions)

            # Optional: Keep a high-level batch update log
            processed_count = min(batch_start + self.batch_size, total_chunks)
            print(f"{I_DB} Batch completed. Total progress: {processed_count}/{total_chunks} chunks written.")

        show_timer(start_time)

        return hypothetical_questions
