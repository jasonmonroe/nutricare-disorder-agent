# storage/table_question_generator.py

# Python Libraries
import time

# Vendor Libraries

# Local Libraries
from models.chroma import ChromaModel
from models.openai import OpenAIModel

from src.constants import (
    AI_ROLE,
    AGENT_EMPTY_RESP,
    I_INFO,
    I_PEN,
    I_QUES,
    PROMPT_INSTR
)
from src.utils import get_new_sleep_time, handle_rate_limit_error, premium_model_tier, show_timer, start_timer
from storages.data_generator import DataGenerator


class TableQuestionGenerator(DataGenerator):
    def __init__(self, dataset: dict, chroma_db: ChromaModel):
        super().__init__(dataset, chroma_db)


    def get_hypothetical_questions(self, page_texts, tables):

        # Check tier status, if we're using the expensive models run this function instead and return
        if premium_model_tier():
            return self.get_hypothetical_questions_with_high_tier_models(page_texts, tables)
        
        
        # free tier version
        print(f'\n# --- {I_QUES} Getting {self.title} {I_QUES} --- #')
        print(f'{I_INFO} USING FREE TIER MODELS')

        start_time = start_timer()
        table_hypothetical_questions = []
        processed_count = 0

        # Loop through each document
        for doc_index, document in enumerate(tables, start=1):
            rate_limit_hit = False
            
            # --- TIER 2: Gather and compact page-level tables ---
            for page_number in tables[document]: 
                table_in_page = tables[document][page_number]
                page_content_text = page_texts.get(document, {}).get(page_number, "")

                # Skip empty data structures
                if not table_in_page:
                    continue

                # Compute independent jitter per page run
                current_sleep_time = get_new_sleep_time()

                # Structure the table explicitly so the LLM can parse its context
                compacted_table_str = (
                    f'<table_context page="{page_number}">\n'
                    f'{str(table_in_page)}\n'
                    f'</table_context>'
                )

                try:
                    # Pass the aggregated data structure in a single request
                    formatted_response = self.prompt.format(
                        AI_ROLE=AI_ROLE,
                        docs=page_content_text,
                        tables=compacted_table_str,
                        PROMPT_INSTR=PROMPT_INSTR,
                    )

                    response = self.llm.invoke(formatted_response)
                    
                    # Expecting a clean list or a structured JSON dictionary back
                    questions = OpenAIModel.filter_response(response, page_number)

                except Exception as e:
                    questions = AGENT_EMPTY_RESP
                    current_sleep_time, rate_limit_hit = handle_rate_limit_error(
                        e, self.collection_name, current_sleep_time, page_number
                    )

                if rate_limit_hit:
                    print(f"⚠️ Terminating run for document '{document}' due to rate limits.")
                    break

                # --- TIER 3: Write to your single open database trait ---
                if questions and questions != AGENT_EMPTY_RESP:
                    questions_metadata = {
                        'original_content': str(table_in_page), 
                        'source': document,  
                        'page': page_number,  
                        'doc_type': self.collection_name,  
                    }

                    table_hypothetical_questions.append(
                        self.doc_handle.create(questions, questions_metadata)
                    )

                processed_count += 1
                print(f"{I_PEN} Page {page_number} Tables (Doc {doc_index}) complete. Throttling {current_sleep_time:.2f}s...")
                
                # Throttling happens exactly ONCE per page instead of once per nested table row artifact
                time.sleep(current_sleep_time)

            if rate_limit_hit:
                break

        show_timer(start_time)
        return table_hypothetical_questions



    """
    Use this method when the models are:
        LLAMA_MODEL=meta-llama/llama-guard-4-12b
        OPENAI_EMBEDDING_MODEL=text-embedding-3-small
        OPENAI_MODEL=gpt-4o-mini
    """
    def get_hypothetical_questions_with_high_tier_models(self, page_texts, tables):
        print(f'\n# --- {I_QUES} Getting {self.title} {I_QUES} --- #')
        
        start_time = start_timer()
        table_hypothetical_questions = []

        # Counter for descriptive logging
        processed_count = 0

        # Generate hypothetical questions for each table in the documents
        for doc_index, document in enumerate(tables, start=1):
            rate_limit_hit = False
            for page_number in tables[document]: 
                table_in_page = tables[document][page_number]

                # Compute independent jitter per nested loop run
                current_sleep_time = get_new_sleep_time()

                try:
                    page_content_text = page_texts.get(document, {}).get(page_number, "")

                    formatted_response = self.prompt.format(
                        AI_ROLE=AI_ROLE,
                        docs=page_content_text,
                        tables=table_in_page,
                        PROMPT_INSTR=PROMPT_INSTR,
                    )

                    response = self.llm.invoke(formatted_response)
                    questions = OpenAIModel.filter_response(response, page_number)

                except Exception as e:
                    questions = AGENT_EMPTY_RESP
                    # Single execution point prevents log thrashing and double mutations
                    current_sleep_time, rate_limit_hit = handle_rate_limit_error(
                        e, self.collection_name, current_sleep_time, page_number
                    )

                if rate_limit_hit:
                    print(f"⚠️ Terminating run for document '{document}' due to rate limits.")
                    break

                if questions and questions != AGENT_EMPTY_RESP:
                    questions_metadata = {
                        'original_content': str(table_in_page), 
                        'source': document,  
                        'page': page_number,  
                        'doc_type': self.collection_name,  
                    }

                    table_hypothetical_questions.append(
                        self.doc_handle.create(questions, questions_metadata)
                    )

                processed_count += 1
                print(f"{I_PEN} Table {processed_count} (Doc: {doc_index}, Page: {page_number}) parsed. Throttling {current_sleep_time:.2f}s...")
                
                # --- ⏳ PER-TABLE THROTTLING ⏳ ---
                # Forces execution tracking to sleep gracefully right after invoking the gateway
                time.sleep(current_sleep_time)

            if rate_limit_hit:
                break

        show_timer(start_time)
        
        return table_hypothetical_questions
