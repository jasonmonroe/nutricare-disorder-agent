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
    I_PEN,
    I_QUES,
    PROMPT_INSTR
)
from src.utils import get_new_sleep_time, handle_rate_limit_error, show_timer, start_timer
from storages.data_generator import DataGenerator


class TableQuestionGenerator(DataGenerator):
    def __init__(self, dataset: dict, chroma_db: ChromaModel):
        super().__init__(dataset, chroma_db)

    def get_hypothetical_questions(self, page_texts, tables):
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
