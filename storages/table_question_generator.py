# storage/table_question_generator.py

# Python Libraries
import os
import time

# Local Libraries
from models.chroma import ChromaModel
from models.openai import OpenAIModel

from src.constants import (
    AGENT_EMPTY_RESP,
    AI_ROLE,
    I_INFO,
    I_PEN,
    I_QUES, DOCUMENT_FILEPATH,
)
from src.model_config import config, ModelConfig
from src.utils import get_new_sleep_time, handle_rate_limit_error, show_timer, start_timer
from storages.data_generator import DataGenerator


class TableQuestionGenerator(DataGenerator):
    def __init__(self, dataset: dict, chroma_db: ChromaModel):
        super().__init__(dataset, chroma_db)

    def get_hypothetical_questions(self, page_texts, tables) -> list:
        # Check tier status, if we're using the expensive models run this function instead and return
        if ModelConfig.is_premium():
            return self._with_premium_models(page_texts, tables)
        else:
            return self._with_free_models(page_texts, tables)

    def _with_free_models(self, page_texts: dict, tables: dict) -> list:
        # Free tier version
        print(f'\n# --- {I_QUES} Getting {self.title} {I_QUES} --- #')
        print(f'{I_INFO}  USING FREE TIER MODELS')

        start_time = start_timer()
        table_hypothetical_questions = []
        processed_count = 0

        # Loop through each document
        for doc_index, document in enumerate(tables, start=1):
            rate_limit_hit = False

            # --- Gather and compact page-level tables ---
            for page_number in tables[document]:
                table_data_package = tables[document][page_number]

                # Skip empty data structures
                if not table_data_package or not table_data_package.get("rows"):
                    continue

                table_in_page = table_data_package["rows"]
                page_content_text = page_texts.get(document, {}).get(page_number, {}).get("content", "")

                # 🎯 Extract straight from the source payload keys!
                parent_id = table_data_package.get("table_id")
                #print(f'DEBUG: line 97: parent_id={parent_id}')

                # Structure the table explicitly so the LLM can parse its context
                compacted_table_str = (
                    f'<table_context id="{parent_id}" page="{page_number}">\n'
                    f'{str(table_in_page)}\n'
                    f'</table_context>'
                )

                current_sleep_time = get_new_sleep_time()

                try:
                    # Pass the aggregated data structure in a single request
                    formatted_response = self.prompt.format(
                        AI_ROLE=AI_ROLE,
                        docs=page_content_text,
                        tables=compacted_table_str,
                        PROMPT_INSTR=config.PROMPT_INSTR,
                    )

                    response = self.llm.invoke(formatted_response)

                    # Expecting a clean list or a structured JSON dictionary back
                    questions = OpenAIModel.filter_response(response, page_number)

                except Exception as e:
                    questions = AGENT_EMPTY_RESP
                    current_sleep_time, rate_limit_hit = handle_rate_limit_error(
                        e,
                        self.collection_name,
                        current_sleep_time,
                        page_number
                    )

                if rate_limit_hit:
                    print(f"⚠️ Terminating run for document '{document}' due to rate limits.")
                    break

                # --- Write to your single open database trait ---
                if questions and questions != AGENT_EMPTY_RESP:
                    questions_metadata = {
                        'doc_type': self.doc_type,
                        'original_page_content': str(table_in_page),
                        'page': page_number,
                        'parent_id': parent_id,
                        'source': document,
                        #'total_pages': <source>.get('total_pages', 0),
                    }

                    table_hypothetical_questions.append(
                        self.doc_handle.create(questions, questions_metadata)
                    )

                processed_count += 1
                print(f"\t{I_PEN} Table: {processed_count}, Page {page_number} Tables (Doc {doc_index}) complete. Throttling {current_sleep_time:.2f}s...")

                # Throttling happens exactly ONCE per page instead of once per nested table row artifact
                time.sleep(current_sleep_time)

            if rate_limit_hit:
                break

        show_timer(start_time)

        return table_hypothetical_questions

    def _with_premium_models(self, page_texts: dict, tables: dict) -> list:
        print(f'\n# --- {I_QUES} Getting {self.title} {I_QUES} --- #')

        start_time = start_timer()
        table_hypothetical_questions = []
        processed_count = 0

        # Generate hypothetical questions for each table in the documents
        for doc_index, document in enumerate(tables, start=1):
            rate_limit_hit = False
            for page_number in tables[document]:
                table_data_package = tables[document][page_number]

                if not table_data_package or not table_data_package.get("rows"):
                    continue

                table_in_page = table_data_package["rows"]
                page_content_text = page_texts.get(document, {}).get(page_number, {}).get("content", "")

                # 🎯 Extract straight from the source payload keys!
                parent_id = table_data_package.get("table_id")
                current_sleep_time = get_new_sleep_time()

                try:
                    #page_content_text = page_texts.get(document, {}).get(page_number, "")

                    formatted_response = self.prompt.format(
                        AI_ROLE=AI_ROLE,
                        docs=page_content_text,
                        tables=table_in_page,
                        PROMPT_INSTR=config.PROMPT_INSTR,
                    )

                    response = self.llm.invoke(formatted_response)
                    questions = OpenAIModel.filter_response(response, page_number)

                except Exception as e:
                    questions = AGENT_EMPTY_RESP

                    # Single execution point prevents log thrashing and double mutations
                    current_sleep_time, rate_limit_hit = handle_rate_limit_error(
                        e,
                        self.collection_name,
                        current_sleep_time,
                        page_number
                    )

                if rate_limit_hit:
                    print(f"⚠️ Terminating run for document '{document}' due to rate limits.")
                    break

                if questions and questions != AGENT_EMPTY_RESP:
                    questions_metadata = {
                        'parent_id': parent_id,
                        'doc_type': self.doc_type,
                        'original_page_content': str(table_in_page),
                        'page': page_number,
                        'source': document,
                    }

                    table_hypothetical_questions.append(
                        self.doc_handle.create(questions, questions_metadata)
                    )

                processed_count += 1
                print(f"\t{I_PEN} Processed: {processed_count} (Doc: {doc_index}, Page: {page_number}) parsed. Throttling {current_sleep_time:.2f}s...")

                # --- ⏳ PER-TABLE THROTTLING ⏳ ---
                time.sleep(current_sleep_time)

            if rate_limit_hit:
                break

        show_timer(start_time)

        return table_hypothetical_questions