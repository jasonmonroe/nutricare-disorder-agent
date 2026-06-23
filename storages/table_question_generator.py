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
    """
    Generates synthetic questions for data tables and embeds them directly inside
    the document content with the source table. Eliminates relational cross-collection lookups.
    """
    def __init__(self, dataset: dict, chroma_db: ChromaModel):
        super().__init__(dataset, chroma_db)

    def get_hypothetical_questions(self, page_texts, tables) -> list:
        if ModelConfig.is_premium():
            return self._with_premium_models(page_texts, tables)
        else:
            return self._with_free_models(page_texts, tables)

    def _with_free_models(self, page_texts: dict, tables: dict) -> list:
        print(f'\n# --- {I_QUES} Getting {self.title} {I_QUES} --- #')
        print(f'{I_INFO}  USING FREE TIER MODELS (FLATTENED SCHEMA)')

        start_time = start_timer()
        table_hypothetical_questions = []
        processed_count = 0

        for doc_index, document in enumerate(tables, start=1):
            rate_limit_hit = False

            for page_number in tables[document]:
                table_data_package = tables[document][page_number]

                if not table_data_package or not table_data_package.get("rows"):
                    continue

                table_in_page = table_data_package["rows"]
                page_content_text = page_texts.get(document, {}).get(page_number, {}).get("content", "")

                # Structure table in simple format for LLM parsing
                compacted_table_str = (
                    f'<table_context page="{page_number}">\n'
                    f'{str(table_in_page)}\n'
                    f'</table_context>'
                )

                current_sleep_time = get_new_sleep_time()

                try:
                    formatted_response = self.prompt.format(
                        AI_ROLE=AI_ROLE,
                        docs=page_content_text,
                        tables=compacted_table_str,
                        PROMPT_INSTR=config.PROMPT_INSTR,
                    )

                    response = self.llm.invoke(formatted_response)
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

                # --- FLATTEN THE RECORD: Combine Questions + Raw Table Content ---
                if questions and questions != AGENT_EMPTY_RESP:
                    if isinstance(questions, list):
                        questions_str = "\n".join([f"- {q}" for q in questions])
                    elif isinstance(questions, dict):
                        # Gracefully handle if filter_response passes back a raw dict collection
                        questions_list = questions.get("questions", [])
                        questions_str = "\n".join([f"- {q}" for q in questions_list]) if questions_list else str(questions)
                    else:
                        questions_str = str(questions)

                    flattened_table_content = f"""### Target Search Terms & Synthetic Questions:
                    {questions_str}
                    
                    ### Source Table Data:
                    {str(table_in_page)}""".strip()

                    questions_metadata = {
                        'doc_type': self.doc_type,
                        'page': page_number,
                        'source': document,
                    }

                    table_hypothetical_questions.append(
                        self.doc_handle.create(flattened_table_content, questions_metadata)
                    )

                processed_count += 1
                print(f"\t{I_PEN} Table: {processed_count}, Page {page_number} Tables (Doc {doc_index}) complete. Throttling {current_sleep_time:.2f}s...")
                time.sleep(current_sleep_time)

            if rate_limit_hit:
                break

        show_timer(start_time)
        return table_hypothetical_questions

    def _with_premium_models(self, page_texts: dict, tables: dict) -> list:
        print(f'\n# --- {I_QUES} Getting {self.title} {I_QUES} --- #')
        print(f'{I_INFO} USING PREMIUM TIER MODELS (FLATTENED SCHEMA)')

        start_time = start_timer()
        table_hypothetical_questions = []
        processed_count = 0

        for doc_index, document in enumerate(tables, start=1):
            rate_limit_hit = False
            for page_number in tables[document]:
                table_data_package = tables[document][page_number]

                if not table_data_package or not table_data_package.get("rows"):
                    continue

                table_in_page = table_data_package["rows"]
                page_content_text = page_texts.get(document, {}).get(page_number, {}).get("content", "")

                current_sleep_time = get_new_sleep_time()

                try:
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
                    current_sleep_time, rate_limit_hit = handle_rate_limit_error(
                        e,
                        self.collection_name,
                        current_sleep_time,
                        page_number
                    )

                if rate_limit_hit:
                    print(f"⚠️ Terminating run for document '{document}' due to rate limits.")
                    break

                # --- FLATTEN THE RECORD ---
                if questions and questions != AGENT_EMPTY_RESP:
                    if isinstance(questions, list):
                        questions_str = "\n".join([f"- {q}" for q in questions])
                    elif isinstance(questions, dict):
                        questions_list = questions.get("questions", [])
                        questions_str = "\n".join([f"- {q}" for q in questions_list]) if questions_list else str(questions)
                    else:
                        questions_str = str(questions)

                    flattened_table_content = f"""### Target Search Terms & Synthetic Questions:
                    {questions_str}
                    
                    ### Source Table Data:
                    {str(table_in_page)}"""

                    questions_metadata = {
                        'doc_type': self.doc_type,
                        'page': page_number,
                        'source': document,
                    }

                    table_hypothetical_questions.append(
                        self.doc_handle.create(flattened_table_content, questions_metadata)
                    )

                processed_count += 1
                print(f"\t{I_PEN} Processed: {processed_count} (Doc: {doc_index}, Page: {page_number}) parsed. Throttling {current_sleep_time:.2f}s...")
                time.sleep(current_sleep_time)

            if rate_limit_hit:
                break

        show_timer(start_time)
        return table_hypothetical_questions