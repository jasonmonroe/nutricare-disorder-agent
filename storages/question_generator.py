# storage/question_generator.py

# Python Libraries
import time

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
    I_WARNING
)
from src.model_config import config, ModelConfig
from src.utils import get_new_sleep_time, handle_rate_limit_error, show_timer
from storages.data_generator import DataGenerator


class QuestionGenerator(DataGenerator):
    """
    Creates hypothetical questions with answers that a user may ask. It is stored and then returned
    quickly for a fast response instead of constantly querying for every possible question.
    """

    def __init__(self, dataset: dict, chroma_db: ChromaModel):
        super().__init__(dataset, chroma_db)

    def get_hypothetical_questions(self, document_chunks) -> list:
        # Check tier status, if we're using the expensive models run this function instead and return
        if ModelConfig.is_premium():
            return self._with_premium_models(document_chunks)
        else:
            return self._with_free_models(document_chunks)

    def _with_free_models(self, document_chunks) -> list:
        # --- Free tier version --- #
        print(f'\n# --- {I_QUES} Getting {self.title} {I_QUES} --- #')
        print(f'{I_INFO}  USING FREE TIER MODELS')

        start_time = time.time()
        hypothetical_questions = []

        total_chunks = len(document_chunks)
        print(f'{I_INFO}  Processing {total_chunks} chunks using prompt batch size {self.batch_size}.\n')

        for batch_start in range(0, total_chunks, self.batch_size):
            batch = document_chunks[batch_start: batch_start + self.batch_size]

            # --- COMPACT THE BATCH INTO XML BLOCKS ---
            compacted_docs_list = []
            for idx, doc in enumerate(batch, start=batch_start):
                parent_id = (
                        doc.metadata.get('text_id') or
                        doc.metadata.get('table_id') or
                        getattr(doc, 'id', None) or
                        doc.metadata.get('doc_id', str(idx))
                )

                chunk_html = f'<chunk id="{parent_id}">\n{doc.page_content}\n</chunk>'
                compacted_docs_list.append(chunk_html)

            compacted_docs_str = "\n".join(compacted_docs_list)
            print(f'compacted_docs_str={compacted_docs_str}')

            # --- SINGLE API TRANSACTION PER BATCH ---
            rate_limit_hit = False
            current_sleep_time = get_new_sleep_time()

            try:
                formatted_response = self.prompt.format(
                    AI_ROLE=AI_ROLE,
                    PROMPT_INSTR=config.PROMPT_INSTR,
                    docs=compacted_docs_str
                )

                response = self.llm.invoke(formatted_response)
                batch_questions_dict = OpenAIModel.filter_response(response, batch_start)
                print(f'line 82: type(batch_question_dict)={type(batch_questions_dict)}, batch_questions_dict={batch_questions_dict}')

            except Exception as e:
                batch_questions_dict = {}
                current_sleep_time, rate_limit_hit = handle_rate_limit_error(
                    e,
                    self.doc_type,
                    int(current_sleep_time),
                    batch_start
                )

            if rate_limit_hit:
                print(f"{I_WARNING}️ Skipping batch starting at chunk {batch_start} due to rate limits.\n")
                continue

            # --- PARSE AND MAP THE GENERATED QUESTIONS ---
            print('Attempting to parse and map the generated questions....')
            if batch_questions_dict and isinstance(batch_questions_dict, dict):
                for idx, document in enumerate(batch, start=batch_start):
                    parent_id = (
                            document.metadata.get('text_id') or
                            document.metadata.get('table_id') or
                            getattr(document, 'id', None) or
                            document.metadata.get('doc_id', str(idx))
                    )

                    # 🎯 FIXED: Dynamic lookup matching the smart JSON keys returned by the LLM
                    questions_for_chunk = (
                            batch_questions_dict.get(str(parent_id)) or
                            batch_questions_dict.get(str(idx)) or
                            batch_questions_dict.get(idx)
                    )

                    if questions_for_chunk:
                        if isinstance(questions_for_chunk, list):
                            page_content_str = "\n".join(questions_for_chunk)
                        else:
                            page_content_str = str(questions_for_chunk)

                        questions_metadata = {
                            'chunk_id': idx,
                            'doc_type': self.doc_type,
                            'original_page_content': document.page_content,
                            'page': document.metadata['page'],
                            'parent_id': parent_id,
                            'source': document.metadata['source'],
                            'total_pages': document.metadata.get('total_pages', 0),
                        }

                        hypothetical_questions.append(
                            self.doc_handle.create(page_content_str, questions_metadata, True)
                        )
            else:
                print(f'line 133: batch_questions_dict={batch_questions_dict}')
                print(f'line 134: type(batch_questions_dict)={type(batch_questions_dict)}')
                #import sys
                #sys.exit(1)

            # --- ⏳ PER-BATCH THROTTLING ⏳ ---
            processed_count = min(batch_start + self.batch_size, total_chunks)
            print(f"\t{I_PEN}  Batch progress: {processed_count}/{total_chunks} completed. Throttling for {current_sleep_time:.2f}s...")
            time.sleep(current_sleep_time)

        show_timer(start_time)

        return hypothetical_questions

    def _with_premium_models(self, document_chunks) -> list:
        # --- Premium tier version --- #
        print(f'\n# --- {I_QUES} Getting {self.title} {I_QUES} --- #')
        print(f'{I_INFO} USING PREMIUM TIER MODELS')

        start_time = time.time()
        hypothetical_questions = []

        total_chunks = len(document_chunks)
        print(f'{I_INFO}  Processing {total_chunks} chunks using batch size {self.batch_size}.\n')

        for batch_start in range(0, total_chunks, self.batch_size):
            batch = document_chunks[batch_start: batch_start + self.batch_size]
            batched_hypothetical_questions = []

            for idx, document in enumerate(batch, start=batch_start):
                rate_limit_hit = False

                # 🎯 Matches the unified schema mapping strategy
                parent_id = (
                        document.metadata.get('text_id') or
                        document.metadata.get('table_id') or
                        getattr(document, 'id', None) or
                        document.metadata.get('doc_id', str(idx))
                )

                current_sleep_time = get_new_sleep_time()

                try:
                    formatted_response = self.prompt.format(
                        AI_ROLE=AI_ROLE,
                        PROMPT_INSTR=config.PROMPT_INSTR,
                        docs=document.page_content
                    )

                    questions = OpenAIModel.filter_response(self.llm.invoke(formatted_response), idx)

                except Exception as e:
                    questions = AGENT_EMPTY_RESP
                    current_sleep_time, rate_limit_hit = handle_rate_limit_error(
                        e,
                        self.doc_type,
                        int(current_sleep_time),
                        idx
                    )

                if rate_limit_hit:
                    print(f"{I_WARNING}️ Skipping chunk {idx} due to rate limit threshold.\n")
                    continue

                if questions and questions != AGENT_EMPTY_RESP:
                    # If premium models return structured dict, dynamically grab by key or fall back to native response
                    if isinstance(questions, dict):
                        questions_for_chunk = (
                                questions.get(str(parent_id)) or
                                questions.get(str(idx)) or
                                questions.get(idx) or
                                questions
                        )
                    else:
                        questions_for_chunk = questions

                    if isinstance(questions_for_chunk, list):
                        page_content_str = "\n".join(questions_for_chunk)
                    else:
                        page_content_str = str(questions_for_chunk)


                    questions_metadata = {
                        'chunk_id': idx,
                        'doc_type': self.doc_type,
                        'original_page_content': document.page_content,
                        'page': document.metadata['page'],
                        'parent_id': parent_id,
                        'source': document.metadata['source'],
                        'total_pages': document.metadata.get('total_pages', 0),
                    }

                    batched_hypothetical_questions.append(
                        self.doc_handle.create(page_content_str, questions_metadata, True)
                    )

                # --- ⏳ PER-REQUEST THROTTLING ⏳ ---
                print(f"\t{I_PEN}  Chunk {(idx + 1)}/{total_chunks} completed. Throttling for {current_sleep_time:.2f}s...")
                time.sleep(current_sleep_time)

            hypothetical_questions.extend(batched_hypothetical_questions)

            processed_count = min(batch_start + self.batch_size, total_chunks)
            print(f"{I_DB} Batch completed. Total progress: {processed_count}/{total_chunks} chunks written.")

        show_timer(start_time)

        return hypothetical_questions