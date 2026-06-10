# storage/table_question_generator.py

# Python Libraries
import time
import random

# Vendor Libraries
from langchain_classic.chains.query_constructor.schema import AttributeInfo

# Local Libraries
from models.chroma import ChromaModel
from models.openai import OpenAIModel

from src.config import (
    AI_ROLE, 
    EMPTY_RESP,
    I_INFO,
    I_QUES,
    PROMPT_INSTR,
    RATE_LIMIT_TIME, 
)
from src.utils import handle_rate_limit_error, show_timer, start_timer


class TableQuestionGenerator(ChromaModel):
    def __init__(self, dataset: dict):
        self.batch_size = None
        self.doc_handle = None
        self.document_content_description = None
        self.llm = None
        
        super().__init__(dataset)

        self.prompt = self._prompt().strip()
        self.title = 'Hypothetical Table Questions'

        #self._set_attrs(dataset)

    @staticmethod
    def _prompt():
        return """
        [SYSTEM INSTRUCTION]
        You are an AI {AI_ROLE} specialized in generating precise, clinically relevant questions for information retrieval.
        Your task is to analyze the provided CONTEXT and TABLE DATA, and generate a list of three hypothetical questions that are directly answerable by the data.

        [RULES]
        1. Focus questions strictly on the **numeric data, specific values, formulas, or definitive lists** found in the table.
        2. Use the **adjacent text (if provided)** to establish the clinical context for the questions.
        3. If no adjacent text is provided, generate questions based on the table data alone.
        4. **DO NOT** include any preamble, conversational text, or explanation in your response.
        5. **OUTPUT ONLY THE JSON OBJECT.**

        [FULL CONTEXT]
        ---TEXT ADJACENT TO TABLE---
        {docs}

        --- TABLE DATA (Structured for Analysis) ---
        {tables}

        [OUTPUT FORMAT]
        Generate a JSON object containing a list of questions under the key "questions".
        The list must contain **a minimum of 1 and a maximum of 3** questions.

        {PROMPT_INSTR}
        """

    def get_hypothetical_questions(self, page_texts, tables):
        print(f'\n# --- {I_QUES} Getting Hypothetical Table Questions {I_QUES} --- #')
        
        start_time = start_timer()
        table_hypothetical_questions = []
        hypothetical_questions_prompt = self._prompt().strip()

        # Counter for descriptive logging
        processed_count = 0

        # Generate hypothetical questions for each table in the documents
        for doc_index, document in enumerate(tables, start=1): 
            for page_number in tables[document]: 
                table_in_page = tables[document][page_number]
                rate_limit_hit = False
                
                # Compute independent jitter per nested loop run
                current_sleep_time = current_sleep_time = self.get_new_sleep_time()

                try:
                    page_content_text = page_texts.get(document, {}).get(page_number, "")

                    formatted_response = hypothetical_questions_prompt.format(
                        AI_ROLE=AI_ROLE,
                        docs=page_content_text,
                        tables=table_in_page,
                        PROMPT_INSTR=PROMPT_INSTR,
                    )

                    response = self.llm.invoke(formatted_response)
                    questions = OpenAIModel.filter_response(response, page_number)

                except Exception as e:
                    questions = EMPTY_RESP
                    # Single execution point prevents log thrashing and double mutations
                    current_sleep_time, rate_limit_hit = handle_rate_limit_error(
                        e, self.collection_name, current_sleep_time, page_number
                    )

                if rate_limit_hit:
                    print(f"⚠️ Terminating run for document '{document}' due to rate limits.")
                    break

                if questions and questions != EMPTY_RESP:
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
                print(f"📝 Table {processed_count} (Doc: {doc_index}, Page: {page_number}) parsed. Throttling {current_sleep_time:.2f}s...")
                
                # --- ⏳ PER-TABLE THROTTLING ⏳ ---
                # Forces execution tracking to sleep gracefully right after invoking the gateway
                time.sleep(current_sleep_time)

            if rate_limit_hit:
                break

        show_timer(start_time)
        return table_hypothetical_questions
