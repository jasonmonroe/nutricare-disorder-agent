# storage/table_question_generator.py
import random

from langchain.chains.query_constructor.base import AttributeInfo

from models.chroma import ChromaModel
from models.openai import OpenAIModel

from src.config import (
    AI_ROLE, 
    EMPTY_RESP,
    PROMPT_INSTR, 
)
from src.utils import handle_rate_limit_error, show_timer, start_timer


class TableQuestionGenerator(ChromaModel):
    print('TableQuestionGenerator')
    def __init__(self, dataset: dict):
        super().__init__(dataset)

        self.batch_size = None
        self.doc_handle = None
        self.document_content_description = None
        self.llm = None
        self.prompt = self._prompt()
        self.title = 'Hypothetical Table Questions'

    @staticmethod
    def _prompt():
        # Define a prompt for generating hypothetical questions for tables
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
        start_time = start_timer()
        rate_limit_hit = False
        table_hypothetical_questions = []
        sleep_time = random.randint(25, 45)

        hypothetical_questions_prompt = self._prompt().strip()

        # Generate hypothetical questions for each table in the documents
        for document in tables:  # Iterate over all processed documents
            for page_number in tables[document]:  # Iterate over pages in the document
                table_in_page = tables[document][page_number]  # Extract the table from the document

                try:
                    # Generate questions using the LLM based on the table content
                    page_content_text = page_texts.get(document, {}).get(page_number, "")

                    formatted_response = hypothetical_questions_prompt.format(
                        ROLE=AI_ROLE,
                        docs=page_content_text,
                        tables=table_in_page,
                        PROMPT_INSTR=PROMPT_INSTR,
                    )

                    response = self.llm.invoke(formatted_response)
                    questions = OpenAIModel.filter_response(response, page_number)

                except Exception as e:
                    handle_rate_limit_error(e, self.collection_name, sleep_time)
                    questions = EMPTY_RESP # Formerly "NA"

                    sleep_time, rate_limit_hit = handle_rate_limit_error(e, self.collection_name, sleep_time)

                if rate_limit_hit:
                    break

                if questions and questions != EMPTY_RESP:

                    # Metadata for each table
                    questions_metadata = {
                        'original_content': str(table_in_page),  # Store the content of the original table
                        'source': document,  # Store the source document name or identifier
                        'page': page_number,  # Store the page number where the table was found
                        'doc_type': self.collection_name,  # Indicate that the content type is a table
                    }

                    # Create a Document object for each set of generated questions
                    table_hypothetical_questions.append(
                        self.doc_handle.create(questions, questions_metadata)
                    )

        show_timer(start_time)

        print(table_hypothetical_questions)
        return table_hypothetical_questions
