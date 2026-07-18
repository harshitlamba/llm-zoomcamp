from ingest import load_faq_data, build_index
from rag_helper import RAGBase
from dotenv import load_dotenv
from openai import OpenAI
import os
import sys

def create_assistant(index, llm_client, model='gpt-5.4-mini'):
    return RAGBase(index=index, llm_client=llm_client, model=model)

# 1) search the indexed faq data for the questions most relevant to the user's question
# 2) build a prompt from that question plus the documents retrieved
# 3) send it to the LLM, which gives us the answer
if __name__=='__main__':
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    llm_client = OpenAI(api_key=api_key)
    documents = load_faq_data()
    index = build_index(documents=documents)

    assistant = create_assistant(index=index, llm_client=llm_client, model='gpt-4o-mini')
    # set default value for query
    query = 'How do I join the course?'
    # allow the query value to be overridden by a command-line argument
    if len(sys.argv) > 1:
        query = sys.argv[1]
    
    answer = assistant.rag(query)
    print(answer)
    