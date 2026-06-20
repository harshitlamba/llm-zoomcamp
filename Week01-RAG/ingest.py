import requests
import time
from minsearch import Index
from sqlitesearch import TextSearchIndex

# fetch and return the documents
def load_faq_data():
    # hard-coded as we want to focus on building a rag for faq documents from this specific source; the index fields depend on this
    url_prefix='https://datatalks.club/faq'
    docs_url='/json/courses.json'

    response = requests.get(url=f'{url_prefix}{docs_url}')
    courses_raw = response.json()

    documents = []
    for course in courses_raw:
        course_url = f'{url_prefix}{course['path']}'
        course_response = requests.get(url=course_url)
        course_response.raise_for_status()
        course_faqs = course_response.json()

        documents.extend(course_faqs)

    return documents

# create and return minsearch index for the fetched documents
def build_index(documents):
    index = Index(
        text_fields=['question','section','answer'],
        keyword_fields=['course']
    )

    index.fit(documents)
    return index

# create and return sqlitesearch index for the fetched documents
def build_index_sqlite(docs_llm):
    index = TextSearchIndex(
        text_fields=['question','section','answer'],
        keyword_fields=['course'],
        db_path='faq.db'
    )

    # index.fit(documents)
    # illustrate a slow ingestion by adding the documents one by one with a small delay
    for idx, doc in enumerate(docs_llm, start=1):
        index.add(doc)
        if idx % 10 == 0:
            print(f'Added {idx} documents...')
        time.sleep(0.5)
    
    index.close() # close the underlying sql connection and release the resources
    print('Done. Index saved to faq.db')