INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''

USER_PROMPT_TEMPLATE = '''
Question:
{question}

Context:
{context}
'''.strip()

# class that contains RAG logic functions
# class is used because we can have different indices (example, elasticsearch) and llm clients (example, anthropic); we could just use the existing indexing and llm client and put all that
# in one file but then it becomes hard to reuse and adjust. So we put the dependencies inside a class instead. The index and the LLM client become constructor arguments. Now we can pass 
# any index or client we want when we create the object. And because it's a class, we can subclass it later to override one piece without touching the rest. For example, we can swap OpenAI 
# for a local model (inherit RAGBase class and overwrite with a different search method)
class RAGBase:
    
    def __init__(self, index, llm_client, instructions=INSTRUCTIONS, prompt_template=USER_PROMPT_TEMPLATE, course='llm-zoomcamp', model='gpt-4o-mini'):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.course = course
        self.model = model
    
    def search(self, question, num_results=5):
        boost_dict = {'question':3.0, 'section':0.5}
        filter_dict = {'course':self.course}

        return self.index.search(question, num_results=num_results, boost_dict=boost_dict, filter_dict=filter_dict)
    
    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc['section'])
            lines.append(f'Q: {doc['question']}')
            lines.append(f'A: {doc['answer']}')
            lines.append('')
        
        return '\n'.join(lines).strip()
    
    def build_prompt(self, question, search_results):
        context = self.build_context(search_results=search_results)
        return self.prompt_template.format(question=question, context=context)
    
    def llm(self, prompt):
        payload = [
            {'role':'developer', 'content':self.instructions},
            {'role':'user', 'content':prompt}
        ]
        response = self.llm_client.responses.create(
            model=self.model,
            input=payload
        )

        return response.output_text
    
    def rag(self, query):
        search_results = self.search(question=query)
        user_prompt = self.build_prompt(question=query, search_results=search_results)
        answer = self.llm(prompt=user_prompt)
        return answer
