# to monitor the system, we capture the metadata info per LLM call
from dataclasses import dataclass, field
from datetime import datetime
from rag_helper import RAGBase
import time

@dataclass
class LLMClassRecord:
    model: str
    prompt: str
    instructions: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    cost: float
    # timestamp is a field of type datetime - if no value is given when creating the object, call datetime.now() at that moment 
    # to generate the default
    timestamp: datetime = field(default_factory=datetime.now)

class RAGWithMetrics(RAGBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_call: LLMClassRecord = None
    
    # sends request to llm
    def _call_llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]
        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )
        return response
    
    def _log_response(self, prompt, response, response_time):
        usage = response.usage
        cost = calculate_cost(model=self.model, usage=usage)

        call_record = LLMClassRecord(
            model=self.model,
            prompt=prompt,
            instructions=self.instructions,
            answer=response.output_text,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            response_time=response_time,
            cost=cost
        )
        self.last_call = call_record

    def llm(self, prompt):
        start_time = time.time()
        response = self._call_llm(prompt)
        response_time = time.time() - start_time
        self._log_response(prompt, response, response_time)
        return response.output_text

# calculate the cost per call
def calculate_cost(model, usage):
    cost = 0
    if 'gpt-5.4-mini' in model:
        # provider charges price per million input/output tokens - we multiply each count by its rate and divide by a million
        # the usage object comes straight from the LLM response - it carries the token counts for the call
        cost = (usage.input_tokens * 0.75 + usage.output_tokens * 4.50) / 1_000_000
    else:
        cost = (usage.input_tokens * 0.15 + usage.output_tokens * 0.60) / 1_000_000
    return cost