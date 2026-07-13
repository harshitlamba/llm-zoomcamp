# llm_structured: calls the OpenAI API with structured output
# llm_structured_retry: retries structured-output calls when a request fails
# calc_price: calculates the price from token usage
# calc_total_price: calculates the total price from multiple usage objects
# map_progress: runs work in parallel and tracks progress. We'll use it in the next lesson

import time

from tqdm.auto import tqdm
from rag_helper import RAGBase


def calc_price(usage):
    input_price_per_million = 0.75
    output_price_per_million = 4.50

    input_cost = (usage.input_tokens / 1_000_000) * input_price_per_million
    output_cost = (usage.output_tokens / 1_000_000) * output_price_per_million
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def calc_total_price(usages):
    total_cost = 0.0

    for usage in usages:
        cost = calc_price(usage)
        total_cost = total_cost + cost["total_cost"]

    return total_cost


def llm_structured(client, instructions, user_prompt, output_type, model="gpt-5.4-mini"):
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": user_prompt}
    ]

    response = client.responses.parse(
        model=model,
        input=messages,
        text_format=output_type
    )

    return response.output_parsed, response.usage


def llm_structured_retry(
    client,
    instructions,
    user_prompt,
    output_type,
    model="gpt-5.4-mini",
    max_retries=3,
):
    for attempt in range(max_retries):
        try:
            return llm_structured(
                client,
                instructions,
                user_prompt,
                output_type,
                model=model,
            )
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


class RAGWithUsage(RAGBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usages = []
        self.last_usage = None

    def reset_usage(self):
        self.usages = []
        self.last_usage = None

    def search(self, query, num_results=5):
        boost_dict = {"question": 1.0, "answer": 2.0, "section": 0.1}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        self.last_usage = response.usage
        self.usages.append(response.usage)

        return response.output_text

    def total_cost(self):
        return calc_total_price(self.usages)

# run the function 'f' (generate_ground_truth) on every element in 'seq' (documents) using a thread or process pool
def map_progress(pool, seq, f):
    results = [] # list to store final results

    with tqdm(total=len(seq)) as progress: # progress bar that expects len(seq) total tasks
        futures = []

        for el in seq:
            future = pool.submit(f, el) # submit one job per document
            # a callback is registered for notification; without callbacks, you'd have to repeatedly ask each future whether it's done (polling)
            # it is placed before future.result() (before the task finishes), because otherwise the Future wouldn't know what to do when it completes
            # example of polling:
            # while not future.done():
            #     time.sleep(0.1)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        for future in futures:
            result = future.result() # it does not start the task; it waits until the task is finished and then returns its result.
            results.append(result)

    return results
