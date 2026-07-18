from assistant import setup, create_assistant
import streamlit as st

index, llm_client = setup()
assistant = create_assistant(index=index, llm_client=llm_client, model='gpt-4o-mini')

st.title(body='Course Assistant')

user_input = st.text_input(label='Enter your question:')

if st.button(label='Ask'):
    with st.spinner(text='Processing...'):
        answer = assistant.rag(query=user_input)
        st.success(body='Completed')
        st.write(answer)

        st.write(f"Response time: {assistant.last_call.response_time:.2f}s")
        st.write(f"Prompt tokens: {assistant.last_call.prompt_tokens}")
        st.write(f"Completion tokens: {assistant.last_call.completion_tokens}")
        st.write(f"Cost: ${assistant.last_call.cost:.4f}")