import os
import pandas as pd
import streamlit as st
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from pandasai.responses.response_parser import ResponseParser

key= os.environ["OPENAI_API_KEY"] 
llm = OpenAI(api_token=key)

class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_df(self, result):
        st.dataframe(result["value"])
        return
    
    def format_plot(self, result):
        st.image(result["value"])
        return
    
    def format_other(self, result):
        st.write(result["value"])
        return

st.title("PandasAI Chatbot")

file = st.file_uploader("Upload file:", type=['csv'])

if file is not None:
    df = pd.read_csv(file)
    st.write(df.head(10))

    prompt = st.text_input("enter your prompt")
    if st.button("GO"):
        if prompt:
            with st.spinner("Generating Response..."):
                pandas_ai = SmartDataframe(df, config={"llm": llm, "response_parser": StreamlitResponse,})
                pandas_ai.chat(prompt)
                

        else:
            st.warning("Please enter prompt")