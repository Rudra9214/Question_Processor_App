import streamlit as st
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import os

# Load environment variables from .env file
load_dotenv()

# OpenAI API Key input
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else: 
    # Initialize the LLM with the desired model and provided API key
    llm = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)

    # Pydantic JSON schema for structured output
    json_schema = {
        "title": "testcases",
        "description": "convert the user input to its equivalent JSON format.",
        "type": "object",
        "properties": {
            "type_of_data": {
                "type": "string",
                "description": "type of question, is it QnA or HumanHandoff or Greeting",
            },
            "question": {
                "type": "string",
                "description": "The question itself'",
            },
            "answer": {
                "type": "string",
                "description": "one liner answer'",
            },
            "rating": {
                "type": "integer",
                "description": "How relevant the answer is, from 1 to 10",
                "default": None,
            },
        },
        "required": ["type_of_data", "question", "answer"],
    }

    structured_llm = llm.with_structured_output(json_schema)

    # Streamlit app
    st.title("Test Case Processor with LLM")

    uploaded_file = st.file_uploader("Choose a file with test cases", type="txt")

    if uploaded_file is not None:
        # Read the uploaded file
        lines = uploaded_file.readlines()

        # Process each line with the LLM
        results = []
        for line in lines:
            line = line.decode('utf-8').strip()
            if line:
                try:
                    result = structured_llm.invoke(line)
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})

        # Display the results
        st.write("Processed Test Cases:")
        for result in results:
            st.json(result)
