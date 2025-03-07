
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv

load_dotenv()
information = "John is a software engineer who loves to play the guitar and is a big fan of the Beatles."

if __name__ == "__main__" :
    print("Hello World")

    summary_template = """ given the information {information} abou t a person, I want you to create :
    1. a short summary
    2. two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(input_variables="information",template=summary_template, streaming=True)
    # temperature means creativity of the llm
    # llm = llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)
    llm = ChatOllama(model="deepseek-r1:1.5b", temperature=0)
    chain = summary_prompt_template | llm | StrOutputParser()
    for chunk in chain.stream(input={"information": information}):
        print(chunk, end="", flush=True)  # Print each chunk immediately

