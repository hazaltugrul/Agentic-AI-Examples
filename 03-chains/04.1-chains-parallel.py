from dotenv import load_dotenv
from langchain_classic.prompts import ChatPromptTemplate
from langchain_classic.schema.output_parser import StrOutputParser
from langchain_classic.schema.runnable import RunnableParallel
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

joke_prompt = ChatPromptTemplate.from_template("Tell a joke about {topic}.")

poem_prompt = ChatPromptTemplate.from_template("Write a poem about {topic}.")

parallel_chain = RunnableParallel(
    {
        "joke" : joke_prompt | model | StrOutputParser(),
        "poem" : poem_prompt | model | StrOutputParser()
    }
)

result = parallel_chain.invoke({"topic": "cats"})
print(result["joke"])
print(result["poem"], "\n")
print(result)
