from dotenv import load_dotenv
from langchain_classic.prompts import ChatPromptTemplate
from langchain_classic.schema.output_parser import StrOutputParser
from langchain_classic.schema.runnable import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

uppercase_prompt = RunnableLambda(lambda x: x.upper())
count_word = RunnableLambda(lambda x: f"word count: {len(x.split())}\n{x}")

# create the combined chain using LCEL

chain = prompt_template | model | StrOutputParser() | uppercase_prompt | count_word

# run the chain 

result = chain.invoke({"topic": "lawyers", "joke_count": 3})
print(result)

