from dotenv import load_dotenv
from langchain_classic.prompts import ChatPromptTemplate
from langchain_classic.schema.output_parser import StrOutputParser
from langchain_classic.schema.runnable import RunnableBranch
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a thank you message for the following positive feedback: {feedback}"),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a response for the following negative feedback: {feedback}"),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a request for more details for this neutral feedback: {feedback}"),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a message to escalate this feedback to a human agent: {feedback}"),
    ]
)

classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Classify the following feedback as positive, negative, or neutral: {feedback}"),
    ]
)

classification_chain = classification_template | model | StrOutputParser()

branches = RunnableBranch(
    (lambda x: "positive" in x.lower(), positive_feedback_template | model | StrOutputParser()),
    (lambda x: "negative" in x.lower(), negative_feedback_template | model | StrOutputParser()),
    (lambda x: "neutral" in x.lower(), neutral_feedback_template | model | StrOutputParser()),
    escalate_feedback_template | model | StrOutputParser(),
)

chain = classification_chain | branches

review = "The product is great, but the delivery was late."
result = chain.invoke({"feedback": review})
print(result)