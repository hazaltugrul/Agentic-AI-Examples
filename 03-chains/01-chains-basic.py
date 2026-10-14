from dotenv import load_dotenv
from langchain_classic.prompts import ChatPromptTemplate
from langchain_classic.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

# https://www.geeksforgeeks.org/artificial-intelligence/1lm-chains/
# https://www.geeksforgeeks.org/data-science/custom-chains-in-langchain/
# https://milvus.io/ai-quick-reference/what-are-chains-in-langchain-and-how-do-they-function

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

#her bir chain llm gibi davranır yani invoke edilerek  çalıştırlabilir. 
#Chainler birden fazla llm ve promptu birleştirerek daha karmaşık görevleri yerine getirebilir. Chainler ayrıca birden fazla adım içerebilir ve her adımın çıktısı bir sonraki adıma girdi olarak kullanılabilir.    
#normalde invoke methodu tek bir promptu çalıştırır. Chainler ise birden fazla promptu ve llm'i birleştirerek daha karmaşık görevleri yerine getirebilir.
##llm chainin componentidir.

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {jokes}"),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

chain = prompt_template | model | StrOutputParser()
#chain = prompt_template | model  yani buradaki chain | ile oluşturulur.

result = chain.invoke({"jokes": "lawyers", "joke_count": 3})
print(result)