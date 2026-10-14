from dotenv import load_dotenv
from langchain_classic.prompts import ChatPromptTemplate
from langchain_classic.schema.output_parser import StrOutputParser
from langchain_classic.schema.runnable import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a expert product reviewer."),
        ("human", "Review the product: {product}"),
    ]
)

# Define pros analysis step
def analyze_pros(features):
    pros_template = ChatPromptTemplate. from_messages( 
        [
            ("system","You are an expert product reviewer."),
            ("human", "Given these features: {features}, list the pros of these features.") ]
    )
    return pros_template.format_prompt(features=features)

# Define cons analysis step
def analyze_cons(features):
    cons_template = ChatPromptTemplate. from_messages( 
        [
            ("system","You are an expert product reviewer."),
            ("human", "Given these features: {features}, list the cons of these features.") ]
    )
    return cons_template.format_prompt(features=features)


# combine pros and cons analysis into a final review
def combine_prod_cons(pros,cons):
    return f"Pros: {pros}\nCons: {cons}"

#runnableBranch ile belirli koşullara bağlı olarak farklı dallara ayrılabiliriz. Bu, daha karmaşık akışları yönetmek için kullanışlıdır. if else yani

# simplfy braches with LCEL
pros_branch_chain = (
    RunnableLambda(analyze_pros) | model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(analyze_cons) | model | StrOutputParser()   
)

# create the combined chain using LCEL
chain = (
    prompt_template 
    | model 
    | StrOutputParser() 
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: combine_prod_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

# run
result = chain.invoke({"product": "Apple iPhone 15 Pro Max"})
print(result)