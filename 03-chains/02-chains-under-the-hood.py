from dotenv import load_dotenv
from langchain_classic.prompts import ChatPromptTemplate
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

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x))
parse_output = RunnableLambda(lambda x: x.content)

chain = RunnableSequence(format_prompt, invoke_model, parse_output)
response = chain.invoke({"topic": "lawyers", "joke_count": 3})
print(response)


#runnableSequence ile birden fazla runnable'ı birleştirerek daha karmaşık görevleri yerine getirebiliriz. 
# Bu örnekte, format_prompt, invoke_model ve parse_output adında üç runnable'ı birleştiriyoruz. format_prompt, prompt_template'i kullanarak girdi verilerini formatlar. 
# invoke_model, model'i çağırarak yanıt alır. parse_output ise model'in yanıtını işleyerek sadece içeriği döndürür.   



#runnableParallel ile birden fazla runnable'ı paralel olarak çalıştırabiliriz. Bu, özellikle bağımsız görevleri aynı anda yürütmek istediğimizde faydalıdır.
#runnableConditional ile belirli koşullara bağlı olarak farklı runnable'ları çalıştırabiliriz. Bu, akış kontrolü ve karar verme süreçlerinde kullanışlıdır.
#runnablePassthrough ile girdiyi değiştirmeden bir runnable'dan diğerine geçirebiliriz. Bu, bazı durumlarda girdiyi korumak ve sadece yan etkiler için runnable'ları kullanmak istediğimizde yararlıdır.    
#runnableBranch ile belirli koşullara bağlı olarak farklı dallara ayrılabiliriz. Bu, daha karmaşık akışları yönetmek için kullanışlıdır. if else yani
