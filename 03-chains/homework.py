from dotenv import load_dotenv
from langchain_classic.prompts import ChatPromptTemplate
from langchain_classic.schema.output_parser import StrOutputParser
from langchain_classic.schema.runnable import RunnableLambda, RunnableParallel, RunnableSequence
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

# --------------------------------------------------
# 1) Sequence chain örneği
# --------------------------------------------------
problem_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful tutor. Solve the problem clearly."),
        ("human", "Problem: {problem}"),
    ]
)

sequence_chain = problem_prompt | model | StrOutputParser()
problem_result = sequence_chain.invoke({
    "problem": "Bir öğrenci 2 saatlik ders için 4 sayfa not tutuyor. Her sayfada ortalama 300 kelime yazıyorsa toplam kaç kelime yazar?"
})

print("=== 1) CHAIN + SEQUENCE ===")
print(problem_result)
print()

# --------------------------------------------------
# 2) RunnableParallel örneği
# --------------------------------------------------
# 2 tane farklı prompt oluşturup aynı input ile paralel olarak çalıştırıyoruz.
# Amaç: aynı topic için hem şaka hem şiir üretmek.

joke_prompt = ChatPromptTemplate.from_template("Tell me a funny joke about {topic}.")
poem_prompt = ChatPromptTemplate.from_template("Write a short poem about {topic}.")

parallel_chain = RunnableParallel(
    {
        "joke": joke_prompt | model | StrOutputParser(),
        "poem": poem_prompt | model | StrOutputParser(),
    }
)

parallel_result = parallel_chain.invoke({"topic": "cats"})

print("=== 2) RUNNABLEPARALLEL ===")
print("JOKE:")
print(parallel_result["joke"])
print()
print("POEM:")
print(parallel_result["poem"])
print()

# --------------------------------------------------
# 3) RunnableParallel kullanılmadan RunnableSequence / RunnableLambda ile çözüm
# --------------------------------------------------
# Burada input'u düzenleyip ardından tek bir chain ile işlem yaptırıyoruz.

prepare_input = RunnableLambda(lambda x: {"topic": x["topic"], "joke_count": x["joke_count"]})

single_prompt = ChatPromptTemplate.from_template(
    "Tell me {joke_count} jokes about {topic}."
)

runnable_sequence_chain = RunnableSequence(
    prepare_input,
    single_prompt | model | StrOutputParser(),
)

runnable_result = runnable_sequence_chain.invoke({"topic": "dogs", "joke_count": 2})

print("=== 3) RUNNABLE (without RunnableParallel) ===")
print(runnable_result)
print()

# --------------------------------------------------
# Ödev notu:
# 1) 1 tane chain ile sequence yapıldı.
# 2) 2 tane problem belirlenip template oluşturuldu.
# 3) Sorun RunnableParallel ile çözüldü.
# 4) Aynı işi RunnableParallel kullanmadan RunnableSequence ile de çözdük.
# --------------------------------------------------
