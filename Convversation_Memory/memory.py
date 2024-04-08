import os
from dotenv import load_dotenv
from langchain.llms import GooglePalm
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.schema import (SystemMessage,HumanMessage,AIMessage,)
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

def main():
    load_dotenv()
    palm_key = os.getenv("PALM_API_KEY")
    if palm_key is None:
        print("Please add API key into .env file")
        exit(1)
    else:
        print("API key loaded")

    model       = GooglePalm(google_api_key=os.getenv("PALM_API_KEY"),temperature=0.25)
    memory      = ConversationBufferMemory()
    conversation= ConversationChain(llm=model,
                                 verbose=True,
                                 memory=memory)
    message = [
        SystemMessage(content="You are a helpful assistant"),
    ]

    while True:
        user_input = input("> ")
        ai_response = conversation(inputs=user_input)
        print(ai_response)
        #message.append(HumanMessage(content=user_input),AIMessage(content=ai_response))
        if user_input.lower() == 'bye jarvis' or user_input.lower() == 'goodbye':
            print('Goodbye!')
            break






if __name__ == "__main__":
    main()

