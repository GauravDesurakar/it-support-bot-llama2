import chainlit as cl
from model import func_qa_bot
from loguru import logger

# Chainlit
# Defining a decorator for a function called start(), which will be triggered when a chat session starts
@cl.on_chat_start
async def start():
    logger.debug("Into decorator async def start")
    chain = func_qa_bot() # Calling function
    msg = cl.Message(content="Bot is initiating...") # Object Initiated
    await msg.send()
    msg.content = "Greetings for the day! Welcome to the IT Support Bot. How I can help?"
    await msg.update()
    cl.user_session.set("chain", chain)


# Defining a decorator for a function named main() that will be triggered when a message is received in the chat.
@cl.on_message
async def main(message):
    logger.debug("Into decorator async def main")
    chain = cl.user_session.get("chain")
    logger.info(f"Chain :: {chain} ")
    # Callback handler
    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True,
                                          answer_prefix_tokens=["FINAL", "ANSWER"])
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    # sources = res["source_documents"] # Storing source doc page no info

    answer += f"\n\n If problem continues then please reach out to IT Support Desk."

    # if sources:
    #     answer += f"\n\nSources:\n" + str(sources)
    # else:
    #     answer += "\nNo sources found"

    logger.info(f"async def main, res:: {answer} ")
    await cl.Message(content=answer).send()