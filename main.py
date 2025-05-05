from dotenv import load_dotenv
import os
import random
import logging

from livekit import agents
from livekit.agents.llm import function_tool
from livekit.agents import AgentSession, Agent, RoomInputOptions, RunContext
from livekit.plugins import openai, deepgram
from langchain_core.retrievers import BaseRetriever

from openai import OpenAI
from openai.types.beta.realtime.session import TurnDetection

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag-agent")

load_dotenv()

client = OpenAI()

pinecone_api_key = os.getenv("PINECONE_API_KEY")

model = OpenAIEmbeddings(
        model="text-embedding-3-small")


index_name = "alsiraat"

pc = Pinecone(api_key=pinecone_api_key)

# pc.create_index(
#     name=index_name,
#     dimension=1536, # Replace with your model dimensions
#     metric="cosine", # Replace with your model metric
#     spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     ) 
# )

namespace = "default_namespace"
index = pc.Index(index_name)

vectorstore = PineconeVectorStore(
                index=index, embedding=model, namespace=namespace
            )

retriever = vectorstore.as_retriever(
                search_type="similarity",
            )


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="""
        You are Alex, a helpful IT assistant for Al Siraat College.
        When users ask quesyions about Al Siraat College, use the lookup_info function to find relevant information.
        When asked about latest info, try adding 2025 or 2026 with the query.
        Keep your responses concise and to the point.
        Do not make anything up.
        Don't use unpronounceable characters in your responses since you communicate through voice.
                         
        if some info is not available, then ask the user their name and email, and tell them you have created a ticket and someone will reach out to them.
        """,
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o"),
        tts=openai.TTS(instructions="You are a helpful assistant with a pleasant voice.")
            )
        
        

    @function_tool
    async def lookup_info(self, context: RunContext, query: str):
        """
        Use this function to look up information using RAG when the user asks a question
        about a topic that might be in our knowledge base.
        
        Args:
            query: The question or topic to look up
        """
        logger.info(f"Looking up information for: {query}")
        
        # Tell the user we're looking things up
        thinking_messages = [
            "One moment while I look into this...",
            "Just a second while I check...",
        ]


        await self.session.say(random.choice(thinking_messages))
        
        try:
            
            docs = retriever.invoke(query)
            # Generate embeddings for the query
            # query_embedding = await openai.create_embeddings(
            #     input=[query],
            #     model=self._embeddings_model,
            #     dimensions=self._embeddings_dimension
            # )
            
            # Query the index
            # results = self._annoy_index.query(query_embedding[0].embedding, n=1)
            

            context = docs[0].page_content
            print("retrieved context: ", context)
            if not docs:
                return None, "I couldn't find any relevant information about that."
                
            # Get the most relevant paragraph
            # paragraph = self._paragraphs_by_uuid.get(results[0].userdata, "")
            
            if not context:
                return None, "I couldn't find any relevant information about that."
            
            # Generate response with context
            context_prompt = f"""
            Question: {query}
            
            Relevant information:
            {context}
            
            Using the relevant information above, please provide a helpful response to the question.
            Keep your response concise and directly answer the question.
            """
            
            #response = await self._llm.complete(context_prompt)

            response = client.responses.create(
                model="gpt-4o-mini",
                input=context_prompt
            )
            
            print("response: ", response.output_text)
            return None, response.output_text
            
        except Exception as e:
            logger.error(f"Error during RAG lookup: {e}")
            return None, "I encountered an error while trying to look that up."


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession()

    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(
    #         voice="coral",
    #         turn_detection=TurnDetection(
    #             type="semantic_vad",
    #             eagerness="auto",
    #             create_response=True,
    #             interrupt_response=True,
    #         ),
    #     )
    # )

    await session.start(
        room=ctx.room,
        agent=Assistant()
    )

    await session.generate_reply(
        instructions="You are Alex, an assistant for Al Siraat College. Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
