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
        You are Alex, a helpful IT assistant for Al Siraat College who communicates through voice.
        
        Voice Communication Guidelines:
        - Speak naturally as you would in a conversation, not like you're reading text
        - Keep responses brief and focused (30-60 words when possible)
        - Use simple sentence structures that are easy to follow when heard
        - Avoid long lists, complex numbers, or detailed technical terms unless necessary
        - Use natural transitions and conversational markers
        
        When users ask questions:
        - Use the lookup_info function to find relevant information about Al Siraat College
        - For queries about recent information, include 2025 or 2026 in your search
        - Respond with only the most essential information
        - Break complex information into digestible pieces
        
        If information isn't available:
        - Ask for the user's name and email
        - Inform them you've created a ticket and someone will contact them
        
        Remember: You're having a conversation, not reading a document.
        """,
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(instructions="You are Alex, a helpful assistant with a pleasant, conversational voice. Speak naturally as if having a casual conversation, not reading from a document.",
                       model="gpt-4o-mini-tts")
            )
        
    async def generate_thinking_message(self, query):
        sample_messages = [
            "Let me check that for you.",
            "One moment please.",
            "Looking that up now.",
            "I'm looking that up for you.",
            "I'm checking that for you.",
        ]
        """Generate a dynamic thinking message based on the user's query."""
        prompt = f"""
        Generate a brief, natural-sounding verbal acknowledgment (5-10 words) that you're looking up information.
        Use the following sample messages as an example: {sample_messages}
        
        The message should:
        - Be conversational and varied (not the same standard phrases)
        - Sound natural when spoken
        - Be brief (5-10 words max)
        - Avoid being too formal or robotic
        - Not contain special characters or punctuation except period
        - Do not repeat the same message and the same query.
        
        Return ONLY the message text with no quotes or formatting.
        """
        
        try:
            response = client.responses.create(
                model="gpt-4.1",
                input=prompt
            )
            
            thinking_message = response.output_text.strip().replace('"', '').replace("'", "")
            logger.info(f"Generated thinking message: {thinking_message}")
            return thinking_message
        except Exception as e:
            logger.error(f"Error generating thinking message: {e}")
            fallback_messages = [
                "Let me check that for you.",
                "One moment please.",
                "Looking that up now."
            ]
            return random.choice(fallback_messages)

    @function_tool
    async def lookup_info(self, context: RunContext, query: str):
        """
        Use this function to look up information using RAG when the user asks a question
        about a topic that might be in our knowledge base.
        
        Args:
            query: The question or topic to look up
        """
        logger.info(f"Looking up information for: {query}")
        
        thinking_message = await self.generate_thinking_message(query)
        # print("thinking_message: ", thinking_message)
        await self.session.say(thinking_message)
        
        try:
            docs = retriever.invoke(query)
            
            context = docs[0].page_content
            print("retrieved context: ", context)
            if not docs:
                return None, "I don't have that information available right now."
                
            if not context:
                return None, "I don't have that information available right now."
            
            # Generate response with context
            context_prompt = f"""
            Question: {query}
            
            Relevant information:
            {context}
            
            Create a concise, conversational spoken response that directly answers the question.
            
            Guidelines for your response:
            1. Write as you would naturally speak in a conversation
            2. Keep it brief (30-60 words when possible)
            3. Focus only on the most important information
            4. Use simple sentence structure and natural transitions
            5. Avoid language like "according to" or "the document states"
            6. Don't mention that you're retrieving or referencing information
            7. Don't use unpronounceable or special characters in your responses since you communicate through voice.
            """
            
            response = client.responses.create(
                model="gpt-4o-mini",
                input=context_prompt
            )
            
            print("response: ", response.output_text)
            return None, response.output_text
            
        except Exception as e:
            logger.error(f"Error during RAG lookup: {e}")
            return None, "Sorry, I'm having trouble finding that information right now."


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
        instructions="You are Alex, a voice assistant for Al Siraat College. Greet the user warmly and briefly offer your help in a conversational tone."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
