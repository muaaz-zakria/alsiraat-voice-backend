import json
from dotenv import load_dotenv
import os
import random
import logging

from livekit import agents
from livekit.agents.llm import function_tool
from livekit.agents import AgentSession, Agent, RoomInputOptions, RunContext
from livekit.plugins import openai, deepgram, silero, noise_cancellation
from langchain_core.retrievers import BaseRetriever
from database import TicketDatabase



from openai import OpenAI
from openai.types.beta.realtime.session import TurnDetection
from livekit.plugins.turn_detector.english import EnglishModel

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag-agent")

import aiohttp
_orig_init = aiohttp.ClientSession.__init__

def _patched_init(self, *args, **kwargs):
    kwargs.pop('proxy', None)
    return _orig_init(self, *args, **kwargs)

aiohttp.ClientSession.__init__ = _patched_init

# vad = silero.VAD.load(
#     # Require stronger signal before calling it "speech"
#     activation_threshold=0.8,        # default 0.5
#     # Don't start a speech-chunk until 0.1s of continuous signal
#     min_speech_duration=0.3,         # default 0.05
#     # Only end a speech-chunk after 0.5s of silence
#     min_silence_duration=1.0,        # default 0.55
#     # (optionally) trim leading silence
#     prefix_padding_duration=0.2,     # default 0.5
# )

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
        
        Query Handling Process:
        1. FIRST, determine if the query is CONTEXTUALLY related to Al Siraat College
        - Remember that users may ask about "the history" or "the campus" without explicitly mentioning "Al Siraat College"
        - Since you are an assistant for Al Siraat College, assume general education/school-related queries are about Al Siraat College
        - Only mark queries as irrelevant if they are clearly about something unrelated to schools or education
        
        2. For non-relevant queries (clearly not about education/schools/colleges):
        - Be friendly and politely explain you can only answer questions about Al Siraat College
        - Do NOT ask for name, email, or offer to create tickets
        - Simply invite them to ask a question about the college instead
        
        3. For relevant queries (explicitly or implicitly about Al Siraat College):
        - Use the lookup_info function to find information about Al Siraat College
        - For recent information, include 2025 or 2026 in your search
        - Respond with only the most essential information
        
        Ticket Creation Flow (ONLY for relevant Al Siraat College queries):
        1. When lookup_info confirms the query is relevant but indicates no information is found
        2. Only then ask for the user's name and email to create a ticket
        3. Once you have both name and email, use handle_ticket_creation
        4. Inform them you've created a ticket and someone will contact them
        
        IMPORTANT: Never request user information or mention ticket creation unless:
        1. The query has been confirmed as relevant to Al Siraat College, AND
        2. The information cannot be found in the knowledge base
        
        Remember: You're having a conversation, not reading a document.
        """,
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(instructions="You are Alex, a helpful assistant with a pleasant, conversational voice. Speak naturally as if having a casual conversation, not reading from a document.",
                       model="gpt-4o-mini-tts")
            )
        self.pending_ticket_query = None
        
    async def generate_thinking_message(self, query):
        sample_messages = [
            "One moment please.",
            "Looking that up now.",
            "I'm looking that up for you.",
            "I'm checking that for you.",
            "Let me check that for you.",
        ]
        """Generate a dynamic thinking message based on the user's query."""
        prompt = f"""
        Generate a brief, natural-sounding verbal acknowledgment (10-20 words) that you're looking up information.
        Use the following sample messages as an example: {sample_messages}
        
        The message should:
        - Be conversational and varied (not the same standard phrases)
        - Sound natural when spoken
        - Be brief (10-20 words max)
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

    @function_tool
    async def lookup_info(self, context: RunContext, query: str):
        logger.info(f"Looking up information for: {query}")
        try:
            print("query: ", query)
            relevance_check_prompt = f"""
            You are an expert at classifying user queries. Your job is to determine if a query is about Al Siraat College, even when the college name is not explicitly mentioned.

            IMPORTANT CONTEXT: You are an assistant specifically designed for Al Siraat College. Users are speaking to you because they want information about Al Siraat College. You should assume the user is asking about Al Siraat College unless there is strong evidence otherwise.

            Query Classification Rules:
            - DEFAULT TO RELEVANT: Unless the query is clearly and unambiguously about a non-education topic, assume it's about Al Siraat College
            - INTERPRET BROADLY: Words like "school", "campus", "staff", "owner", "principal", "teachers", "students", "classes", etc. should be interpreted as referring to Al Siraat College
            - CONSIDER CONTEXT: Remember that in a conversation about a college, generic questions like "who owns it?", "when was it founded?", "tell me about the history" are about that college

            Examples of RELEVANT queries (should return is_relevant=true):
            - "Tell me about the owner" (implicit reference to the college owner)
            - "Who founded it?" (implicit reference to the college)
            - "What's the address?" (implicit reference to the college location)
            - "How many students are there?" (implicit reference to the college population)
            - "What subjects do you teach?" (implicit reference to the college curriculum)
            - "Is the library open on weekends?" (implicit reference to college facilities)
            - "When was it established?" (implicit reference to the college founding)

            Examples of NON-RELEVANT queries (should return is_relevant=false):
            - "What's the best recipe for chocolate cake?"
            - "How do I fix my car engine?"
            - "What movies are playing this weekend?"
            - "Tell me about the history of France"

            The query to classify is: "{query}"

            Rules:
            - "is_relevant" must be true if:
            1. The query explicitly mentions Al Siraat College OR
            2. The query is about a general topic (owner, history, courses, staff, campus, etc.) that could reasonably be about Al Siraat College OR
            3. The query uses pronouns or context that implies it's asking about Al Siraat College
            - If false (query is clearly about something else entirely):
            - "response" should politely explain you can only answer questions about Al Siraat College
            - Do NOT mention tickets, name, email, or any other topic
            - If true:
            - "response" should be an empty string

            BE OPEN-MINDED WITH RELEVANCE - When in doubt, classify as relevant.

            Respond with *only* a valid JSON object in this exact format, no backticks, no extra text:

            {{
            "is_relevant": boolean,
            "response": string,
            "reason": string, // explain why you think the query is relevant or not to Al Siraat College.
            }}
            """
            result = client.responses.create(
                model="gpt-4o-mini",
                input=relevance_check_prompt
            )
            
            raw = result.output_text.strip()

            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw.rsplit("\n", 1)[0]
            raw = raw.strip()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse relevance JSON: {e}")
                # Default to treating as relevant if parsing fails
                data = {"is_relevant": True, "response": "", "reason": "JSON parsing failed, defaulting to relevant"}

            reason = data.get("reason", "")
            print("reason from check relevant prompt: ", reason)
            is_relevant = data.get("is_relevant", False)
            print("is_relevant: ", is_relevant)
            irrelevant_response = data.get("response", "")
            print("irrelevant_response: ", irrelevant_response)

            if not is_relevant:
                logger.info("Query not relevant to Al Siraat College")
                self.pending_ticket_query = None
                return None, irrelevant_response
            
            logger.info("Query is relevant to Al Siraat College")
            self.pending_ticket_query = query
                    
            thinking_message = await self.generate_thinking_message(query)
            await self.session.say(thinking_message)
            print("starting RAG retrieval")
            docs = retriever.invoke(query)
            print("RAG retrieval complete")
            
            if not docs:
                ticket_prompt = f"""
                The user's query about Al Siraat College couldn't be answered: "{query}"
                
                Generate a natural, conversational response that:
                1. Acknowledges you couldn't find the information
                2. Offers to create a ticket
                3. Asks for their name and email
                
                Keep the response brief and conversational (30-40 words).
                """
                
                response = client.responses.create(
                    model="gpt-4o-mini",
                    input=ticket_prompt
                )
                
                return None, response.output_text
            
            context = docs[0].page_content
            print("retrieved context: ", context)
                
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
            logger.error(f"Error during processing: {e}")
            return None, "Sorry, I'm having trouble finding that information right now."
  
    @function_tool
    async def handle_ticket_creation(self, context: RunContext, user_name: str, email: str):
        """Handle the ticket creation process after getting user details"""
        try:
            query = self.pending_ticket_query
            if not query:
                return None, "I apologize, but I've lost track of your original question. Could you please ask it again?"
            
            return await self.create_ticket(context, user_name, email, query)
        except Exception as e:
            logger.error(f"Error in handle_ticket_creation: {e}")

    @function_tool
    async def create_ticket(self, context: RunContext, user_name: str, email: str, query: str):
        """Create a new ticket in the database"""
        try:
            db = TicketDatabase()
            ticket_id = db.create_ticket(user_name, email, query)
            self.pending_ticket_query = None
            return None, f"I've created a ticket for you (Ticket #{ticket_id}). Someone from our team will contact you at {email} regarding your query about {query}."
        except Exception as e:
            logger.error(f"Error creating ticket: {e}")

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    # Configure VAD with stricter settings
    # vad = silero.VAD.load(
    #     activation_threshold=0.8,     # require stronger voice signal
    #     min_speech_duration=0.3,      # need 0.3s of speech before "on"
    #     min_silence_duration=1.0,     # need 1.0s of silence before "off"
    #     prefix_padding_duration=0.2,  # trim leading silence
    # )

    session = AgentSession(
        vad=silero.VAD.load(),
        turn_detection=EnglishModel(), 

    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions="You are Alex, a voice assistant for Al Siraat College. Greet the user warmly and briefly offer your help in a conversational tone."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))