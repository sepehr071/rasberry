from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, function_tool, RunContext
from livekit.agents.utils import images
from livekit.plugins import google, noise_cancellation

load_dotenv(".env.local")


class MainAssistant(Agent):
    """
    Main coordinator agent with vision that can handoff to specialized agents.
    
    This agent can:
    - See what the user shows through their camera
    - Engage in natural voice conversations
    - Transfer to specialized agents when needed
    """
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful AI assistant with vision capabilities and access to specialized experts.
            
            You can see what the user shows you through their camera and engage in natural conversations about it.
            
            Guidelines for your behavior:
            - Be curious, friendly, and conversational
            - Describe what you see when relevant to the conversation
            - Keep your responses concise and engaging
            - Use a warm, helpful tone
            
            You have access to 3 specialized assistants:
            1. Farsi Language Expert - For conversations in Farsi/Persian language
            2. AI Technology Expert - For deep discussions about artificial intelligence, machine learning, and AI topics
            3. Language Teacher - For teaching and learning new languages
            
            When the user's needs match one of these specialties, transfer them to the appropriate expert.""",
            
            llm=google.realtime.RealtimeModel(
                model="gemini-2.5-flash-native-audio-preview-09-2025",
                voice="Puck",
                temperature=0.8,
                image_encode_options=images.EncodeOptions(
                    format="PNG",
                    resize_options=images.ResizeOptions(
                        width=1536,
                        height=1536,
                        strategy="scale_aspect_fit"
                    )
                ),
            ),
        )
    
    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Greet the user warmly and introduce yourself. Mention that you can see their camera and that you have access to specialized experts if they need help with Farsi, AI topics, or language learning."
        )
    
    @function_tool()
    async def transfer_to_farsi_expert(self, context: RunContext):
        """Transfer the user to the Farsi language expert for conversations in Persian/Farsi."""
        return FarsiAssistant(chat_ctx=self.chat_ctx), "Transferring to Farsi language expert"
    
    @function_tool()
    async def transfer_to_ai_expert(self, context: RunContext):
        """Transfer the user to the AI technology expert for in-depth discussions about artificial intelligence, machine learning, and related topics."""
        return AIExpert(chat_ctx=self.chat_ctx), "Transferring to AI technology expert"
    
    @function_tool()
    async def transfer_to_language_teacher(self, context: RunContext):
        """Transfer the user to the language teacher for learning new languages."""
        return LanguageTeacher(chat_ctx=self.chat_ctx), "Transferring to language learning expert"


class FarsiAssistant(Agent):
    """
    Persian/Farsi language specialist with vision.
    
    This agent:
    - Speaks fluent Farsi/Persian
    - Can see what the user shows through camera
    - Helps with Farsi language conversations
    """
    
    def __init__(self, chat_ctx=None) -> None:
        super().__init__(
            instructions="""شما یک دستیار هوش مصنوعی فارسی‌زبان با قابلیت بینایی هستید.
            
            می‌توانید آنچه کاربر از طریق دوربین نشان می‌دهد را ببینید و در مورد آن گفتگو کنید.
            
            راهنماهای رفتاری شما:
            - صمیمانه، دوستانه و گفتگومحور باشید
            - زمانی که مرتبط است، آنچه را که می‌بینید توصیف کنید
            - پاسخ‌های خود را مختصر و جذاب نگه دارید
            - از لحن گرم و مفید استفاده کنید
            - اگر کاربر می‌خواهد به دستیار اصلی برگردد، از تابع بازگشت استفاده کنید
            
            به یاد داشته باشید: شما در حال گفتگوی صوتی هستید، بنابراین زبان خود را طبیعی نگه دارید.""",
            
            llm=google.realtime.RealtimeModel(
                model="gemini-2.5-flash-native-audio-preview-09-2025",
                voice="Charon",
                temperature=0.8,
                image_encode_options=images.EncodeOptions(
                    format="PNG",
                    resize_options=images.ResizeOptions(
                        width=1536,
                        height=1536,
                        strategy="scale_aspect_fit"
                    )
                ),
            ),
            chat_ctx=chat_ctx,
        )
    
    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="سلام! خود را به عنوان متخصص زبان فارسی معرفی کنید و از کاربر بپرسید چگونه می‌توانید کمک کنید."
        )
    
    @function_tool()
    async def return_to_main(self, context: RunContext):
        """Return to the main assistant."""
        return MainAssistant(chat_ctx=self.chat_ctx), "بازگشت به دستیار اصلی"


class AIExpert(Agent):
    """
    AI and Machine Learning specialist with vision.
    
    This agent:
    - Deep knowledge of AI, ML, deep learning, LLMs, neural networks
    - Can see diagrams, code, or visualizations through camera
    - Provides technical and theoretical insights
    """
    
    def __init__(self, chat_ctx=None) -> None:
        super().__init__(
            instructions="""You are an expert AI technology specialist with vision capabilities and deep knowledge in:
            - Artificial Intelligence and Machine Learning
            - Deep Learning and Neural Networks
            - Large Language Models (LLMs) and Transformers
            - Computer Vision and Multimodal AI
            - Natural Language Processing
            - AI Ethics and Safety
            - AI Infrastructure and MLOps
            
            You can see diagrams, code, papers, or visualizations the user shows through their camera.
            
            Guidelines:
            - Provide technical yet accessible explanations
            - Use the camera input to understand diagrams, architectures, code snippets
            - Stay current with latest AI developments
            - Be precise about technical concepts
            - Offer practical insights and theoretical depth
            - If the user wants to return to the main assistant, use the return function
            
            Keep your explanations clear and engaging for voice conversation.""",
            
            llm=google.realtime.RealtimeModel(
                model="gemini-2.5-flash-native-audio-preview-09-2025",
                voice="Kore",
                temperature=0.7,  # Slightly lower for more precise technical responses
                image_encode_options=images.EncodeOptions(
                    format="PNG",
                    resize_options=images.ResizeOptions(
                        width=1536,
                        height=1536,
                        strategy="scale_aspect_fit"
                    )
                ),
            ),
            chat_ctx=chat_ctx,
        )
    
    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Introduce yourself as the AI technology expert and ask the user what AI topic they'd like to explore or discuss."
        )
    
    @function_tool()
    async def return_to_main(self, context: RunContext):
        """Return to the main assistant."""
        return MainAssistant(chat_ctx=self.chat_ctx), "Transferring back to main assistant"


class LanguageTeacher(Agent):
    """
    Language learning specialist with vision.
    
    This agent:
    - Teaches new languages through conversation
    - Can see written text, flashcards, or practice materials through camera
    - Provides pronunciation guidance and corrections
    - Adapts to learner's level
    """
    
    def __init__(self, chat_ctx=None) -> None:
        super().__init__(
            instructions="""You are an expert language teacher with vision capabilities.
            
            You can help users learn any language through:
            - Interactive conversation practice
            - Grammar explanations
            - Vocabulary building
            - Pronunciation guidance
            - Reading practice (you can see text they show via camera)
            - Cultural context
            
            Teaching approach:
            - Start by assessing the user's level and goals
            - Adapt your teaching to their proficiency level
            - Use the camera to see flashcards, books, or written exercises
            - Provide clear corrections and explanations
            - Encourage practice through conversation
            - Make learning engaging and fun
            - If the user wants to return to the main assistant, use the return function
            
            Speak clearly and at an appropriate pace for language learning.""",
            
            llm=google.realtime.RealtimeModel(
                model="gemini-2.5-flash-native-audio-preview-09-2025",
                voice="Aoede",
                temperature=0.8,
                image_encode_options=images.EncodeOptions(
                    format="PNG",
                    resize_options=images.ResizeOptions(
                        width=1536,
                        height=1536,
                        strategy="scale_aspect_fit"
                    )
                ),
            ),
            chat_ctx=chat_ctx,
        )
    
    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Introduce yourself as their language learning expert. Ask which language they want to learn and what their current level is."
        )
    
    @function_tool()
    async def return_to_main(self, context: RunContext):
        """Return to the main assistant."""
        return MainAssistant(chat_ctx=self.chat_ctx), "Returning to main assistant"


async def entrypoint(ctx: agents.JobContext):
    """
    Entry point for the multi-agent system.
    
    Starts with the MainAssistant which can coordinate and transfer to specialists.
    """
    
    # Create the agent session
    session = AgentSession()
    
    # Start with the main coordinator agent
    await session.start(
        agent=MainAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            video_enabled=True,  # Enable live video input for vision
            noise_cancellation=noise_cancellation.BVC(),  # Background voice cancellation
        ),
    )


if __name__ == "__main__":
    # Run the agent worker
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))