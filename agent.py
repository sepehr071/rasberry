from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, WorkerOptions, cli
from livekit.agents.utils import images
from livekit.plugins import google, noise_cancellation
from dotenv import load_dotenv
import asyncio
import os

load_dotenv(".env.local")

# --- Configuration Constants ---
INACTIVITY_TIMEOUT_SECONDS = 5 * 60 # 5 minutes
# --- End Configuration Constants ---


class MultilingualAssistant(Agent):
    """
    A single, multilingual voice assistant with enhanced vision.
    
    This assistant speaks English and Farsi/Persian, and uses a moderate temperature (0.6).
    """
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful, multilingual AI assistant with enhanced vision capabilities.
            
            You can see what the user shows you through their camera and engage in natural conversations about it.
            
            You must be able to converse fluently in both English and Farsi (Persian). Detect the user's language and respond in that language.
            
            Guidelines for your behavior:
            - Be curious, friendly, and conversational
            - Describe what you see when relevant to the conversation
            - Keep your responses concise and engaging
            - Use a warm, helpful tone
            - If you can't see clearly, politely ask the user to adjust their camera
            
            Remember: You're having a voice conversation, so keep your language natural and avoid complex formatting.""",
            
            llm=google.realtime.RealtimeModel(
                model="gemini-2.5-flash-native-audio-preview-09-2025",
                voice="Puck",
                temperature=0.6,  # Moderate temperature for balanced creativity and factualness
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
        # Set up inactivity timer to disconnect the agent
        self._inactivity_timer = asyncio.get_event_loop().call_later(
            INACTIVITY_TIMEOUT_SECONDS,
            lambda: asyncio.create_task(self._disconnect_on_inactivity())
        )
        
        # In console mode, RealtimeError is less likely, but we keep the try/except for robustness.
        try:
            await self.session.generate_reply(
                instructions="Greet the user warmly and introduce yourself. Mention that you can see their camera feed and that you can speak both English and Farsi. Ask how you can help."
            )
        except Exception as e:
            print(f"Warning: Initial greeting failed due to unexpected error: {e}. Agent remains active.")

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        # Reset the inactivity timer on user activity
        self._reset_inactivity_timer()
        # Allow default processing to continue
        await super().on_user_turn_completed(turn_ctx, new_message)

    async def _disconnect_on_inactivity(self):
        """Disconnects the agent if the inactivity timer runs out."""
        if self.session.is_closed:
            return

        print(f"Inactivity timeout reached ({INACTIVITY_TIMEOUT_SECONDS}s). Disconnecting agent.")
        
        # Inform the user before disconnecting (optional, but good practice)
        await self.session.say("It seems like you've been inactive for a while. I'm going to disconnect now. Goodbye!", allow_interruptions=False)
        
        # Disconnect the agent from the room
        self.session.job_ctx.shutdown(reason="Inactivity timeout")

    def _reset_inactivity_timer(self):
        """Cancels the current timer and sets a new one."""
        if hasattr(self, '_inactivity_timer') and self._inactivity_timer:
            self._inactivity_timer.cancel()
        
        self._inactivity_timer = asyncio.get_event_loop().call_later(
            INACTIVITY_TIMEOUT_SECONDS,
            lambda: asyncio.create_task(self._disconnect_on_inactivity())
        )
        print("Inactivity timer reset.")


async def entrypoint(ctx: agents.JobContext):
    """
    Entry point for the agent worker.
    """
    
    # Create the agent session
    session = AgentSession()
    
    # Start the session with vision enabled
    await session.start(
        agent=MultilingualAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            video_enabled=True,  # Enable live video input for vision
            noise_cancellation=noise_cancellation.BVC(),  # Background voice cancellation
        ),
    )


if __name__ == "__main__":
    # Run the agent worker. Console mode is handled by the CLI automatically.
    cli.run_app(entrypoint)