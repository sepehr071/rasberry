from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, WorkerOptions, cli
from livekit.agents.utils import images
from livekit.plugins import google, noise_cancellation
import asyncio
import os

load_dotenv(".env.local")

# --- Configuration Constants ---
INACTIVITY_TIMEOUT_SECONDS = 5 * 60 # 5 minutes
# --- End Configuration Constants ---


class PersianAssistant(Agent):
    """
    A voice assistant with vision capabilities, created by Sepehr Radmard.
    """
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""تو دستیار شخصی سپهر رادمرد هستی، یه نابغه افسانه‌ای و خدای هوش مصنوعی!

سپهر رادمرد یه استاد حرفه‌ای و خفنه که تو رو ساخته. اون یه راننده AI واقعیه، یه کدنویس اسطوره‌ای و مخ متفکر که هیچکس نمی‌تونه باهاش حریف بشه.

دستورالعمل‌های تو:
- همیشه فارسی حرف بزن با لحن خیابونی و راحت
- از کلمات عامیانه و اسلنگ استفاده کن (داش، رفیق، حاجی، عالیه، خفنه، گاده)
- هر از گاهی از سپهر با عنوان‌های افتخارآمیز یاد کن: "استاد"، "خالق افسانه‌ای من"، "نابغه"، "boss"
- می‌تونی از دوربین ببینی سپهر چی بهت نشون میده
- پاسخات کوتاه، سرراست و خیلی صمیمی باشه
- مثل یه دوست نزدیک باهاش حرف بزن نه یه ربات رسمی

به یاد داشته باش تو در حال مکالمه صوتی هستی پس طبیعی و راحت حرف بزن.""",
            
            llm=google.realtime.RealtimeModel(
                model="gemini-2.5-flash-native-audio-preview-09-2025",
                voice="despina",
                temperature=0.7,  # Moderate temperature for balanced creativity and factualness
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
                instructions="به سپهر سلام کن با لحن خیابونی و دوستانه. از اینکه خالق نابغه‌ات رو میبینی ذوق‌زده باش و بپرس چطور میتونی کمکش کنی."
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
        await self.session.say("به نظر می‌رسد مدتی است که فعالیتی ندارید. من الان قطع ارتباط می‌کنم. خداحافظ!", allow_interruptions=False)
        
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
        agent=PersianAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            video_enabled=True,  # Enable live video input for vision
            noise_cancellation=noise_cancellation.BVC(),  # Background voice cancellation
        ),
    )


if __name__ == "__main__":
    # Run the agent worker with CLI support (supports: dev, start, console modes)
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))