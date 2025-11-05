# AI Coding Assistant Guide for Voice Agent with Vision

This document is optimized for AI coding assistants (like Claude, GPT-4, etc.) to help them understand and work effectively with this LiveKit voice agent project.

## Project Overview

**Type**: Voice AI Agent with Vision Capabilities
**Framework**: LiveKit Agents (Python)
**AI Model**: Google Gemini Live API (gemini-2.5-flash-native-audio-preview-09-2025)
**Purpose**: General-purpose voice assistant that can see and discuss what users show via camera

## Architecture

```
                   LiveKit WebRTC
User Camera/Mic <=================> LiveKit Room <=> Agent Worker <=> Gemini Live API
                   Low latency                        Python          Multimodal AI
                                                       
                   Text Streams
Frontend UI    <=================> LiveKit Room <=> Agent Worker
                   Transcriptions                     RoomIO
```

## Key Technologies

1. **LiveKit Agents SDK v1.2+**: Orchestration framework
2. **Google Gemini Live API**: Realtime multimodal model (speech-to-speech + vision)
3. **LiveKit WebRTC**: Low-latency audio/video transport
4. **Python 3.9+**: Runtime environment

## File Structure

- [`agent.py`](agent.py): Main agent implementation with `VisionAssistant` class
- [`requirements.txt`](requirements.txt): Python dependencies
- `.env.local`: Environment variables (not committed, use `.env.local.example`)
- [`.env.local.example`](.env.local.example): Template for environment setup
- [`.gitignore`](.gitignore): Git ignore patterns
- [`README.md`](README.md): User-facing documentation

## Core Components

### 1. VisionAssistant Class

Location: [`agent.py`](agent.py:13-46)

```python
class VisionAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="...",
            llm=google.realtime.RealtimeModel(...)
        )
```

**What it does**:
- Inherits from `Agent` base class
- Defines agent personality via `instructions`
- Configures Gemini Live API with voice and temperature

**Key parameters**:
- `instructions`: System prompt for agent behavior
- `model`: Gemini model identifier
- `voice`: Voice personality (Puck, Charon, Kore, Fenrir, Aoede)
- `temperature`: Response randomness (0.0-1.0)

### 2. Entrypoint Function

Location: [`agent.py`](agent.py:49-68)

```python
async def entrypoint(ctx: agents.JobContext):
    session = AgentSession()
    await session.start(
        agent=VisionAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            video_enabled=True,
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
```

**What it does**:
- Creates an `AgentSession` to manage the conversation
- Starts the session with vision enabled
- Configures noise cancellation

**Critical settings**:
- `video_enabled=True`: Enables live video input for vision
- `noise_cancellation.BVC()`: Background voice cancellation
- Alternative for telephony: `noise_cancellation.BVCTelephony()`

## LiveKit Agents Framework Concepts

### AgentSession
- Main orchestrator for voice AI
- Manages user input, voice pipeline, LLM invocation, and output
- Handles turn detection automatically

### RoomInputOptions
- Configures how the agent receives input
- `video_enabled`: Enable/disable video input
- `audio_enabled`: Enable/disable audio input
- `text_enabled`: Enable/disable text chat
- `noise_cancellation`: Audio preprocessing

### RoomIO
- Bridge between AgentSession and LiveKit room
- Manages track subscriptions/publications
- Created automatically by AgentSession

## Vision System

### How It Works

1. **Video Sampling**: 
   - Default: 1 frame per second while user speaks
   - 1 frame per 3 seconds otherwise
   - Automatic adjustment based on speech activity

2. **Frame Processing**:
   - Frames resized to 1024x1024
   - Encoded to JPEG
   - Sent to Gemini along with audio

3. **Context Window**:
   - Video frames added to conversation context
   - Gemini processes both audio and visual information simultaneously

### Customization Options

**Change frame rate** (requires custom implementation):
```python
# Example: Custom video sampler
session = AgentSession(
    video_sampler=CustomVideoSampler(fps=2)  # 2 frames per second
)
```

**Change frame encoding**:
```python
from livekit.agents.utils import images

image_bytes = images.encode(
    frame,
    images.EncodeOptions(
        format="PNG",
        resize_options=images.ResizeOptions(
            width=768,
            height=768,
            strategy="scale_aspect_fit"
        )
    )
)
```

## Environment Variables

Required in `.env.local`:

```bash
GOOGLE_API_KEY=            # From https://aistudio.google.com/app/apikey
LIVEKIT_API_KEY=          # From LiveKit Cloud project
LIVEKIT_API_SECRET=       # From LiveKit Cloud project
LIVEKIT_URL=              # wss://your-project.livekit.cloud
```

Optional for Vertex AI:
```bash
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

## Common Modifications

### 1. Change Agent Voice

Edit [`agent.py`](agent.py:38):
```python
voice="Aoede",  # Options: Puck, Charon, Kore, Fenrir, Aoede
```

### 2. Adjust Response Temperature

Edit [`agent.py`](agent.py:39):
```python
temperature=0.8,  # Range: 0.0 (deterministic) to 1.0 (creative)
```

### 3. Modify Agent Instructions

Edit [`agent.py`](agent.py:17-32):
```python
instructions="""Your custom system prompt here"""
```

### 4. Add Function Tools

```python
from livekit.agents import function_tool

class VisionAssistant(Agent):
    @function_tool()
    async def analyze_object(self, object_name: str) -> str:
        """Analyze a specific object in the video"""
        # Your implementation
        return f"Analyzing {object_name}..."
```

### 5. Disable Video Input

Edit [`agent.py`](agent.py:61):
```python
room_input_options=RoomInputOptions(
    video_enabled=False,  # Disable vision
    noise_cancellation=noise_cancellation.BVC(),
)
```

## Turn Detection

Current setup uses Gemini's built-in VAD-based turn detection (automatic).

**To use LiveKit's turn detector instead**:

```python
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from google.genai import types

session = AgentSession(
    turn_detection=MultilingualModel(),
    llm=google.realtime.RealtimeModel(
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=True,  # Disable Gemini's built-in turn detection
            ),
        ),
    ),
    stt="assemblyai/universal-streaming",  # Required for LiveKit turn detection
)
```

## Transcription System

- Automatically enabled by default
- Uses `lk.transcription` text stream topic
- Synchronized with audio playback
- Includes user and agent speech

**Access transcriptions in code**:
```python
class VisionAssistant(Agent):
    async def on_conversation_item_added(self, item):
        print(f"New transcript: {item.text}")
```

## Running the Agent

### Development Mode
```bash
python agent.py dev
```
- Connects to LiveKit Cloud
- Provides playground URL
- Hot reloads on code changes

### Console Mode (audio only)
```bash
python agent.py console
```
- Terminal-based testing
- No video support in this mode

### Production Mode
```bash
python agent.py start
```
- Optimized for production
- No hot reload

## Debugging Tips

### Enable Debug Logging
```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("voice-agent")
```

### Check Video Feed
```python
class VisionAssistant(Agent):
    async def on_enter(self):
        room = get_job_context().room
        for participant in room.remote_participants.values():
            for pub in participant.track_publications.values():
                if pub.track.kind == rtc.TrackKind.KIND_VIDEO:
                    logger.info(f"Video track found: {pub.track.sid}")
```

### Monitor Token Usage
Gemini Live API charges per token. Monitor usage through Google Cloud Console or AI Studio.

## Common Issues

### 1. No Video Input
**Problem**: Agent doesn't process video
**Solutions**:
- Verify `video_enabled=True` in `RoomInputOptions`
- Check frontend is publishing video track
- Ensure camera permissions granted

### 2. Slow Responses
**Problem**: High latency in agent responses
**Solutions**:
- Reduce video frame rate
- Lower image resolution
- Check network connectivity
- Adjust temperature parameter

### 3. API Key Errors
**Problem**: "API key is required" error
**Solutions**:
- Verify `.env.local` exists (not `.env.local.example`)
- Check `GOOGLE_API_KEY` is set correctly
- Ensure file is in same directory as `agent.py`

### 4. Model Not Available
**Problem**: Model name not found
**Solutions**:
- Verify model name: `gemini-2.5-flash-native-audio-preview-09-2025`
- Check API key has access to Gemini Live API
- Try alternative: `gemini-2.0-flash-live-001`

## Best Practices

1. **Keep instructions concise**: Long system prompts can increase latency
2. **Use appropriate temperature**: 0.8 for conversation, lower for factual tasks
3. **Monitor costs**: Vision + audio increases token usage significantly
4. **Handle errors gracefully**: Network issues, API limits, etc.
5. **Test with playground first**: Before building custom frontend
6. **Version control `.env.local.example`**: Never commit actual credentials

## Frontend Integration

This agent works with any LiveKit-compatible frontend. Key requirements:

1. **Publish camera track**: `room.localParticipant.publishTrack(cameraTrack)`
2. **Publish microphone track**: `room.localParticipant.publishTrack(micTrack)`
3. **Subscribe to agent audio**: Automatic playback
4. **Display transcriptions**: Listen to `lk.transcription` text streams

Example with React:
```typescript
import { useLocalParticipant, useParticipants } from '@livekit/components-react';

function VoiceAgentUI() {
  const { localParticipant } = useLocalParticipant();
  
  // Enable camera and mic
  useEffect(() => {
    localParticipant.setCameraEnabled(true);
    localParticipant.setMicrophoneEnabled(true);
  }, [localParticipant]);
  
  // Rest of UI implementation
}
```

## Advanced Features to Add

### 1. Multi-Agent Workflows
```python
from livekit.agents import WorkflowAgent

# Create specialized agents and delegate tasks
```

### 2. Custom Video Processing
```python
async def process_frame(frame):
    # Apply filters, object detection, etc.
    return processed_frame
```

### 3. Memory and State
```python
class VisionAssistant(Agent):
    def __init__(self):
        super().__init__(...)
        self.conversation_memory = []
```

### 4. External Integrations
```python
@function_tool()
async def search_database(self, query: str) -> str:
    # Query your database based on visual + voice context
    pass
```

## Deployment

### LiveKit Cloud
```bash
lk agent create
```
- Automatic Docker build
- Global deployment
- Managed scaling

### Custom Infrastructure
- Build Docker image
- Deploy to Kubernetes/ECS/etc.
- Set environment variables
- Configure autoscaling

## Resources

- [LiveKit Agents Docs](https://docs.livekit.io/agents/)
- [Gemini Live API](https://ai.google.dev/gemini-api/docs/live)
- [Vision Guide](https://docs.livekit.io/agents/build/vision/)
- [Turn Detection](https://docs.livekit.io/agents/build/turns/)
- [Python SDK Reference](https://docs.livekit.io/reference/python/v1/livekit/agents/index.html)
- [LiveKit Slack](https://livekit.io/join-slack)

## When Making Changes

1. **Test locally first**: Use `python agent.py dev`
2. **Verify in playground**: Before custom frontend
3. **Check logs**: Look for errors or warnings
4. **Monitor performance**: Response time, token usage
5. **Update documentation**: Keep README.md current

## MCP Server Integration

If using an AI coding assistant with MCP (Model Context Protocol):

Install LiveKit Docs MCP server for latest documentation:
```bash
# See: https://docs.livekit.io/home/get-started/mcp-server/
```

This ensures the AI assistant has access to the most current LiveKit documentation while helping you build your agent.

---

**Remember**: This is a real-time multimodal AI system. Small changes can have significant impacts on performance and cost. Test thoroughly!