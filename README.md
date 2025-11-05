# Voice Agent with Vision

A general-purpose voice assistant with vision capabilities built using LiveKit Agents and Google's Gemini Live API.

## Features

- 🎤 **Natural Voice Conversations**: Powered by Gemini's speech-to-speech capabilities
- 👁️ **Live Video Input**: The agent can see what you show through your camera
- 🔊 **Background Noise Cancellation**: Clean audio even in noisy environments
- 💬 **Real-time Transcriptions**: See what's being said on both sides
- 🌐 **WebRTC Reliability**: Fast, low-latency communication

## Prerequisites

- Python 3.9 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager (recommended) or pip
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)
- A [LiveKit Cloud account](https://cloud.livekit.io/) (free tier available)

## Setup

### 1. Install Dependencies

Using uv (recommended):

```bash
uv pip install -r requirements.txt
```

Or using pip:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the example environment file:

```bash
cp .env.local.example .env.local
```

Edit `.env.local` and add your credentials:

```bash
# Google Gemini API Key
GOOGLE_API_KEY=your_gemini_api_key_here

# LiveKit Credentials
LIVEKIT_API_KEY=your_livekit_api_key_here
LIVEKIT_API_SECRET=your_livekit_api_secret_here
LIVEKIT_URL=wss://your-project.livekit.cloud
```

**Getting Your Credentials:**

- **Gemini API Key**: Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
- **LiveKit Credentials**: 
  1. Sign up at [LiveKit Cloud](https://cloud.livekit.io/)
  2. Create a new project
  3. Use the LiveKit CLI: `lk app env -w` to automatically populate `.env.local`

### 3. Download Model Files

The agent uses local models for voice activity detection and turn detection. Download them with:

```bash
python agent.py download-files
```

## Running the Agent

### Development Mode (with Playground)

Run the agent in development mode to test it with the LiveKit Agents Playground:

```bash
python agent.py dev
```

This will:
1. Start the agent worker
2. Connect to your LiveKit Cloud project
3. Provide a URL to the playground where you can test the agent
4. Enable video and audio streaming

### Console Mode (Terminal Testing)

For quick testing in your terminal without a web interface:

```bash
python agent.py console
```

Note: Console mode only supports audio, not video.

### Production Mode

For production deployment:

```bash
python agent.py start
```

## How It Works

### Architecture

```
User (Camera + Mic) <-> LiveKit WebRTC <-> Agent Worker <-> Gemini Live API
                              ↓
                         Transcriptions
```

1. **User connects**: Camera and microphone streams are sent to LiveKit
2. **Agent subscribes**: The agent receives audio and video frames
3. **Video sampling**: Gemini processes video at ~1 frame per second while the user speaks
4. **Speech processing**: Gemini understands both audio and visual context
5. **Response**: Natural voice responses are sent back to the user
6. **Transcriptions**: Both user and agent speech are transcribed in real-time

### Key Components

- **[`agent.py`](agent.py)**: Main agent implementation
  - `VisionAssistant` class: Defines agent behavior and personality
  - `entrypoint()`: Initializes and starts the agent session

- **Vision System**: 
  - Automatically samples video frames from the user's camera
  - Frames are resized to 1024x1024 and encoded to JPEG
  - Default sampling: 1 fps while user speaks, 1 frame per 3s otherwise

- **Turn Detection**: 
  - Uses Gemini's built-in VAD-based turn detection
  - Automatically detects when user finishes speaking

## Customization

### Change the Voice

Available voices: `Puck`, `Charon`, `Kore`, `Fenrir`, `Aoede`

Edit [`agent.py`](agent.py:38):

```python
llm=google.realtime.RealtimeModel(
    model="gemini-2.5-flash-native-audio-preview-09-2025",
    voice="Aoede",  # Change this
    temperature=0.8,
)
```

### Adjust Instructions

Modify the agent's instructions in [`agent.py`](agent.py:17-32) to change its behavior:

```python
instructions="""You are a helpful AI assistant specialized in [your domain].
Your custom instructions here..."""
```

### Custom Video Sampling

For different frame rates or resolutions, you can customize the video sampler. See the [LiveKit Vision documentation](https://docs.livekit.io/agents/build/vision/) for details.

### Add Tool Calling

You can extend the agent with function calling capabilities:

```python
from livekit.agents import function_tool

class VisionAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="...",
            llm=google.realtime.RealtimeModel(...),
        )
    
    @function_tool()
    async def get_weather(self, location: str) -> str:
        """Get the current weather for a location"""
        # Your implementation
        return f"Weather in {location}: Sunny, 72°F"
```

## Deployment

### Deploy to LiveKit Cloud

The easiest way to deploy:

```bash
lk agent create
```

This will:
1. Create a `Dockerfile` in your project
2. Build and deploy your agent to LiveKit Cloud
3. Make it available globally

### Custom Deployment

For custom infrastructure, you can:
- Build a Docker container
- Deploy to Kubernetes
- Run on your own servers

See the [LiveKit deployment guide](https://docs.livekit.io/agents/ops/deployment/) for details.

## Frontend Integration

This agent is ready to work with any LiveKit-compatible frontend. You mentioned you'll create a web frontend later.

### Quick Testing with Playground

Use the LiveKit Agents Playground (available when running in `dev` mode) to test your agent without building a frontend.

### Building a Custom Frontend

For production, you'll want to build a custom web frontend. Recommended stack:

- React with [`@livekit/components-react`](https://docs.livekit.io/reference/components/react/)
- TypeScript
- LiveKit Client SDK

Key features to implement:
1. Camera and microphone publishing
2. Audio playback from agent
3. Transcription display
4. Video preview
5. Connection status

See the [LiveKit frontend guide](https://docs.livekit.io/agents/start/frontend/) for examples.

## Troubleshooting

### "API key is required" error

Make sure you've:
1. Created a `.env.local` file (not `.env.local.example`)
2. Added your actual `GOOGLE_API_KEY`
3. The file is in the same directory as `agent.py`

### No video input

Ensure:
1. Your frontend is publishing a video track
2. The agent has `video_enabled=True` in `RoomInputOptions`
3. Your camera permissions are granted in the browser

### Slow responses

Try:
1. Reducing video frame rate or resolution
2. Using a different Gemini model
3. Checking your network connection
4. Adjusting `temperature` parameter

### Connection issues

1. Verify your `LIVEKIT_URL` is correct
2. Check your API key and secret
3. Ensure your LiveKit Cloud project is active
4. Check firewall settings

## Resources

- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [Gemini Live API Documentation](https://ai.google.dev/gemini-api/docs/live)
- [LiveKit Vision Guide](https://docs.livekit.io/agents/build/vision/)
- [LiveKit Community Slack](https://livekit.io/join-slack)

## License

This project is provided as-is for educational and development purposes.

## Next Steps

1. ✅ Set up the agent (you're here!)
2. 🔧 Customize the agent's instructions and behavior
3. 🌐 Build a custom web frontend
4. 🚀 Deploy to production
5. 📊 Add analytics and monitoring

For questions or issues, refer to the [LiveKit documentation](https://docs.livekit.io/) or join the [community Slack](https://livekit.io/join-slack).