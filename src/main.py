import argparse
import asyncio
import logging
import platform
import signal

from src.config.loader import load_config
from src.core.orchestrator import Orchestrator
from src.core.session import Session
from src.services.agent import AgentService
from src.services.audio_capture import LiveAudioCaptureService, MockAudioCaptureService
from src.services.audio_playback import MockAudioPlaybackService
from src.services.stt import SpeechToTextService
from src.services.tts import TextToSpeechService
from src.services.wake_word import OpenWakeWordService
from src.tools.builtin.device_control import DeviceControlTool
from src.tools.builtin.web_fetch import WebFetchTool
from src.tools.builtin.web_search import WebSearchTool
from src.tools.registry import ToolRegistry
from src.util.logging import setup_logging

logger = logging.getLogger(__name__)


async def main(args) -> None:
    config = load_config()
    setup_logging(config.logging)

    logger.info("Starting Pi Voice Assistant")

    # Tool registry
    registry = ToolRegistry()
    registry.register(DeviceControlTool())
    registry.register(WebFetchTool())
    registry.register(WebSearchTool())

    # Exit early for print mode
    if args.print:
        logger.info(f"Starting agent in headless mode (--print) with prompt: {args.print}")
        session = Session(config.session)
        session.start(config.agent.system_prompt)
        agent = AgentService(config.agent, registry)
        await agent.start()
        async for chunk in agent.run(args.print, session):
            print(chunk, end="", flush=True)
        print()
        await agent.stop()
        return

    # Services
    wake_word = OpenWakeWordService(config.wake_word)
    if platform.system() == "Linux":
        audio_capture = LiveAudioCaptureService(config.audio)
    else:
        audio_capture = MockAudioCaptureService(config.audio)
    stt = SpeechToTextService(config.stt)
    agent = AgentService(config.agent, registry)
    tts = TextToSpeechService(config.tts)
    audio_playback = MockAudioPlaybackService()

    # Session
    session = Session(config.session)

    # Orchestrator
    orchestrator = Orchestrator(
        args=args,
        config=config,
        wake_word=wake_word,
        audio_capture=audio_capture,
        stt=stt,
        agent=agent,
        tts=tts,
        audio_playback=audio_playback,
        session=session,
    )

    # Graceful shutdown on SIGINT/SIGTERM
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(orchestrator.stop()))

    try:
        await orchestrator.start()
    except Exception:
        logger.exception("Fatal error")
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Voice Assistant")
    parser.add_argument("--no-wake-wait", action="store_true", help="Skip the waiting for wake word state")
    parser.add_argument("--print", action="store", help="Get a single headless response from the agent")
    args = parser.parse_args()
    asyncio.run(main(args))
