from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic

load_dotenv()

fast  = ChatAnthropic(model="claude-haiku-4-5-20251001",  temperature=0)  # dev + bulk work
smart = ChatAnthropic(model="claude-sonnet-5-5", temperature=0)  # quality step only