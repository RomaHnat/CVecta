from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic

load_dotenv()

fast  = ChatAnthropic(model="<haiku-model-id>",  temperature=0)  # dev + bulk work
smart = ChatAnthropic(model="<sonnet-model-id>", temperature=0)  # quality step only