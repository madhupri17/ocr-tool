from PIL import Image
import pytesseract

# LangChain imports
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

# Your OCR tool (example)
from langchain.tools import tool

@tool
def ocr_read_document(image_path: str) -> str:
    """Reads an image from the given path and returns extracted text using OCR."""
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return text
    except Exception as e:
        return f"Error reading image: {e}"

# 1. Define your tools list
tools = [ocr_read_document]
##

# 2. Define your LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# 3. Create your agent using the stable API
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# 4. Run the agent
response = agent.run("Extract text from the image at 'sample.png'")
print(response)


