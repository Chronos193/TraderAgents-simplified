# test_debatecoordinator.py

from src.agents.researcher_team import DebateCoordinator, BullishResearcher, BearishResearcher
from src.schemas.researcher_schemas import DebateTurn, DebateResult, BullishThesisOutput

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

# ✅ 1. Load your Gemini API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in environment. Please set it!")

# ✅ 2. Create Gemini LLM instance
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=GOOGLE_API_KEY
)

# ✅ 3. Instantiate researchers (same structure)
bullish_researcher = BullishResearcher(ticker="AAPL", llm=llm)
bearish_researcher = BearishResearcher(ticker="AAPL", llm=llm)

# ✅ 4. Dummy analyst bundle
summary_bundle = {
    "technical": (
        "Apple's stock has broken out of a multi-week consolidation pattern on above-average volume. "
        "RSI is at 67, MACD bullish crossover, price above 50-day & 200-day MAs. "
        "Mild bearish divergence on weekly RSI."
    ),
    "sentiment": (
        "Post-earnings sentiment is mixed. Analysts raised targets due to AI, "
        "but institutional investors are cautious on weak iPhone revenue. "
        "Retail sentiment stays positive. Insider selling by execs last week dampened optimism."
    ),
    "news": (
        "Apple confirmed on-device generative AI in iOS 18, partnering with OpenAI. "
        "EU investigating App Store under DMA. Foxconn warned of iPhone 16 delays due to China floods. "
        "Apple acquired a quantum dot display startup."
    ),
    "fundamental": (
        "Gross margins improved to 45.2% with more services revenue. "
        "Free cash flow at $95B, cash reserves $166B. "
        "iPhone revenue down 4% YoY, especially in China. "
        "R&D up 13% for AI and hardware innovation. Buybacks shrink float by 2.1%."
    )
}

# ✅ 5. Generate structured theses
bullish_output: BullishThesisOutput = bullish_researcher.generate_thesis(summary_bundle)
bearish_output: BullishThesisOutput = bearish_researcher.generate_thesis(summary_bundle)

def thesis_to_text(thesis_obj: BullishThesisOutput) -> str:
    points = "\n".join(f"- {point}" for point in thesis_obj.supporting_points)
    return f"{thesis_obj.thesis}\n{points}\nConfidence: {thesis_obj.confidence}"

bullish_text = thesis_to_text(bullish_output)
bearish_text = thesis_to_text(bearish_output)

# ✅ 6. Run the debate
coordinator = DebateCoordinator(ticker="AAPL", llm=llm)
debate_result: DebateResult = coordinator.conduct_debate(
    bullish_thesis=bullish_text,
    bearish_thesis=bearish_text,
    rounds=2
)

# ✅ 7. Output results
print("\n🗣️ Debate Transcript:")
for turn in debate_result.turns:
    print(f"{turn.speaker}: {turn.message}\n")

print("📋 Summary:", debate_result.summary)
print("🏆 Winner:", debate_result.winner)
print("🧠 Total Tokens Used:", debate_result.total_tokens)
