from src.agents.researcher_team import DebateCoordinator, BullishResearcher, BearishResearcher
from src.schemas.researcher_schemas import DebateTurn, DebateResult
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from src.schemas.researcher_schemas import BullishThesisOutput  # or shared schema if same for bearish
from langchain_core.language_models import BaseChatModel
import os 
load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
from langchain_groq import ChatGroq
import os

#print("✅ Using GROQ Key:", os.getenv("GROQ_API_KEY"))



# 1. Instantiate LLM
llm: BaseChatModel = ChatGroq(model_name="llama3-70b-8192", temperature=0.3)

# 2. Instantiate researchers
bullish_researcher = BullishResearcher(ticker="AAPL", llm=llm)
bearish_researcher = BearishResearcher(ticker="AAPL", llm=llm)

# 3. Prepare dummy analyst summaries (normally these come from analyst agents)
summary_bundle = {
    "technical": (
        "Apple's stock has broken out of a multi-week consolidation pattern on above-average volume. "
        "The RSI remains elevated at 67 but hasn't entered overbought territory. "
        "MACD shows a bullish crossover, and price action is consistently above the 50-day and 200-day moving averages. "
        "However, there's mild bearish divergence forming on the weekly RSI."
    ),

    "sentiment": (
        "Post-earnings sentiment is mixed: while analysts raised price targets due to AI service announcements, "
        "institutional investors are cautious due to weaker-than-expected iPhone revenue. "
        "Retail sentiment remains positive on social media, with Reddit and X discussing bullish AI prospects. "
        "However, insider selling by two senior executives last week has slightly dampened optimism."
    ),

    "news": (
        "Apple has confirmed plans to integrate on-device generative AI into iOS 18, partnering with OpenAI for edge deployments. "
        "Meanwhile, the EU has launched a formal investigation into Apple’s App Store policies under the Digital Markets Act. "
        "Foxconn has also warned of delays in the iPhone 16 supply chain due to floods in Southern China, which could affect Q4 shipments. "
        "Lastly, Apple acquired a German startup specializing in quantum dot display technology."
    ),

    "fundamental": (
        "Gross margins improved to 45.2% due to a higher mix of services revenue, now 26% of total income. "
        "Free cash flow remains robust at $95B, with $166B in cash reserves. "
        "However, iPhone revenue declined 4% YoY, especially in China, where competition from Huawei is intensifying. "
        "R&D spending increased 13%, indicating long-term bets on AI and hardware innovation. "
        "Buyback programs continue aggressively, shrinking float by 2.1% this quarter."
    )
}


# 4. Get structured outputs
bullish_output: BullishThesisOutput = bullish_researcher.generate_thesis(summary_bundle)
bearish_output: BullishThesisOutput = bearish_researcher.generate_thesis(summary_bundle)  # same schema assumed

# 5. Convert structured outputs to string for debate
def thesis_to_debate_text(thesis_obj: BullishThesisOutput) -> str:
    points = "\n".join(f"- {point}" for point in thesis_obj.supporting_points)
    return f"{thesis_obj.thesis}\n{points}\nConfidence: {thesis_obj.confidence}"

bullish_text = thesis_to_debate_text(bullish_output)
bearish_text = thesis_to_debate_text(bearish_output)

# 6. Debate
coordinator = DebateCoordinator(ticker="AAPL", llm=llm)
debate_result: DebateResult = coordinator.conduct_debate(
    bullish_thesis=bullish_text,
    bearish_thesis=bearish_text,
    rounds=2
)

# 7. Output
print("\n🗣️ Debate Transcript:")
for turn in debate_result.turns:
    print(f"{turn.speaker}: {turn.message}\n")

print("📋 Summary:", debate_result.summary)
print("🏆 Winner:", debate_result.winner)
print("🧠 Total Tokens Used:", debate_result.total_tokens)
