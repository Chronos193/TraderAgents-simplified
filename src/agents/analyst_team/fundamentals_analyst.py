from src.agents.analyst_team import Analyst
import yfinance as yf
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.callbacks.manager import get_openai_callback
from src.schemas.analyst_schemas import FundamentalsAnalysisOutput


class FundamentalsAnalyst(Analyst):
    def __init__(self, ticker, llm: BaseChatModel):
        super().__init__(ticker, llm)
        self.stock = yf.Ticker(ticker)
        self._last_token_count = 0

    def _get_latest(self, df, wanted_keys):
        summary = {}
        if df.empty:
            print("[WARN] DataFrame is empty")
            return summary
        df = df.transpose()
        for key in wanted_keys:
            matched_col = next((col for col in df.columns if col.lower() == key.lower()), None)
            if matched_col:
                try:
                    latest_value = df[matched_col].dropna().iloc[-1]
                    summary[key] = latest_value
                except Exception as e:
                    print(f"[WARN] Failed to extract {key}: {e}")
            else:
                print(f"[WARN] Key not found: {key}")
        return summary

    def structured_analyze(self) -> FundamentalsAnalysisOutput:

        financials = self.stock.financials
        balance_sheet = self.stock.balance_sheet
        cashflow = self.stock.cashflow

        #print("[DEBUG] balance_sheet columns:", list(balance_sheet.index))
        #print("[DEBUG] cashflow columns:", list(cashflow.index))

        financials_summary = self._get_latest(financials, ["Total Revenue", "Gross Profit", "Net Income"])
        balance_summary = self._get_latest(balance_sheet, [
            "Total Assets", 
            "Total Liabilities Net Minority Interest", 
            "Stockholders Equity"
        ])
        cashflow_summary = self._get_latest(cashflow, [
            "Operating Cash Flow",
            "Investing Cash Flow",
            "Financing Cash Flow"
        ])

        if not any([financials_summary, balance_summary, cashflow_summary]):
            raise ValueError("🛑 Empty financial data. Cannot analyze fundamentals.")

        system_msg = SystemMessagePromptTemplate.from_template(
            "You are a financial analyst. Summarize the company’s financial health, risks, and growth in 5 short bullet points each."

        )
        human_msg = HumanMessagePromptTemplate.from_template(
            "Give a brief summary for {ticker}. Use no more than 5 bullet points each for:\n"
            "- Financial Health\n- Risks\n- Opportunities.\n"
            "Data:\n"
            "Financials: {financials_summary}\n"
            "Balance Sheet: {balance_summary}\n"
            "Cashflow: {cashflow_summary}"
        )
        prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])
        chain = prompt | self.llm | StrOutputParser()

        inputs = {
            "ticker": self.ticker,
            "financials_summary": financials_summary,
            "balance_summary": balance_summary,
            "cashflow_summary": cashflow_summary
        }
        if financials.empty:
            print("[WARN] financials DataFrame is empty")
        if balance_sheet.empty:
            print("[WARN] balance_sheet DataFrame is empty")
        if cashflow.empty:
            print("[WARN] cashflow DataFrame is empty")

        with get_openai_callback() as cb:
            analysis_text = chain.invoke(inputs)
            self._last_token_count = cb.total_tokens

        return FundamentalsAnalysisOutput(
            ticker=self.ticker,
            financials_summary=financials_summary,
            balance_summary=balance_summary,
            cashflow_summary=cashflow_summary,
            analysis_text=analysis_text
        )

    def analyze(self):
        return self.structured_analyze().analysis_text

    def summary(self):
        info = self.stock.info or {}
        summary_fields = {k: info.get(k) for k in [
            "longName", "sector", "industry", "marketCap",
            "trailingPE", "dividendYield", "returnOnEquity",
            "debtToEquity", "address1", "city", "country"
        ] if info.get(k) is not None}

        system_msg = SystemMessagePromptTemplate.from_template(
            "You are a financial analyst. Summarize the company's key business details concisely."
        )
        human_msg = HumanMessagePromptTemplate.from_template(
            "Company Info: {summary_fields}"
        )
        prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])
        chain = prompt | self.llm | StrOutputParser()

        with get_openai_callback() as cb:
            result = chain.invoke({"summary_fields": summary_fields})
            self._last_token_count = cb.total_tokens

        return result

    def get_last_token_count(self):
        return self._last_token_count

    def as_runnable_node(self) -> RunnableLambda:
        return RunnableLambda(lambda _: self.structured_analyze())
