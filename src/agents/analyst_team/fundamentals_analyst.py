from src.agents.analyst_team import Analyst
import yfinance as yf
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.callbacks.manager import get_openai_callback
from src.schemas.analyst_schemas import FundamentalsAnalysisOutput


class FundamentalsAnalyst(Analyst):
    def __init__(self, llm: BaseChatModel):
        super().__init__( llm=llm)
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

    def structured_analyze(self, ticker: str) -> FundamentalsAnalysisOutput:
        stock = yf.Ticker(ticker)
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow

        financials_summary = self._get_latest(financials, ["Total Revenue", "Gross Profit", "Net Income"])
        balance_summary = self._get_latest(balance_sheet, [
            "Total Assets", "Total Liabilities Net Minority Interest", "Stockholders Equity"
        ])
        cashflow_summary = self._get_latest(cashflow, [
            "Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow"
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

        with get_openai_callback() as cb:
            analysis_text = chain.invoke({
                "ticker": ticker,
                "financials_summary": financials_summary,
                "balance_summary": balance_summary,
                "cashflow_summary": cashflow_summary
            })
            self._last_token_count = cb.total_tokens

        return FundamentalsAnalysisOutput(
            ticker=ticker,
            financials_summary=financials_summary,
            balance_summary=balance_summary,
            cashflow_summary=cashflow_summary,
            analysis_text=analysis_text
        )

    def analyze(self, ticker: str):
        return self.structured_analyze(ticker).analysis_text

    def get_last_token_count(self):
        return self._last_token_count

    def summary(self, ticker: str):
        stock = yf.Ticker(ticker)
        info = stock.info or {}
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

    def as_runnable_node(self) -> RunnableLambda:
        return RunnableLambda(lambda state: self.__call__(state))

    def __call__(self, state: dict) -> dict:
        ticker = state.get("ticker")
        if not ticker:
            print("[ERROR] Ticker missing from state.")
            return {**state, "fundamentals_analysis": None}
    
        try:
            output = self.structured_analyze(ticker)
            return {**state, "fundamentals_analysis": output}
        except Exception as e:
            print(f"[ERROR] FundamentalsAnalyst failed for {ticker}: {e}")
            return {**state, "fundamentals_analysis": None}

