import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# LLM configuration
llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0
)

# Pydantic schema
class ReasoningOutput(BaseModel):
    language: Literal["arabic", "english"]
    sentiment: Literal["Positive", "Negative", "Neutral"]
    explanation: str = Field(
        description="Explanation of why this sentiment was chosen."
    )

# Prompt template
reasoning_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert sentiment analysis and review interpretation system.

Your task:
1. Determine the OVERALL sentiment of the text.
2. Detect the language of the input text (Arabic or English).
3. Write the explanation in the SAME language as the input text.
4. Explain clearly WHY this sentiment was chosen.
5. Explain briefly WHY the other sentiment options were NOT chosen.

Critical interpretation rules:
- Do NOT rely on sentiment keywords alone.
- Interpret meaning, intent, and context.
- If the reviewer explicitly states dissatisfaction, stress, regret,
  or lack of enjoyment, treat this as NEGATIVE sentiment
  unless clearly outweighed by a strong positive conclusion.
- Statements like "I didn't enjoy the experience" or
  "متبسطتش" indicate negative sentiment
  even if some positive aspects are mentioned.

Weigh mixed opinions by:
- importance
- severity
- repetition
- emotional impact
- the final emotional takeaway of the experience

For long or complex reviews:
- Identify the main negative themes
- Identify the main positive themes
- Decide which side defines the overall experience

Sentiment labels (internal decision):
- Positive: positives clearly outweigh negatives and the reviewer expresses enjoyment
- Negative: negatives outweigh positives OR the reviewer expresses dissatisfaction or lack of enjoyment
- Neutral: balanced or factual with no clear emotional dominance

Explanation requirements:
- Be concise but clear (3–5 sentences)
- Explicitly mention the emotional outcome
- Clearly justify why this label fits better than the other two
- Use the SAME language as the input text
"""
    ),
    (
        "human",
        """
Example 1 (English):
Text:
"I loved the pasta, but the restaurant experience was disappointing. The service was slow and the staff were rude."

Sentiment: Negative
Explanation:
Although the pasta was praised, the negative aspects related to slow service and rude staff had a stronger impact on the overall experience. The reviewer clearly expresses disappointment, which outweighs the positive mention of the food. Therefore, the sentiment is negative rather than positive or neutral.

---

Example 2 (English):
Text:
"It’s giving pure elegance and high-end vibes—so chic!"

Sentiment: Positive
Explanation:
The sentence expresses strong admiration and approval using highly positive phrases like “pure elegance,” “high-end vibes,” and “so chic.” These words clearly convey excitement and appreciation without any negative elements. Therefore, the overall sentiment is strongly positive.
---

Example 3 (Arabic):
Text:
"تحفة بجد ونظافة رهيبة بعشق المكان ده والجودة الخطيرة والاحترام حرفيا"

Sentiment: Positive
Explanation:
النص بيعبر عن انبهار وإعجاب شديد بالمكان من خلال ألفاظ قوية زي “تحفة بجد” و*“نظافة رهيبة”* و*“بعشق المكان ده”*. كمان الإشادة بالجودة والاحترام بتعكس رضا كامل عن التجربة بدون أي ملاحظات سلبية. لذلك التصنيف الأنسب هنا هو إيجابي قوي.

Now analyze the following text:
{text}
"""
    )
])

# reasoning chain
reasoning_chain = (
    reasoning_prompt
    | llm.with_structured_output(ReasoningOutput)
)

# Streamlit app
st.set_page_config(
    page_title="Sentiment Chat",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Multilingual Sentiment Chat")
st.caption("Arabic + English review sentiment with reasoning")

# Session State (Chat Memory)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Enter a review in Arabic or English...")

if user_input:

    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # LLM Processing
    with st.chat_message("assistant"):
        with st.spinner("Analyzing sentiment..."):

            result = reasoning_chain.invoke(
                {"text": user_input}
            )

            # Format output
            if result.language == "arabic":
                sentiment_label = {
                    "Positive": "إيجابي",
                    "Negative": "سلبي",
                    "Neutral": "محايد"
                }[result.sentiment]
            else:
                sentiment_label = result.sentiment

            response_text = f"""
**Sentiment:** {sentiment_label}

**Explanation:**  
{result.explanation}
"""

            st.markdown(response_text)

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text}
    )
