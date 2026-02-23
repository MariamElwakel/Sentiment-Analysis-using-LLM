import os
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json


# ENV
load_dotenv()


# Load reviews from text file
def load_reviews(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    reviews = []
    current = []

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue

        if line[0].isdigit() and line[1] == ".":
            if current:
                reviews.append(" ".join(current).strip())
                current = []
        else:
            current.append(line)

    if current:
        reviews.append(" ".join(current).strip())

    return reviews


# Define structured output schemas
class ReliabilityOutput(BaseModel):
    language: Literal["arabic", "english"]
    sentiment: Literal["Positive", "Negative", "Neutral"]

class ReasoningOutput(BaseModel):
    language: Literal["arabic", "english"]
    sentiment: Literal["Positive", "Negative", "Neutral"]
    explanation: str = Field(
        description="A brief explanation justifying why the sentiment was classified as Positive, Negative, or Neutral."
    )


# LLM configuration
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Prompt templates for reliability
reliability_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a professional sentiment classification system.

Your task:
- Determine the OVERALL sentiment of the text.
- Detect the language of the input text (Arabic or English).
- Return the sentiment label in the SAME language as the input.
- Return ONLY the final sentiment label.
- Do NOT provide explanations, reasoning, or extra text.

Critical interpretation rules:
- Focus on the overall experience, not isolated words.
- Do NOT rely on keywords alone (e.g., "love", "hate").
- If the reviewer explicitly states lack of enjoyment, discomfort, stress, or regret
  (e.g., "مش مبسوط", "I didn't enjoy it"),
  this MUST be treated as Negative sentiment,
  even if some positive aspects are mentioned.
- Neutral sentiment should ONLY be used when there is no clear emotional outcome.

Decision rules:
- Positive: overall enjoyment or satisfaction is clearly expressed.
- Negative: dissatisfaction, discomfort, stress, or lack of enjoyment dominates.
- Neutral: balanced or factual with no clear emotional dominance.

Be consistent and deterministic.

Sentiment labels:
- English: Positive, Negative, Neutral
- Arabic: إيجابي، سلبي، محايد
"""
    ),
    (
        "human",
        """
Example 1 (English):
Text: "I loved the pasta, but the service was terrible and the staff were rude."
Sentiment: Negative

Example 2 (English):
Text: "The food was amazing, the staff were friendly, and the atmosphere was great."
Sentiment: Positive

Example 3 (English):
Text: "The place was okay. Nothing special, nothing awful."
Sentiment: Neutral

Example 5 (Arabic):
Text: "كان عادي، لا حلو قوي ولا وحش."
Sentiment: Neutral

Example 6 (Arabic):
Text: "الاكلير كان سيئ، ماعجبنيش خالص."
Sentiment: Negative

Now analyze the following text and return the sentiment in the SAME language:
{text}
"""
    )
])


# reliability chain
reliability_chain = (
    reliability_prompt
    | llm.with_structured_output(ReliabilityOutput)
)


# Prompt templates for reasoning
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


# Save outputs to JSON
def save_json(filename, data):
    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/{filename}", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# Processes a review text file through two analysis pipelines: reliability and reasoning
def process_input_file(input_file: str):
    reviews = load_reviews(input_file)

    reliability_outputs = []
    reasoning_outputs = []

    for idx, review in enumerate(reviews, start=1):
        # Reliability
        reliability_result = reliability_chain.invoke({"text": review})
        reliability_outputs.append({
            "id": idx,
            "text": review,
            **reliability_result.model_dump()
        })

        # Reasoning
        reasoning_result = reasoning_chain.invoke({"text": review})
        reasoning_outputs.append({
            "id": idx,
            "text": review,
            **reasoning_result.model_dump()
        })

    return reliability_outputs, reasoning_outputs

# Test English input
eng_reliability, eng_reasoning = process_input_file("English_input.txt")

# Test Arabic input
ar_reliability, ar_reasoning = process_input_file("Arabic_input.txt")

# Save results to JSON files
save_json("reliability_english.json", eng_reliability)
save_json("reasoning_english.json", eng_reasoning)

save_json("reliability_arabic.json", ar_reliability)
save_json("reasoning_arabic.json", ar_reasoning)