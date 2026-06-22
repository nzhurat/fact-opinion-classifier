import random


cur_exmp = [
    {
        "text": (
            "The 2025 Zadar Open, known as the Falkensteiner Punta Skala Zadar "
            "Open, was a professional tennis tournament played on clay courts. "
            "It was the fifth edition of the tournament which was part of the "
            "2025 ATP Challenger Tour. It took place in Zadar, Croatia between "
            "17 and 23 March 2025."
        ),
        "label": "Fact",
        "reasoning": (
            "Contains verifiable details: dates, location, event name, and tour "
            "classification. No subjective language."
        ),
    },
    {
        "text": (
            "Martin Lewis has warned that millions of households face a "
            "\"ticking timebomb\" on their energy bills as global fuel prices surge "
            "amid the conflict involving the United States, Israel and Iran. "
            "The MoneySavingExpert founder said families could see costs rise by "
            "as much as 30 per cent if elevated oil and gas prices persist for several months."
        ),
        "label": "Opinion",
        "reasoning": (
            "This is a prediction/warning about future outcomes. Even with numbers, "
            "it is not a confirmed event but a forecast."
        ),
    },
    {
        "text": (
            "NASA reported that the DART mission successfully altered the orbit "
            "of a small asteroid in a binary system. The probe was launched in 2021."
        ),
        "label": "Fact",
        "reasoning": (
            "Describes a confirmed, completed mission with measurable results."
        ),
    },
    {
        "text": (
            "These pancakes are everything you want in a breakfast. They are so fluffy "
            "and melt in your mouth."
        ),
        "label": "Opinion",
        "reasoning": (
            "Subjective sensory judgments ('everything you want', 'fluffy', 'melt in your mouth') "
            "cannot be objectively verified."
        ),
    },
    {
        "text": (
            "Children spending more than three hours daily on social media are more "
            "likely to develop depression, according to research."
        ),
        "label": "Fact",
        "reasoning": (
            "Reports study findings with attribution ('according to research')."
        ),
    },
    {
        "text": (
            "AI is the future. I strongly believe we need ethical guidelines for its use."
        ),
        "label": "Opinion",
        "reasoning": (
            "Explicit personal belief and normative judgment."
        ),
    },
    {
        "text": (
            "Nearly 55k job cuts were attributed to AI according to company reports."
        ),
        "label": "Fact",
        "reasoning": (
            "Quantified claim supported by reported data and attribution."
        ),
    },
    {
        "text": (
            "I'm flabbergasted by the relentless pessimism I'm seeing in much of the commentariat. "
            "We are less than two weeks into a war that will almost surely be over by the end of the month, "
            "and already there are predictions that it's 'another Iraq.' American casualties, "
            "heartbreaking as they are, have been minor for a conflict of this scale."
        ),
        "label": "Opinion",
        "reasoning": (
            "Emotional expression + prediction about uncertain future outcome."
        ),
    },
]


seed = 40

def load_curated_examples(shuffle=True):
    examples = list(cur_exmp)
    if shuffle:
        random.Random(seed).shuffle(examples)
    return examples


marker_opinion = [
    "I think", "I believe", "I feel", "in my opinion", "seems",
    "should", "arguably", "surprisingly", "best", "worst",
    "love", "hate", "flabbergasted"
]

marker_fact = [
    "according to", "data shows", "study found", "reported",
    "percent", "confirmed", "measured", "official"
]

marker_lingua = (
    "These are soft hints only, not rules. Many fact texts can include "
    "markers, and opinions may not include them."
)


chain_of_thought = """
Decide Fact vs Opinion using:

1. Verifiability - can it be objectively checked?
2. Subjectivity - does it include judgment or feelings?
3. Predictions - future claims are usually Opinion.
4. Attribution - reported studies/news without commentary → Fact.
5. Mixed content - choose dominant intent.
"""