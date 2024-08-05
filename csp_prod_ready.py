import streamlit as st
import json
import os
import logging
from functools import lru_cache
import re
from collections import Counter
from typing import List, Dict
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import pandas as pd
import altair as alt
from textblob import TextBlob
import networkx as nx
import matplotlib.pyplot as plt
import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.stem import WordNetLemmatizer
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from dotenv import load_dotenv

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="jsonschema")

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables
load_dotenv()

# Constants
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
JSON_FILE_PATH = "chat_transcript.json"
MODEL_NAME = "gemini-pro"
NUM_TOPICS_DEFAULT = 3

# Download NLTK data
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Streamlit configurations
st.set_page_config(layout="wide", page_title="Chat Analyzer Pro", page_icon="🤖")

# Custom CSS
st.markdown(
    """
<style>
    .main > div { padding-top: 2rem; }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
        border-radius: 12px;
        border: none;
    }
    .stButton>button:hover { background-color: #45a049; }
    .chat-container {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        height: 400px;
        overflow-y: scroll;
        background-color: #f9f9f9;
    }
    .user-message {
        background-color: #DCF8C6;
        padding: 10px 15px;
        border-radius: 18px;
        margin: 5px 0;
        max-width: 80%;
        float: right;
        clear: both;
    }
    .bot-message {
        background-color: #FFFFFF;
        padding: 10px 15px;
        border-radius: 18px;
        margin: 5px 0;
        max-width: 80%;
        float: left;
        clear: both;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .stProgress > div > div { background-color: #4CAF50; }
</style>
""",
    unsafe_allow_html=True,
)


# Utility functions
@st.cache_data
def load_json_data(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error(f"JSON file not found: {file_path}")
        st.error(f"Error: JSON file not found. Please check the file path: {file_path}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON file: {file_path}")
        st.error(
            f"Error: Invalid JSON file. Please check the file content: {file_path}"
        )
        return None


@lru_cache(maxsize=1000)
def preprocess_text(text):
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 3:
            result.append(WordNetLemmatizer().lemmatize(token, pos="v"))
    return result


@lru_cache(maxsize=1000)
def extract_keywords(text):
    words = re.findall(r"\b\w+\b", text.lower())
    stop_words = set(
        [
            "the",
            "a",
            "an",
            "in",
            "on",
            "at",
            "for",
            "to",
            "of",
            "and",
            "is",
            "are",
            "was",
            "were",
            "hi",
            "hello",
        ]
    )
    return [word for word in words if word not in stop_words and len(word) > 2]


@lru_cache(maxsize=1000)
def calculate_sentiment(text):
    return TextBlob(text).sentiment.polarity


@lru_cache(maxsize=1000)
def extract_entities(text):
    blob = TextBlob(text)
    return [item for item in blob.noun_phrases]


def calculate_response_times(conversation):
    return [
        1
        for i in range(1, len(conversation))
        if conversation[i]["speaker"] != conversation[i - 1]["speaker"]
    ]


def mask_sensitive_info(text: str) -> str:
    """
    Mask sensitive information in the given text.
    """
    # Mask email addresses
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", text
    )

    # Mask phone numbers (various formats including XXX-XXX-XXXX)
    text = re.sub(
        r"\b(\+\d{1,2}\s?)?1?\-?\.?\s?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
        "[PHONE]",
        text,
    )

    # Mask social security numbers
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", text)

    # Mask credit card numbers (simplified, doesn't catch all formats)
    text = re.sub(r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "[CREDIT_CARD]", text)

    # Mask common disease names (add more as needed)
    diseases = ["cancer", "diabetes", "hiv", "aids", "hepatitis", "tuberculosis"]
    for disease in diseases:
        text = re.sub(
            r"\b" + re.escape(disease) + r"\b", "[DISEASE]", text, flags=re.IGNORECASE
        )

    return text


def mask_conversation(conversation: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Mask sensitive information in the entire conversation.
    """
    masked_conversation = []
    for message in conversation:
        masked_message = message.copy()
        masked_message["message"] = mask_sensitive_info(message["message"])
        masked_conversation.append(masked_message)
    return masked_conversation


def create_conversation_flow(conversation):
    G = nx.DiGraph()
    for i, msg in enumerate(conversation):
        G.add_node(i, speaker=msg["speaker"], message=msg["message"])
        if i > 0:
            G.add_edge(i - 1, i)
    return G


# Analysis functions
def perform_topic_modeling(conversation, num_topics=NUM_TOPICS_DEFAULT):
    processed_docs = [preprocess_text(msg["message"]) for msg in conversation]
    dictionary = corpora.Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    lda_model = gensim.models.LdaMulticore(
        corpus=corpus, id2word=dictionary, num_topics=num_topics
    )
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    return lda_model, vis


def analyze_conversation(conversation, model):
    formatted_conversation = "\n".join(
        [f"{msg['speaker']}: {msg['message']}" for msg in conversation]
    )
    prompt = f"""
    Analyze the following conversation between a user and a bot. 
    Focus on identifying any instances where:
    1. The bot fails to understand the user's question or intent.
    2. The bot provides inconsistent or contradictory information.
    3. The bot hallucinates or makes up information not present in the conversation.
    4. The bot repeats itself unnecessarily.
    5. The bot fails to solve the user's problem.

    Conversation:
    {formatted_conversation}

    Provide a detailed analysis covering the points above, and give an overall assessment of the bot's performance.
    Format your response as JSON with the following structure:
    {{
        "misunderstandings": [list of instances],
        "inconsistencies": [list of instances],
        "hallucinations": [list of instances],
        "repetitions": [list of instances],
        "unresolved_issues": [list of instances],
        "overall_assessment": "A brief overall assessment",
        "performance_score": A number between 0 and 10, where 10 is perfect performance
    }}
    """

    try:
        response = model.generate_content(prompt)
        parsed_json = extract_json_from_text(response.text)
        if parsed_json:
            return parsed_json
        else:
            logging.warning("Failed to parse JSON from Gemini response")
            return {"error": "Failed to parse JSON", "raw_response": response.text}
    except Exception as e:
        logging.error(f"Error in analyze_conversation: {str(e)}")
        return {"error": f"An error occurred: {str(e)}"}


def extract_json_from_text(text):
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        json_str = text[start : end + 1]
        json_str = re.sub(r",\s*}$", "}", json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logging.warning("Failed to parse JSON from text")
            return None
    return None


# Display functions
def display_chat_transcript(conversation: List[Dict[str, str]]):
    masked_conversation = mask_conversation(conversation)
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in masked_conversation:
            if message["speaker"] == "Bot":
                st.markdown(
                    f'<div class="bot-message">🤖 Bot: {message["message"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="user-message">👤 User: {message["message"]}</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)


def display_metrics(gemini_analysis):
    col1, col2, col3 = st.columns(3)
    with col1:
        score = gemini_analysis.get("performance_score", "N/A")
        if isinstance(score, (int, float)):
            st.metric("Overall Performance", f"{score}/10")
            st.progress(score / 10)
    with col2:
        misunderstandings = len(gemini_analysis.get("misunderstandings", []))
        st.metric("Misunderstandings", misunderstandings)
    with col3:
        unresolved = len(gemini_analysis.get("unresolved_issues", []))
        st.metric("Unresolved Issues", unresolved)


def display_insights(gemini_analysis, selected_user, analysis_options):
    st.subheader("📝 Overall Assessment")
    st.info(gemini_analysis.get("overall_assessment", "No overall assessment provided"))

    if "Sentiment Analysis" in analysis_options:
        st.subheader("😊 Sentiment Analysis")
        sentiments = [
            calculate_sentiment(msg["message"]) for msg in selected_user["conversation"]
        ]
        sentiment_df = pd.DataFrame(
            {"message_id": range(len(sentiments)), "sentiment": sentiments}
        )
        chart = (
            alt.Chart(sentiment_df)
            .mark_line()
            .encode(x="message_id", y="sentiment", tooltip=["message_id", "sentiment"])
            .properties(width=600, height=200)
        )
        st.altair_chart(chart, use_container_width=True)


def display_details(gemini_analysis):
    col1, col2 = st.columns(2)
    with col1:
        for key in ["misunderstandings", "inconsistencies", "hallucinations"]:
            with st.expander(f"{key.capitalize()} 🔍"):
                for item in gemini_analysis.get(key, []):
                    st.write(f"• {item}")
    with col2:
        for key in ["repetitions", "unresolved_issues"]:
            with st.expander(f"{key.capitalize()} 🔍"):
                for item in gemini_analysis.get(key, []):
                    st.write(f"• {item}")


def display_topic_modeling(selected_user, analysis_options):
    if "Topic Modeling" in analysis_options:
        num_topics = st.slider(
            "Number of Topics", min_value=2, max_value=10, value=5, key="topic_slider"
        )
        lda_model, vis = perform_topic_modeling(
            selected_user["conversation"], num_topics=num_topics
        )

        st.subheader("Topic-Word Distribution")
        topics = lda_model.print_topics(num_words=10)
        for idx, topic in topics:
            st.write(f"Topic {idx + 1}:")
            words = [
                (word, float(prob))
                for prob, word in re.findall(r'(0\.\d+)\*"(.+?)"', topic)
            ]
            words.sort(key=lambda x: x[1], reverse=True)

            with st.expander(f"Topic {idx + 1} Words"):
                for word, prob in words:
                    st.write(f"{word}: {prob:.4f}")

        st.subheader("Interactive Topic Model Visualization")
        html_string = pyLDAvis.prepared_data_to_html(vis)
        st.components.v1.html(html_string, width=1300, height=800)

        display_entity_topic_association(selected_user, lda_model)
    else:
        st.info("Please select 'Topic Modeling' in the sidebar to see the analysis.")


def display_entity_topic_association(selected_user, lda_model):
    st.subheader("Entity-Topic Association")
    entities = [
        item
        for msg in selected_user["conversation"]
        for item in extract_entities(msg["message"])
    ]
    entity_counts = Counter(entities)
    top_entities = dict(entity_counts.most_common(10))

    entity_topics = {}
    for entity in top_entities:
        entity_doc = preprocess_text(entity)
        entity_bow = lda_model.id2word.doc2bow(entity_doc)
        entity_topics[entity] = lda_model.get_document_topics(entity_bow)

    selected_entity = st.selectbox(
        "Select an entity to see topic association:",
        list(top_entities.keys()),
        key="entity_select",
    )
    if selected_entity:
        st.write(f"Topic distribution for '{selected_entity}':")
        for topic_id, prob in entity_topics[selected_entity]:
            st.write(f"Topic {topic_id + 1}: {prob:.4f}")


def display_additional_analyses(selected_user, analysis_options):
    if "Entity Extraction" in analysis_options:
        st.subheader("🏷️ Top Entities Mentioned")
        entities = [
            item
            for msg in selected_user["conversation"]
            for item in extract_entities(msg["message"])
        ]
        entity_counts = Counter(entities)
        entity_df = pd.DataFrame(
            entity_counts.most_common(10), columns=["Entity", "Count"]
        )
        st.bar_chart(entity_df.set_index("Entity"))

    if "Response Time Analysis" in analysis_options:
        st.subheader("⏱️ Response Time Analysis")
        response_times = calculate_response_times(selected_user["conversation"])
        st.line_chart(response_times)
        avg_response_time = (
            sum(response_times) / len(response_times) if response_times else 0
        )
        st.metric("Average Response Time", f"{avg_response_time:.2f} units")

    if "Conversation Flow" in analysis_options:
        st.subheader("🔀 Conversation Flow")
        G = create_conversation_flow(selected_user["conversation"])
        fig, ax = plt.subplots(figsize=(10, 5))
        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="lightblue",
            node_size=500,
            font_size=8,
            arrows=True,
        )
        st.pyplot(fig)


def main():
    # Load data
    data = load_json_data(JSON_FILE_PATH)
    if data is None:
        st.error(
            "Error: Unable to load chat data. Please check the JSON file and try again."
        )
        return

    # Configure Gemini API
    if not GOOGLE_API_KEY:
        st.error(
            "Error: GOOGLE_API_KEY not found. Please set the environment variable."
        )
        return

    genai.configure(api_key=GOOGLE_API_KEY)

    # Set up the model
    generation_config = {
        "temperature": 0.1,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }

    safety_settings = [
        {
            "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
            "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        },
        {
            "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        },
        {
            "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        },
        {
            "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        },
    ]

    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
    except Exception as e:
        logging.error(f"Error initializing Gemini model: {str(e)}")
        st.error(
            "Error: Unable to initialize the Gemini model. Please check your API key and try again."
        )
        return

    # App title and description
    st.title("🤖 Chat Analyzer Pro")
    st.markdown("Dive deep into your chat conversations with advanced NLP analysis.")

    # Sidebar for user selection and controls
    with st.sidebar:
        st.header("🛠️ Analysis Controls")
        user_selection = st.selectbox(
            "Select a user:",
            options=[f"{user['id']} - {user['name']}" for user in data["users"]],
        )
        analysis_options = st.multiselect(
            "Select Analysis Options:",
            [
                "Sentiment Analysis",
                "Entity Extraction",
                "Conversation Flow",
                "Topic Modeling",
            ],
            default=["Sentiment Analysis", "Entity Extraction"],
        )

    # Get selected user's data
    selected_user = next(
        user
        for user in data["users"]
        if f"{user['id']} - {user['name']}" == user_selection
    )

    # Display chat transcript
    st.header(f"💬 Chat Transcript: {selected_user['name']}")
    display_chat_transcript(selected_user["conversation"])

    # Analyze conversation button
    if st.button("🔍 Analyze Conversation", key="analyze_button"):
        with st.spinner("🧠 Analyzing conversation..."):
            try:
                gemini_analysis = analyze_conversation(
                    selected_user["conversation"], model
                )

                if "error" in gemini_analysis:
                    st.error(f"Analysis Error: {gemini_analysis['error']}")
                    if "raw_response" in gemini_analysis:
                        st.text("Raw response:")
                        st.text(gemini_analysis["raw_response"])
                else:
                    # Display analysis results
                    st.header("🧐 Chat Analysis Results")

                    tab1, tab2 = st.tabs(
                        [
                            "📊 Metrics",
                            "💡 Insights & Details",
                        ]
                    )

                    with tab1:
                        display_metrics(gemini_analysis)

                    with tab2:
                        st.subheader("📝 Overall Assessment")
                        st.info(
                            gemini_analysis.get(
                                "overall_assessment", "No overall assessment provided"
                            )
                        )

                        if "Sentiment Analysis" in analysis_options:
                            st.subheader("😊 Sentiment Analysis")
                            sentiments = [
                                calculate_sentiment(msg["message"])
                                for msg in selected_user["conversation"]
                            ]
                            sentiment_df = pd.DataFrame(
                                {
                                    "message_id": range(len(sentiments)),
                                    "sentiment": sentiments,
                                }
                            )
                            chart = (
                                alt.Chart(sentiment_df)
                                .mark_line()
                                .encode(
                                    x="message_id",
                                    y="sentiment",
                                    tooltip=["message_id", "sentiment"],
                                )
                                .properties(width=600, height=200)
                            )
                            st.altair_chart(chart, use_container_width=True)
                        if "Conversation Flow" in analysis_options:
                            st.subheader("🔀 Conversation Flow")
                            G = create_conversation_flow(selected_user["conversation"])
                            fig, ax = plt.subplots(figsize=(10, 5))
                            pos = nx.spring_layout(G)
                            nx.draw(
                                G,
                                pos,
                                with_labels=True,
                                node_color="lightblue",
                                node_size=500,
                                font_size=8,
                                arrows=True,
                            )
                            st.pyplot(fig)

                        if "Entity Extraction" in analysis_options:
                            st.subheader("🏷️ Top Entities Mentioned")
                            entities = [
                                item
                                for msg in selected_user["conversation"]
                                for item in extract_entities(msg["message"])
                            ]
                            entity_counts = Counter(entities)
                            entity_df = pd.DataFrame(
                                entity_counts.most_common(10),
                                columns=["Entity", "Count"],
                            )
                            st.bar_chart(entity_df.set_index("Entity"))

                        # st.subheader("🔍 Detailed Analysis")
                        col1, col2 = st.columns(2)
                        with col1:
                            for key in [
                                "misunderstandings",
                                "inconsistencies",
                                "hallucinations",
                            ]:
                                with st.expander(f"{key.capitalize()} 🔍"):
                                    for item in gemini_analysis.get(key, []):
                                        st.write(f"• {item}")
                        with col2:
                            for key in ["repetitions", "unresolved_issues"]:
                                with st.expander(f"{key.capitalize()} 🔍"):
                                    for item in gemini_analysis.get(key, []):
                                        st.write(f"• {item}")

            except Exception as e:
                logging.error(f"Error during analysis: {str(e)}")
                st.error(f"An error occurred during the analysis: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("Developed with ❤️ Sincera | © 2024")


if __name__ == "__main__":
    main()
