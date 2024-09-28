import gradio as gr
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re
import json

from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="token-census",
)



# Load SBERT embedding_model
embedding_model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5", device="cuda" if torch.cuda.is_available() else "cpu"
)


# Function to extract JSON from text
def extract_json_from_string(text):
    start_idx = text.find("{")
    end_idx = text.rfind("}") + 1
    json_str = text[start_idx:end_idx]
    return json.loads(json_str)


# Import data
data = pd.read_csv(
    "~/projects/vector-search/data/allCensus_RT_sample.csv"
)  # Update the path to your CSV file
data = data.dropna(subset=["Message"]).reset_index(drop=True)
embeddings = embedding_model.encode(
    data["Message"],
    convert_to_tensor=True,
    show_progress_bar=True,
    batch_size=32,
    normalize_embeddings=True,
)


# Function to filter data and embeddings by SocialNetwork
def filter_data_and_embeddings_by_network(network, data, embeddings):
    if network == "BOTH":
        return data.reset_index(drop=True), embeddings
    else:
        mask = data["SocialNetwork"] == network
        return data[mask].reset_index(drop=True), embeddings[mask]


# Initially, load Twitter data
current_network = "TWITTER"
filtered_data, filtered_embeddings = filter_data_and_embeddings_by_network(
    current_network, data, embeddings
)


def search_vector(
    query,
    top_n,
    data=filtered_data,
    embeddings=filtered_embeddings,
    embedding_model=embedding_model,
):
    query_embedding = embedding_model.encode(
        query, convert_to_tensor=True, normalize_embeddings=True
    )
    posts = data["Message"]
    platforms = data["SocialNetwork"]
    cos_scores = torch.matmul(query_embedding, embeddings.T).to("cpu").numpy().flatten()
    top_results = np.argpartition(-cos_scores, range(min(top_n, len(posts))))
    results = [(posts[i], cos_scores[i], platforms[i]) for i in top_results[:top_n]]
    return results


def display_search_results(query, top_n, data, embeddings):
    results = search_vector(query, top_n, data=data, embeddings=embeddings)
    df = pd.DataFrame(results, columns=["Message", "Relevance", "SocialNetwork"])
    df["Relevance"] = df["Relevance"].apply(lambda x: f"{x:.4f}")
    return df


def stance_annotation(
    df,
    topic,
    stance1,
    stance2,
    stance3,
    stance4,
    example1,
    example2,
    example3,
    example4,
    progress=gr.Progress(),
    batch_size=256,
):
    stances = [stance1, stance2, stance3, stance4]
    examples = [example1, example2, example3, example4]

    responses = []
    total_rows = len(df)
    examples_content = "### Examples:\n" if any(examples) else ""

    if examples[0]:
        examples_content += (
            f"Tweet1: {examples[0]}\n{{\n    'stance': '{stances[0]}'\n}}\n\n"
        )
    if examples[1]:
        examples_content += (
            f"Tweet2: {examples[1]}\n{{\n    'stance': '{stances[1]}'\n}}\n\n"
        )
    if examples[2]:
        examples_content += (
            f"Tweet3: {examples[2]}\n{{\n    'stance': '{stances[2]}'\n}}\n\n"
        )
    if examples[3]:
        examples_content += (
            f"Tweet4: {examples[3]}\n{{\n    'stance': 'irrelevant'\n}}\n\n"
        )

    for i, row in tqdm(df.iterrows(), desc="Preparing Prompts", total=total_rows):
        prompt = [
            {
                "role": "system",
                "content": f"You are a human annotator. You will be presented with a tweet, delimited by triple backticks, concerning {topic}. Please make the following assessment:",
            },
            {
                "role": "user",
                "content": f"""
        Determine whether the tweet discusses the topic of {topic}. If it does, indicate the stance of the Twitter user who posted the tweet as '{stances[0]}', '{stances[1]}', '{stances[2]}', or '{stances[3]}'. Your response should be in JSON format as shown below and do not give me anything else:
        {{
            "stance": "selected stance here"
        }}

        The stance must be one of the following: "{stances[0]}", "{stances[1]}", "{stances[2]}", "irrelevant".

        {examples_content}

        ### Your Task:
        Tweet: ```{row['Message']}```
        """,
            },
        ]

        completion = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=prompt,
        )
        response = completion.choices[0].message.content
        responses.append(response)

    df["Stance"] = responses
    return df


def search_and_display(query, top_n, data, embeddings):
    results_df = display_search_results(query, top_n, data, embeddings)
    return results_df


def annotate_results(
    df,
    topic,
    stance1,
    stance2,
    stance3,
    stance4,
    example1,
    example2,
    example3,
    example4,
):
    if df is None or df.empty:
        return None, [], None
    progress_bar = gr.Progress()
    annotated_df = stance_annotation(
        df,
        topic,
        stance1,
        stance2,
        stance3,
        stance4,
        example1,
        example2,
        example3,
        example4,
        progress=progress_bar,
    )
    return (
        annotated_df,
        ["All"] + annotated_df["Stance"].unique().tolist(),
        annotated_df,
    )


def sort_by_stance(df):
    if "Stance" in df.columns:
        sorted_df = df.sort_values(by=["Stance"])
        return sorted_df
    else:
        return df


def switch_network(network):
    global filtered_data, filtered_embeddings
    filtered_data, filtered_embeddings = filter_data_and_embeddings_by_network(
        network, data, embeddings
    )
    return filtered_data, filtered_embeddings


with gr.Blocks() as demo:
    gr.Markdown("# Semantic Search and Stance Annotation")

    with gr.Row():
        network_selector = gr.Dropdown(
            label="Select Social Network",
            choices=["REDDIT", "TWITTER", "BOTH"],
            value="TWITTER",
        )
        query_input = gr.Textbox(
            lines=2, placeholder="Enter your query here...", label="Query"
        )
        top_n_input = gr.Number(value=100, precision=0, label="Number of Top Results")
        search_button = gr.Button("Search")

    with gr.Row():
        sort_button = gr.Button("Sort by Stance", size="sm")
        stance_filter_selection = gr.CheckboxGroup(
            label="Filter by Stance", choices=["All"], value=["All"]
        )

    results_df = gr.DataFrame(
        label="Search Results",
        interactive=True,
        wrap=True,
        column_widths=["50%", "20%", "15%", "15%"],
    )

    show_annotation = gr.Checkbox(label="Show Stance Annotation Inputs", value=False)

    with gr.Column(visible=False) as stance_inputs:
        topic_input = gr.Textbox(
            lines=1, placeholder="Enter the topic here...", label="Topic"
        )

        with gr.Row():
            stance1_input = gr.Textbox(
                lines=2, placeholder="Enter stance 1 here...", label="Stance 1"
            )
            stance2_input = gr.Textbox(
                lines=2, placeholder="Enter stance 2 here...", label="Stance 2"
            )

        with gr.Row():
            stance3_input = gr.Textbox(
                lines=2, placeholder="Enter stance 3 here...", label="Stance 3"
            )
            stance4_input = gr.Textbox(
                lines=2, placeholder="Enter stance 4 here...", label="Stance 4"
            )

        with gr.Row():
            example1_input = gr.Textbox(
                lines=2, placeholder="Enter example 1 here...", label="Example 1"
            )
            example2_input = gr.Textbox(
                lines=2, placeholder="Enter example 2 here...", label="Example 2"
            )

        with gr.Row():
            example3_input = gr.Textbox(
                lines=2, placeholder="Enter example 3 here...", label="Example 3"
            )
            example4_input = gr.Textbox(
                lines=2, placeholder="Enter example 4 here...", label="Example 4"
            )

        submit_annotation_button = gr.Button("Submit Annotation")

    # Create a State component to store the top_n value
    top_n_state = gr.State(value=100)
    stance_filter_state = gr.State(value=["All"])
    full_annotated_df = gr.State(None)

    def update_top_n(top_n):
        top_n_state.value = top_n
        return top_n_state.value

    top_n_input.change(update_top_n, inputs=top_n_input, outputs=top_n_state)

    def search_and_update(query, top_n, network):
        data, embeddings = switch_network(network)
        results_df = search_and_display(query, top_n, data, embeddings)
        return results_df

    def filter_results(full_df, stances):
        if full_df is None:
            return None
        if "All" in stances:
            return full_df
        if "Stance" in full_df.columns:
            filtered_df = full_df[full_df["Stance"].isin(stances)]
            return filtered_df
        else:
            return full_df

    def update_stance_selection(df):
        if df is None or "Stance" not in df.columns:
            return gr.update(choices=["All"], value=["All"])

        unique_stances = df["Stance"].unique().tolist()
        choices = ["All"] + unique_stances
        return gr.update(choices=choices, value=["All"])

    search_button.click(
        search_and_update,
        inputs=[query_input, top_n_state, network_selector],
        outputs=[results_df],
    )
    top_n_input.change(
        search_and_update,
        inputs=[query_input, top_n_state, network_selector],
        outputs=[results_df],
    )
    sort_button.click(sort_by_stance, inputs=[results_df], outputs=[results_df])
    stance_filter_selection.change(
        filter_results,
        inputs=[full_annotated_df, stance_filter_selection],
        outputs=[results_df],
    )

    show_annotation.change(
        lambda x: gr.Column(visible=x),
        inputs=[show_annotation],
        outputs=[stance_inputs],
    )

    submit_annotation_button.click(
        annotate_results,
        inputs=[
            results_df,
            topic_input,
            stance1_input,
            stance2_input,
            stance3_input,
            stance4_input,
            example1_input,
            example2_input,
            example3_input,
            example4_input,
        ],
        outputs=[results_df, stance_filter_selection, full_annotated_df],
        postprocess=True,
    ).then(
        fn=update_stance_selection,
        inputs=[full_annotated_df],
        outputs=[stance_filter_selection],
    )

demo.launch(share=True)
