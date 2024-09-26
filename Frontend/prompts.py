from tqdm import tqdm
import openai

AI_SUMMARY_PROMPT = """I would like you to help me by summarizing a group of tweets, delimited by triple backticks, 
and each tweet is labeled by a number in a given format: number-[tweet]. 
Give me a comprehensive summary in a concise paragraph and as you generate each sentence, 
provide the comma seperated identifying numbers of the tweets on which that sentence is based, no other response is necessary."



Here are the actual tweets for you to summarize in the triple backticks:
```
{tweets}
```
"""


stanceClient = openai.OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="token-census",
)

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

        completion = stanceClient.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=prompt,
        )
        response = completion.choices[0].message.content
        responses.append(response)

    df["Stance"] = responses
    return df