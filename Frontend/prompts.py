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

summarizerClient = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-census",
)

def ai_summarize(tweets):
    llama3_gen_prompt = """system

{}user

{}assistant {}"""
    input_text = llama3_gen_prompt.format(
        AI_SUMMARY_PROMPT, 
        tweets, 
        ""
    ) 
    completion = summarizerClient.chat.completions.create(
        model="Lllama3TS_unsloth_vllm",
        messages=[{"role": "user", "content": input_text}]
    )
    result = completion.choices[0].message.content
    return result


stanceClient = openai.OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="token-census",
)

def stance_annotation(tweets, topic, stances, examples):
    # if examples[0]:
    #     examples_content += (
    #         f"Tweet1: {examples[0]}\n{{\n    'stance': '{stances[0]}'\n}}\n\n"
    #     )
    prompt = [
        {
            "role": "system",
            "content": f"You are a human annotator. You will be presented with a list of tweets, delimited by triple backticks, concerning {topic}. Please make the following assessment:",
        },
        {
            "role": "user",
            "content": f"""
    Determine whether each tweet discusses the topic of {topic}. If it does, indicate the stance of the Twitter user who posted the tweet as one of '{stances}', otherwise label the stance as "{{-1: irrelevant}}". Your response should be in JSON format as shown below. No other output is necessary:
    {{
        "tweet_number": "stance_number"
    }}

    The stance number must be between -1 and {len(stances)-1}.

    {examples}

    ### Your Task:
    Tweets: ```{tweets}```
    """,
        },
    ]

    completion = stanceClient.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=prompt,
    )
    response = completion.choices[0].message.content

    return response