import streamlit as st
import numpy as np
import pandas as pd
import requests
from io import BytesIO
import zipfile
from transformers import AutoModelWithLMHead, AutoTokenizer


@st.cache()
def load_data():

    url = 'https://dh-abstracts.library.cmu.edu/downloads/dh_conferences_works.csv'
    zipped = requests.get(url).content
    fp = BytesIO(zipped)
    csv = zipfile.ZipFile(fp, "r").open('dh_conferences_works.csv')
    data = pd.read_csv(csv)
    data = pd.DataFrame(data[data.languages == "English"])
    data = data[["work_title", "topics"]]
    data = pd.DataFrame(data[~data.topics.isnull()])
    data['topics'] = data.topics.apply(lambda x: x
        .replace(' / ', ';')
        .replace(' and ', ';')
        .replace(' & ', ';')
        .split(";")
    )
    all_topics = sorted(list(set(topic for topic_ll in data.topics for topic in topic_ll)))
    return data, all_topics


model = AutoModelWithLMHead.from_pretrained("xlnet-base-cased")    
tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
data, all_topics = load_data()

chosen_topics = st.multiselect("choose topics", all_topics, default=["visualization"])

PADDING_TEXT = ". ".join(data[
    np.logical_or.reduce([
        data.topics.apply(lambda x: kw in x) for kw in chosen_topics
    ])
].sample(8).work_title) + ".\n"


st.write("### Context ###")
st.write(PADDING_TEXT + "\n\n")

prompt = "I have just finished my article for the upcoming Digital Humanities conference. The {}. The title of the article is: "

chosen_topics_string = ""
if len(chosen_topics) == 1:
    chosen_topics_string = "topic of the article is " + chosen_topics[0]
elif len(chosen_topics) == 2:
    chosen_topics_string = "topics of the article are " + chosen_topics[0] + " and " + chosen_topics[1]
else:
    chosen_topics_string = "topics of the article are " + ", ".join(chosen_topics[:-1]) + " and " + chosen_topics[-1]

prompt = prompt.format(chosen_topics_string)

st.write("### Prompt: ###")
st.write(prompt)

inputs = tokenizer.encode(PADDING_TEXT + prompt, add_special_tokens=False, return_tensors="pt")


prompt_length = len(tokenizer.decode(inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
outputs = model.generate(inputs, max_length=250, do_sample=True, top_p=0.95, top_k=60)
generated = tokenizer.decode(outputs[0])[prompt_length:]


st.write("### Generated: ###")
st.write(generated)

