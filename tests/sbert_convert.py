import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu"
)
texts = pd.read_csv("T:\\projects\\Tweet_browser\\tests\\allCensus_sample.csv")[
    "Message"
].tolist()
embeddings = model.encode(texts, convert_to_tensor=True).cpu().numpy()
# save the embeddings to a csv file
np.savetxt(
    "T:\\projects\\Tweet_browser\\tests\\allCensus_sample_embeddings.csv",
    embeddings,
    delimiter=",",
)
