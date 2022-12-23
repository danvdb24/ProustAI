import streamlit as st
import pandas as pd
import numpy as np

import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer


st.title('Marcel Proust')
st.subheader("Commencez la phrase, l'algorithme la termine.")
st.write("Note : la génération du texte prend ~ 5 minutes")




with st.form("my_form"):
   
   text = st.text_input("Début de la phrase :", 'Je me souvenais de ce jour où Jean-Michel')
   
   # Every form must have a submit button.
   submitted = st.form_submit_button("Générer la suite")
   
   # Load the model ---
   model_checkpoint = "bigscience/bloom-560m"
   tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
   model = AutoModelForCausalLM.from_pretrained("dan-vdb/ProustAI")
   device = torch.device("cpu")

   # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

   pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, device=device
   )
   # ---

   
   if submitted:
       st.write(pipe(text, num_return_sequences=1, max_length=100)[0]["generated_text"])