import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("tokenizer", use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained("microsoft/deberta-base")
    model.load_state_dict(torch.load("microsoft-deberta-base_fold0_best.pth", map_location=torch.device('cpu')))
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Prediction logic
def get_predicted_spans(text):
    encoding = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True)
    offset_mapping = encoding.pop("offset_mapping")[0]
    with torch.no_grad():
        outputs = model(**encoding)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

    spans = []
    current_span = []
    for idx, label in enumerate(predictions):
        if label > 0:  # label 0 = "O" (outside), label >0 = entity
            start, end = offset_mapping[idx].tolist()
            if start != end:
                current_span.append((start, end))
        elif current_span:
            span_text = text[current_span[0][0]:current_span[-1][1]]
            spans.append(span_text.strip())
            current_span = []
    
    # Catch leftover span
    if current_span:
        span_text = text[current_span[0][0]:current_span[-1][1]]
        spans.append(span_text.strip())
    
    return spans

# Streamlit UI
st.title("NBME Span Prediction App")
input_text = st.text_area("Enter a clinical note to extract relevant span(s):")

if st.button("Extract Span(s)"):
    if input_text.strip():
        spans = get_predicted_spans(input_text)
        if spans:
            st.success("Predicted Span(s):")
            for span in spans:
                st.write(f"🔹 `{span}`")
        else:
            st.info("No span detected.")
    else:
        st.warning("Please enter some text.")

