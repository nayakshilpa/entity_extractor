import streamlit as st
import pandas as pd
import ast
from llm_model_mapper.llm_mapper import LLMMapper

st.title("Cargo Description Entity Extractor")

# Load secrets
azure_endpoint = st.secrets["AZURE_ENDPOINT"]
azure_api_version = st.secrets["AZURE_API_VERSION"]
azure_api_key = st.secrets["AZURE_API_KEY"]
model_name = st.secrets["MODEL_NAME"]

# Helper function to call the LLM
def extract_entities(text: str) -> dict:
    prompt = f"Extract PartNumber, OverallCartonTotal and OverallPalletTotal from the following cargo description. {text}. No preamble."
    llm_creator = LLMMapper(
        llm_provider="langchain",
        framework="openai",
        model_name=model_name,
        api_key=azure_api_key,
        end_point=azure_endpoint,
        api_version=azure_api_version,
    )
    llm_creator.construct_llm()
    llm = llm_creator.model
    result = llm.invoke(str(prompt))
    content = getattr(result, "content", result)

    entity_dict = None
    try:
        entity_dict = ast.literal_eval(content)
        if not isinstance(entity_dict, dict):
            entity_dict = None
    except Exception:
        pass

    if entity_dict is None:
        entity_dict = {}
        for line in str(content).splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                entity_dict[key.strip()] = value.strip()

    return entity_dict or {"raw_output": content}


# Landing page with options
option = st.radio("Choose input method:", ["Free Text", "Bulk Upload"])

# --- Free Text Mode ---
if option == "Free Text":
    st.subheader("Enter Cargo Description Text")
    user_text = st.text_area("Cargo Description", height=200)

    if st.button("Extract Entities"):
        if user_text.strip() == "":
            st.warning("Please enter a cargo description text.")
        else:
            try:
                entities = extract_entities(user_text)
                st.subheader("Extracted Entities")
                for k, v in entities.items():
                    st.write(f"**{k}:** {v}")
            except Exception as e:
                st.error(f"Error extracting entities: {e}")

# --- Bulk Upload Mode ---
elif option == "Bulk Upload":
    st.subheader("Upload an Excel file with cargo descriptions")
    uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])

    if uploaded_file is not None:
        st.info("Uploading File and Processing...")

        try:
            df = pd.read_excel(uploaded_file)

            if "Cargo Description" not in df.columns:
                st.error("The Excel file must contain a 'Cargo Description' column.")
            else:
                results = []
                with st.spinner("Extracting entities from cargo descriptions..."):
                    for desc in df["Cargo Description"].dropna():
                        entities = extract_entities(desc)
                        results.append({
                            "Cargo Description": desc,
                            "PartNumber": entities.get("PartNumber", ""),
                            "OverallCartonTotal": entities.get("OverallCartonTotal", ""),
                            "OverallPalletTotal": entities.get("OverallPalletTotal", ""),
                        })

                results_df = pd.DataFrame(results)
                st.success("Extraction complete!")
                st.subheader("Extracted Results")
                st.dataframe(results_df)

                # Option to download results
                csv = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name="extracted_entities.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")
