import streamlit as st
from llm_model_mapper.llm_mapper import LLMMapper
import os
st.title("Cargo Description Entity Extractor")


# Hardcoded LLM configuration
azure_endpoint = os.environ.get("AZURE_ENDPOINT")
azure_api_version = os.environ.get("AZURE_API_VERSION")
azure_api_key = os.environ.get("AZURE_API_KEY")
model_name = os.environ.get("MODEL_NAME")

# Main input area
st.subheader("Enter Cargo Description Text")
user_text = st.text_area("Cargo Description", height=200)

if st.button("Extract Entities"):
    if user_text.strip() == "":
        st.warning("Please enter a cargo description text.")
    else:
        prompt = f"Extract PartNumber, OverallCartonTotal and OverallPalletTotal from the following cargo description. {user_text}. No preamble."
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
        try:
            result = llm.invoke(str(prompt))
            st.subheader("Extracted Entities")
            # Extract content if result is an AIMessage or similar object
            content = getattr(result, 'content', result)
            import ast
            entity_dict = None
            # Try parsing as dict
            try:
                entity_dict = ast.literal_eval(content)
                if not isinstance(entity_dict, dict):
                    entity_dict = None
            except Exception:
                pass
            # If not dict, try parsing as key: value lines
            if entity_dict is None:
                entity_dict = {}
                for line in str(content).splitlines():
                    if ':' in line:
                        key, value = line.split(':', 1)
                        entity_dict[key.strip()] = value.strip()
            if entity_dict:
                for k, v in entity_dict.items():
                    st.write(f"**{k}:** {v}")
            else:
                st.code(content)
        except Exception as e:
            st.error(f"Error extracting entities: {e}")
