
import streamlit as st
import os
import sys
import io
from script import process_requirements  # Import your pipeline function
from pathlib import Path


# Custom class to capture print statements
class StreamlitLogger(io.StringIO):
    def __init__(self, text_placeholder):
        super().__init__()
        self.text_placeholder = text_placeholder
        self.logs = ""

    def write(self, message):
        self.logs += message
        self.text_placeholder.text_area("Logs", self.logs, height=300)

    def flush(self):
        pass  # No need to implement for Streamlit

# Streamlit UI
st.title("Automated Test Case Generator")

st.text("Upload Excel file with 2 columns named 'req_id' and 'requirement'")
st.text("Suggestion: use the req.xlsx file provided in the repository")
st.markdown("View source code & req.xlsx on[GitHub](https://github.com/yashmantri20/test-case-generation)")

# File Upload
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file is not None:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyCYOURReNyI2g1G2jpEmw2yMj4AgiP9VyM"
    print(os.getenv("GOOGLE_API_KEY"))
    # Save uploaded file temporarily
    input_file = "uploaded_req.xlsx"
    with open(input_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"File {uploaded_file.name} uploaded successfully!")

    # Button to start processing
    if st.button("Generate Test Cases"):
        st.info("Processing started...")

        # Redirect stdout to capture print statements
        text_placeholder = st.empty()
        logger = StreamlitLogger(text_placeholder)
        sys.stdout = logger  # Redirect print statements

        # Run the pipeline
        result = process_requirements(input_file)

        # Reset stdout
        sys.stdout = sys.__stdout__

        if result:
            st.success("Test case generation completed!")
            for res in result["results"]:
                st.subheader(f"Requirement: {res['req_id']}")
                st.write(res["requirement"])
                st.code(res["test_cases"], language="markdown")
                
                # Provide download link for the generated file
                filename = f"./generatedTestCases/{res['req_id']}.xlsx"
                display_name = f"{res['req_id']}.xlsx"
                if Path(filename).exists():
                    with open(filename, "rb") as file:
                        st.download_button(label=f"Download {display_name}", data=file, file_name=display_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.error("An error occurred during processing.")
