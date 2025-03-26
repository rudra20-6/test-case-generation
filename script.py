

import os
import pandas as pd
from typing import Dict, List, Any, TypedDict
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

import warnings
warnings.filterwarnings("ignore")

# Configure the Google Generative AI model (Gemini 2.0 Flash)
def create_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.2,
        convert_system_message_to_human=True
    )

# Define the state structure using TypedDict
class State(TypedDict, total=False):
    requirements: List[Dict[str, str]]
    current_index: int
    results: List[Dict[str, Any]]
    input_file: str
    __next_node__: str  # We'll store the next node name here

# Load requirements from Excel file
def load_requirements(state: State) -> State:
    """Load requirements from an Excel file."""
    try:
        print(f"Loading requirements from {state['input_file']}...")
        df = pd.read_excel(state["input_file"])
        requirements = []
        for _, row in df.iterrows():
            requirements.append(dict(row))
        
        print(f"Loaded {len(requirements)} requirements")
        
        # Create a new state
        return {
            "requirements": requirements,
            "current_index": 0,  # Start with first requirement
            "results": [],
            "input_file": state["input_file"]
        }
    except Exception as e:
        print(f"Error loading requirements from {state['input_file']}: {e}")
        return state

# Generate test cases for a single requirement
def generate_test_cases(state: State) -> State:
    """Generate test cases for the current requirement using Gemini 2.0 Flash."""
    current_index = state["current_index"]
    requirements = state["requirements"]
    
    if current_index >= len(requirements):
        print(f"No more requirements to process. Index: {current_index}, Total: {len(requirements)}")
        return state
    
    current_req = requirements[current_index]
    req_id = current_req.get("req_id", f"req_{current_index + 1}")
    req_text = current_req.get("requirement", "")
    
    # print(f"Generating test cases for requirement {req_id} at index {current_index}...")
    
    prompt = f"""
    Generate EXHAUSTIVE test cases for the following requirement. 
    
    Requirement: {req_text}
    
    Create test cases that cover all possible scenarios, edge cases, and boundary conditions.
    Format your response ONLY as a table :
    - the table should contain a time column
    - it should contain separate columns for all the input parameters
    
    Remember the generated test cases will be pasted into an Excel file, and this .xlsx file will be used as a direct input to matlab simulink model to test it.
    Provide the test cases in a clean table format without any introduction or conclusion text.d
    
    """
    
    llm = create_llm()
    messages = [HumanMessage(content=prompt)]
    
    try:
        response = llm.invoke(messages)
        test_cases = response.content
        print(f"Successfully generated test cases for requirement {req_id}")
        
        # Create a new results list with the new result appended
        new_results = list(state.get("results", []))
        new_results.append({
            "req_id": req_id,
            "requirement": req_text,
            "test_cases": test_cases
        })
        
        # Return a new state with updated results and SAME current_index
        return {
            "requirements": state["requirements"],
            "current_index": current_index,  # Keep the same index
            "results": new_results,
            "input_file": state["input_file"]
        }
    except Exception as e:
        print(f"Error generating test cases for requirement {req_id}: {e}")
        new_results = list(state.get("results", []))
        new_results.append({
            "req_id": req_id,
            "requirement": req_text,
            "test_cases": f"Error: {str(e)}"
        })
        return {
            "requirements": state["requirements"],
            "current_index": current_index,  # Keep the same index
            "results": new_results,
            "input_file": state["input_file"]
        }

# # Save test cases to Excel file
# def save_test_cases(state: State) -> State:
#     """Save the generated test cases to individual Excel files."""
#     current_index = state["current_index"]
#     results = state["results"]
    
#     # The result to save should be the last one added
#     if not results:
#         print("No results to save")
#         return state
        
#     result = results[-1]  # Get the most recently added result
#     req_id = result["req_id"]
#     test_cases_text = result["test_cases"]
    
#     # print(f"Saving test cases for requirement {req_id} at index {current_index}...")
    
#     # Convert the markdown table to Excel
#     try:
#         # Parse the markdown table
#         lines = test_cases_text.strip().split('\n')
        
#         # Find the table content (skip headers and separator lines)
#         table_start = 0
#         for i, line in enumerate(lines):
#             if '|' in line and '-|-' in line:
#                 table_start = i + 1
#                 break
        
#         headers = [h.strip() for h in lines[table_start-2].split('|') if h.strip()]
        
#         rows = []
#         for line in lines[table_start:]:
#             if '|' in line:
#                 row = [cell.strip() for cell in line.split('|') if cell.strip()]
#                 if row:
#                     rows.append(row)
        
#         # Create DataFrame and save to Excel
#         df = pd.DataFrame(rows, columns=headers)
#         filename = f"./generatedTestCases/{req_id}.xlsx"
#         df.to_excel(filename, index=False)
        
#         print(f"Successfully saved test cases to {filename}")
        
#         # Return state unchanged - do NOT modify the index here
#         return state
#     except Exception as e:
#         print(f"Error saving test cases for requirement {req_id}: {e}")
#         return state

def save_test_cases(state: State) -> State:
    """Save the generated test cases to individual Excel files."""
    current_index = state["current_index"]
    results = state["results"]

    # The result to save should be the last one added
    if not results:
        print("No results to save")
        return state

    result = results[-1]  # Get the most recently added result
    req_id = result["req_id"]
    test_cases_text = result["test_cases"]

    try:
        # Parse the markdown table
        lines = test_cases_text.strip().split('\n')

        # Find the table separator line to locate headers
        table_start = 0
        for i, line in enumerate(lines):
            if '|' in line and '-|-' in line:
                table_start = i + 1
                break

        # Extract header row; assumes header row is two lines above the separator
        headers = [h.strip() for h in lines[table_start-2].split('|') if h.strip()]

        rows = []
        for line in lines[table_start:]:
            if '|' in line:
                row = [cell.strip() for cell in line.split('|') if cell.strip()]
                if row:
                    rows.append(row)

        # Fix row lengths: adjust rows to match header length
        fixed_rows = []
        for row in rows:
            if len(row) != len(headers):
                print(f"Warning: Row length {len(row)} doesn't match header length {len(headers)}. Adjusting row: {row}")
                if len(row) > len(headers):
                    row = row[:len(headers)]
                else:
                    row = row + [''] * (len(headers) - len(row))
            fixed_rows.append(row)

        # Create DataFrame and save to Excel
        df = pd.DataFrame(fixed_rows, columns=headers)
        filename = f"./generatedTestCases/{req_id}.xlsx"
        df.to_excel(filename, index=False)

        print(f"Successfully saved test cases to {filename}")
        return state
    except Exception as e:
        print(f"Error saving test cases for requirement {req_id}: {e}")
        return state



# Move to the next requirement
def increment_index(state: State) -> State:
    """
    Increment the current index and set the next node in state["__next_node__"].
    Return the updated state directly.
    """
    current_index = state["current_index"]
    new_index = current_index + 1
    
    # print(f"Moving from requirement index {current_index} to {new_index}")
    state["current_index"] = new_index  # update in-place
    
    if new_index < len(state["requirements"]):
        # print(f"Processing next requirement at index {new_index}")
        state["__next_node__"] = "generate_test_cases"
    else:
        print("Completed processing all requirements")
        state["__next_node__"] = END
    
    return state

# Create the LangGraph pipeline
def create_testing_pipeline():
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("load_requirements", load_requirements)
    workflow.add_node("generate_test_cases", generate_test_cases)
    workflow.add_node("save_test_cases", save_test_cases)
    workflow.add_node("increment_index", increment_index)
    
    # Add edges
    workflow.add_edge("load_requirements", "generate_test_cases")
    workflow.add_edge("generate_test_cases", "save_test_cases")
    workflow.add_edge("save_test_cases", "increment_index")
    
    # Now the next node is stored in state["__next_node__"]
    workflow.add_conditional_edges(
        "increment_index",
        lambda s: s["__next_node__"]
    )
    
    # Set the entry point
    workflow.set_entry_point("load_requirements")
    
    return workflow.compile()

# Main function to run the pipeline
def process_requirements(input_file: str):
    """Process all requirements from an Excel file and generate test cases."""
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist!")
        return None
        
    pipeline = create_testing_pipeline()
    
    # Initialize state as a dictionary
    initial_state = {
        "requirements": [],
        "current_index": 0,
        "results": [],
        "input_file": input_file
    }
    
    print(f"Starting processing of requirements from {input_file}")
    
    # Execute the pipeline
    try:
        result = pipeline.invoke(initial_state, {"recursion_limit": 150})
        
        # print(f"Processed {len(result['requirements'])} requirements.")
        print(f"Generated test cases for {len(result['results'])} requirements.")
        
        return result
    except Exception as e:
        print(f"Error in pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        return None

# Example usage
if __name__ == "__main__":
    # Make sure to set your Google API key
    os.environ["GOOGLE_API_KEY"] = "AIzaSyCYOURReNyI2g1G2jpEmw2yMj4AgiP9VyM"
    
    # Check if API key is set
    if "GOOGLE_API_KEY" not in os.environ:
        print("WARNING: GOOGLE_API_KEY not set. Please set your API key!")
    
    # Assume requirements.xlsx has columns: req_id, requirement
    process_requirements("req.xlsx")
