import streamlit as st
import sys
from io import StringIO
from groq import Groq
import os
import re
from dotenv import load_dotenv
import pandas as pd
from typing import Literal, TypedDict, Annotated
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from pprint import pprint
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage
from langchain_openai import ChatOpenAI
from clinical_trials_module import get_clinical_trials_data
from pprint import pprint

# Set up environment variables
os.environ["GROQ_API_KEY"]= st.secrets["GROQ_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]


# Initialize Streamlit app
st.title("Clinical Trial Data Analysis using DeepSeek 🐳")

if  'text' not in st.session_state:
    st.session_state.CONNECTED =  False
    st.session_state.text = ''

def _connect_form_cb(connect_status):
    st.session_state.CONNECTED = connect_status

def display_db_connection_menu():
    with st.form(key="connect_form"):
        st.text_input('Enter the condition', help='Click on search, pressing enter will not work', value=st.session_state.text, key='text')
        submit_button = st.form_submit_button(label='Search', on_click=_connect_form_cb, args=(True,))
        if submit_button:
            if st.session_state.text=='':
                st.write("Please enter a condition")
                st.stop()



display_db_connection_menu()

if st.session_state.CONNECTED:
    # st.write('You are Searching for:',  st.session_state.text)

    # Load data and initialize session state
    if 'df' not in st.session_state:
        with st.spinner(f'🔍 Fetching data for: **{st.session_state.text}**'):
            st.session_state.df = get_clinical_trials_data(st.session_state.text)
                
        # Update status message
        st.success(f"✅ Data fetched successfully for '{st.session_state.text}'!")
        
        st.session_state.df = get_clinical_trials_data(st.session_state.text)
        st.session_state.context = """Pandas DataFrame you are using is name 'df' :
            Useful for answering questions related to clinical trials
            The column headings and its description are given
            -nctId: The unique identifier for each clinical trial registered on ClinicalTrials.gov.
            -organization:  The name of the organization conducting the clinical trial.
            -organizationType:  The type of organization, such as 'OTHER', 'INDUSTRY', 'NIH', 'OTHER_GOV', 'INDIV', 'FED', 'NETWORK', 'UNKNOWN'.
            -briefTitle:  A short title for the clinical trial, intended for easy reference.
            -officialTitle:  The full official title of the clinical trial.
            -statusVerifiedDate:  The date when the status of the clinical trial was last verified.
            -overallStatus:  The current overall status of the clinical trial like 'COMPLETED', 'UNKNOWN', 'ACTIVE_NOT_RECRUITING', 'RECRUITING', 'WITHDRAWN', 'TERMINATED', 'ENROLLING_BY_INVITATION', 'NOT_YET_RECRUITING', 'APPROVED_FOR_MARKETING', 'SUSPENDED','AVAILABLE'.
            -hasExpandedAccess:  It has boolean values and it indicates whether the clinical trial includes expanded access to the investigational drug or device outside of the clinical trial.
            -startDate:  The date when the clinical trial began.
            -completionDate:  The date when the clinical trial was completed.
            -completionDateType:  The type of completion date, specifying whether it refers to the ACTUAL or ESTIMATED.
            -studyFirstSubmitDate:  The date when the clinical trial information was first submitted to ClinicalTrials.gov.
            -studyFirstPostDate:  The date when the clinical trial information was first posted on ClinicalTrials.gov.
            -lastUpdatePostDate:  The date when the clinical trial information was last updated on ClinicalTrials.gov.
            -lastUpdatePostDateType:  The type of last update post date, specifying whether it refers to the actual or anticipated date.
            -HasResults:  It contains boolean values and indicates whether the results of the clinical trial have been posted on ClinicalTrials.gov.
            -responsibleParty:  The individual or organization responsible for the overall conduct of the clinical trial.
            -leadSponsor:  The primary sponsor responsible for the initiation, management, and financing of the clinical trial.
            -leadSponsorType:  The type of the lead sponsor, such as academic, industry, or government.
            -collaborators:  Other organizations or individuals collaborating on the clinical trial.
            -collaboratorsType:  The types of collaborators involved in the clinical trial.
            -briefSummary:  A brief summary of the clinical trial, providing an overview of the study's purpose and key details.
            -detailedDescription:  A detailed description of the clinical trial, including comprehensive information about the study design, methodology, and objectives.
            -conditions:  The medical conditions or diseases being studied in the clinical trial.
            -studyType:  The type of study (e.g., 'INTERVENTIONAL', 'OBSERVATIONAL', 'EXPANDED_ACCESS').
            -phases:  The phase of the clinical trial (e.g., 'NA', 'PHASE2', 'PHASE2, PHASE3', 'PHASE3', 'PHASE1', 'PHASE4','PHASE1, PHASE2', 'EARLY_PHASE1').
            -allocation:  The method of assigning participants to different arms of the clinical trial (e.g., 'RANDOMIZED','NON_RANDOMIZED').
            -interventionModel:  The model of intervention used in the clinical trial (e.g., 'SINGLE_GROUP', 'PARALLEL', 'CROSSOVER', 'SEQUENTIAL', 'FACTORIAL').
            -primaryPurpose:  The primary purpose of the clinical trial like 'PREVENTION', 'TREATMENT', 'SUPPORTIVE_CARE','BASIC_SCIENCE', 'DIAGNOSTIC', 'OTHER', 'ECT', 'SCREENING','HEALTH_SERVICES_RESEARCH', 'DEVICE_FEASIBILITY').
            -masking:  The method used to prevent bias by concealing the allocation of participants (e.g., 'QUADRUPLE', 'NONE', 'DOUBLE', 'TRIPLE', 'SINGLE').
            -whoMasked:  Specifies who is masked in the clinical trial etc. PARTICIPANT, INVESTIGATOR etc).
            -enrollmentCount:  The number of participants enrolled in the clinical trial.
            -enrollmentType:  The type of enrollment, specifying whether the number is ACTUAL or ESTIMATED.
            -arms:  The number of rms or groups in the clinical trial.
            -interventionDrug:  The drugs or medications being tested or used as interventions in the clinical trial.
            -interventionDescription:  Descriptions of the interventions used in the clinical trial.
            -interventionOthers:  Other types of interventions used in the clinical trial (e.g., devices, procedures).
            -primaryOutcomes:  The primary outcome measures being assessed in the clinical trial.
            -secondaryOutcomes:  The secondary outcome measures being assessed in the clinical trial.
            -eligibilityCriteria:  The criteria that determine whether individuals can participate in the clinical trial.
            -healthyVolunteers:  Indicates whether healthy volunteers are accepted in the clinical trial.
            -eligibilityGender:  The gender eligibility criteria for participants in the clinical trial.
            -eligibilityMinimumAge:  The minimum age of participants eligible for the clinical trial.
            -eligibilityMaximumAge:  The maximum age of participants eligible for the clinical trial.
            -eligibilityStandardAges:  Standard age groups eligible for the clinical trial.
            -LocationName:  The names of the locations where the clinical trial is being conducted.
            -city:  The city where the clinical trial locations are situated.
            -state:  The state where the clinical trial locations are situated.
            -country:  The country where the clinical trial locations are situated.
            -interventionBiological:  Biological interventions (e.g., vaccines, blood products) used in the clinical trial.
    """
        st.session_state.messages = []

        #Display the dataframe
        st.write('Data Sample: Top 10 rows from the data')
        st.dataframe(st.session_state.df.head(10))

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "```text" in message["content"]:
                # Extract and display the code block properly
                code_content = message["content"].split("```text")[1].strip().strip("```")
                st.code(code_content, language='text')
            else:
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask your clinical trial analysis question:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare the graph state
        class MessageState(TypedDict):
            messages: list
            code: str
            result: dict

        # Define the workflow functions
        def generate_code(state: MessageState):
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            system_prompt = """Generate Python code that:
        - Uses only provided variables
        - Prints results
        - Includes necessary imports
        - Fixes errors while maintaining logic
        """

            available_variables = list(st.session_state.df.columns)

            user_message = f"""{st.session_state.context}
            Available variables: {available_variables}
            Task: {state['messages'][-1]}"""


            # Error-based rectification request
            user_message_rectify = f"""The previous code:
                {state['code'] if isinstance(state.get('code'), str) else 'N/A'}

                gave the following error:
                {state['result']['error'] if isinstance(state.get('result', {}).get('error'), str) else 'N/A'}


                Based on the task: {state['messages'][-1]} and available variables: {available_variables},
                please fix the error while maintaining the original logic."""

            # Choose the correct prompt based on whether there's an error
            selected_message = user_message if not state.get('error') else user_message_rectify

            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": selected_message}
                ],
                model="deepseek-r1-distill-llama-70b",
            )

            def clean_code_response(response):
                response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
                code_match = re.search(r'```python\n(.*?)```', response, flags=re.DOTALL)
                return code_match.group(1).strip() if code_match else response.strip()

            response = chat_completion.choices[0].message.content
            return {'code': clean_code_response(response)}

        def agent(state: MessageState):
            def execute_python(code):
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                local_vars = {"df": st.session_state.df}

                try:
                    exec(code, local_vars)
                    output = sys.stdout.getvalue()
                    return {"output": output, "error": None}
                except Exception as e:
                    return {"output": None, "error": str(e)}
                finally:
                    sys.stdout = old_stdout

            result = execute_python(state['code'])
            return {"result": result}

        def check_for_errors(state):
            error = state["result"].get("error", None)
            state.setdefault("retry_count", 0)
            MAX_RETRIES = 3

            if error and state["retry_count"] < MAX_RETRIES:
                state["retry_count"] += 1
                return "retry"
            return "final"

        # Build and run the graph
        builder = StateGraph(MessageState)
        builder.add_node("generate_code", generate_code)
        builder.add_node("agent", agent)
        builder.add_edge(START, "generate_code")
        builder.add_edge("generate_code", "agent")
        builder.add_conditional_edges("agent", check_for_errors, {"retry": "generate_code", "final": END})
        graph = builder.compile()

        # Invoke the graph
        answer = graph.invoke({
            "messages": [HumanMessage(content=prompt)],
            "code": "",
            "result": {},
        })

        # Display assistant response
        response = answer['result']['output'] if answer['result'].get('output') else f"Error: {answer['result'].get('error')}"
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display assistant response
        with st.chat_message("assistant"):
            # Always show generated code
            # st.code(answer.get('code', ''), language='python')
            with st.expander("View Code"):
                code_placeholder = st.empty()
                full_code= answer.get('code', '')
                code_placeholder.code(full_code, language="python")
                
            # Handle execution results
            if answer['result'].get('output'):
                # Preserve whitespace formatting using a code block
                st.code(answer['result']['output'], language='text')
            elif answer['result'].get('error'):
                st.error(f"Error: {answer['result']['error']}")
            
            # Update chat history with formatted output
            # formatted_response = f"```text\n{answer['result'].get('output', '')}\n```" if answer['result'].get('output') else f"Error: {answer['result'].get('error')}"
            # st.session_state.messages.append({"role": "assistant", "content": formatted_response})
