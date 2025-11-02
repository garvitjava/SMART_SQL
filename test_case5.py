import streamlit as st
import pandas as pd
import json
import plotly.express as px
import re
import io
from sqlalchemy import create_engine, text, inspect
from groq import Groq

# --- Key Configuration ---

# A set of SQL tasks considered "destructive" or "state-changing"
DESTRUCTIVE_TASKS = {
    "CREATE_TABLE",
    "INSERT_INTO",
    "UPDATE",
    "DELETE",
    "ALTER_TABLE",
    "INSERT_CSV_EXISTING",
    "INSERT_CSV_NEW"
}

# --- 1. Groq LLM API Call Function (Unchanged) ---

def groq_call(api_key, system_prompt, user_prompt, model="llama-3.1-8b-instant", is_json=False):
    """
    Sends a prompt to the Groq API.
    """
    try:
        client = Groq(api_key=api_key)
        
        response_format = {"type": "json_object"} if is_json else {"type": "text"}
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=model,
            temperature=0.2,
            response_format=response_format
        )
        response = chat_completion.choices[0].message.content
        
        if is_json:
            return json.loads(response)
        
        return response
        
    except Exception as e:
        st.error(f"Groq API call failed: {e}")
        return None

# --- 2. Data Visualization (Unchanged) ---

def get_visualization_suggestion(api_key, data_columns):
    """
    Gets visualization suggestions from Groq in a structured JSON format.
    """
    system_prompt = (
        "You are a data visualization assistant. Your response MUST be a single, valid JSON object."
        "Suggest one visualization. Use *only* the column names provided."
    )
    user_prompt = f"""
These are the exact dataset column names: {data_columns}.

Return ONLY the JSON in this format:
{{
    "x": "column_name",
    "y": "column_name or list_of_column_names",
    "chart_type": "bar/line/scatter/pie"
}}
"""
    
    response_json = groq_call(
        api_key=api_key,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        is_json=True
    )
    
    return response_json


# --- 3. Demo Data Generator (Unchanged) ---

def generate_demo_data(api_key, user_input, num_rows=10):
    """
    Generates realistic demo data using the LLM, requested as JSON for robustness.
    """
    system_prompt = (
        "You are a demo data generator. You MUST respond with a single, valid JSON object."
        "Do not include any other text."
    )
    user_prompt = f"""
Generate a structured dataset with {num_rows} rows based on this request:
"{user_input}"

Return ONLY a JSON object in this exact format:
{{
    "columns": ["Column1Name", "Column2Name"],
    "rows": [
        ["value1", "value2"],
        ["value3", "value4"]
    ]
}}
"""
    
    response_json = groq_call(
        api_key=api_key,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        is_json=True
    )

    if response_json:
        try:
            df = pd.DataFrame(response_json['rows'], columns=response_json['columns'])
            file_path = "generated_data.csv"
            df.to_csv(file_path, index=False)
            return "✅ Demo data generated as CSV.", file_path
        except Exception as e:
            st.error(f"Error creating DataFrame from JSON: {e}")
            st.json(response_json)  # Show what was received
            return f"Error: Invalid JSON format received. {str(e)}", None
    
    return "Error: No valid JSON data received from the API.", None


# --- 4. SQL Task Classification (Unchanged) ---

def extract_sql_code_blocks(text):
    """Extracts SQL code blocks between ```sql and ```."""
    return re.findall(r"```sql\s+(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)

def classify_sql_task(api_key, user_input: str) -> str:
    """
    Classifies the user's input into a specific SQL task.
    """
    system_prompt = """
You are an expert SQL task classifier. Your job is to classify the user's input into ONE of the following tasks.
Respond with ONLY the task name and nothing else.
"""
    user_prompt = f"""
Classify this input into one of these tasks:
[CREATE_TABLE, INSERT_INTO, SELECT, UPDATE, DELETE, ALTER_TABLE, INSERT_CSV_EXISTING, INSERT_CSV_NEW, UNKNOWN]

User Input:
"{user_input}"

Task:
"""

    classification = groq_call(
        api_key=api_key,
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    
    if classification:
        cleaned = classification.strip().upper()
        for task in ["CREATE_TABLE", "INSERT_INTO", "SELECT", "UPDATE", "DELETE", "ALTER_TABLE", "INSERT_CSV_EXISTING", "INSERT_CSV_NEW"]:
            if task in cleaned:
                return task
    return "UNKNOWN"

# --- 5. SQL Generation (Unchanged) ---

def generate_sql(api_key, user_input, engine, task_type):
    """
    Generates a SQL query based on the task type.
    Does NOT execute it.
    """
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        system_prompt = "You are a highly accurate SQL code generation assistant. You must respond ONLY with a single SQL code block in markdown format (e.g., ```sql ... ```). Do not add any explanation before or after."

        if task_type == "CREATE_TABLE":
            system_prompt = "You are a SQL assistant. You MUST NOT use the `FOREIGN KEY` constraint. Any use of `FOREIGN KEY` is strictly forbidden. Respond ONLY with a single SQL code block."
            user_prompt = f"Existing tables: {tables}. \n\nAvoid conflicts. Generate a valid SQL CREATE TABLE query for: \n{user_input}"
        
        elif task_type == "INSERT_INTO":
            user_prompt = f"Existing tables: {tables}. \n\nGenerate a valid SQL INSERT INTO query for an existing table based on: \n{user_input}"
        
        elif task_type == "SELECT":
            user_prompt = f"Existing tables: {tables}. \n\nGenerate a valid SQL SELECT query based on: \n{user_input}"
        
        elif task_type == "UPDATE":
            user_prompt = f"Existing tables: {tables}. \n\nGenerate a valid SQL UPDATE query. Use a WHERE clause. Based on: \n{user_input}"
        
        elif task_type == "DELETE":
            user_prompt = f"Existing tables: {tables}. \n\nGenerate a valid SQL DELETE query. Use a WHERE clause. Based on: \n{user_input}"
        
        elif task_type == "ALTER_TABLE":
            user_prompt = f"Existing tables: {tables}. \n\nGenerate a valid SQL ALTER TABLE query based on: \n{user_input}"
        
        else:
            return None, "❌ Error: Invalid task type or unsupported instruction."

        raw_response = groq_call(
            api_key=api_key,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        if not raw_response:
            return None, "❌ Error: No response from LLM."
        
        sql_code_blocks = extract_sql_code_blocks(raw_response)
        
        if not sql_code_blocks:
            if "SELECT" in raw_response or "CREATE" in raw_response:
                 return raw_response, "✅ Generated SQL:"
            return None, f"❌ Error: LLM did not return a valid SQL code block. \nResponse: {raw_response}"
        
        sql_code = "\n".join(sql_code_blocks)
        return sql_code, "✅ Generated SQL (pending execution):"

    except Exception as e:
        return None, f"❌ Error in SQL generation: {e}"

# --- 6. SQL Execution & Explanation (Unchanged) ---

def execute_sql(sql_code, engine):
    """
    Executes a SQL query and returns the result.
    """
    try:
        if isinstance(sql_code, list):
            sql_code = "\n".join(sql_code)

        filtered_lines = []
        lines = sql_code.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip().upper()
            if stripped.startswith("FOREIGN KEY"):
                if filtered_lines:
                    filtered_lines[-1] = filtered_lines[-1].rstrip(',')
                continue
            filtered_lines.append(line)
        
        filtered_sql = "\n".join(filtered_lines)
        statements = [stmt.strip() for stmt in filtered_sql.split(';') if stmt.strip()]
        
        with engine.connect() as conn:
            result_data = None
            for stmt in statements:
                if stmt.upper().startswith("SELECT"):
                    result = conn.execute(text(stmt + ";"))
                    result_data = pd.DataFrame(result.fetchall(), columns=result.keys())
                else:
                    conn.execute(text(stmt + ";"))
            conn.commit()

        if result_data is not None:
            return "✅ Query executed successfully.", result_data
        else:
            return "✅ All non-SELECT queries executed successfully.", None

    except Exception as e:
        return f"❌ SQL Error: {e}", None

def explain_sql_command(api_key, sql_code, task_type):
    """
    Uses Groq to explain a SQL command.
    """
    system_prompt = "You are a SQL expert. Concisely explain what this SQL query does in plain English."
    user_prompt = f"Task: {task_type}\n\nSQL Code:\n{sql_code}\n\nExplanation:"
    
    explanation = groq_call(
        api_key=api_key,
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    return explanation

# --- 7. CSV Insert Functions (Unchanged) ---

def insert_csv_existing(table_name, csv_file, engine):
    try:
        df = pd.read_csv(csv_file)
        df.to_sql(table_name, engine, if_exists='append', index=False)
        return f"✅ Inserted CSV data into existing table '{table_name}'.", None
    except Exception as e:
        return f"❌ Error: {e}", None

def insert_csv_new(table_name, csv_file, engine):
    try:
        df = pd.read_csv(csv_file)
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        return f"✅ Created new table '{table_name}' from CSV.", None
    except Exception as e:
        return f"❌ Error: {e}", None

# --- 8. Streamlit App UI (UPGRADED) ---

st.set_page_config(page_title="10/10 AI Dashboard", layout="wide")
st.title("🤖 10/10 AI-Powered Multi-Feature Dashboard")

# --- Groq API Key ---
st.sidebar.title("🔐 API Configuration")
try: # <-- CHANGED
    # Read the key from Streamlit Secrets
    groq_api_key = st.secrets["GROQ_API_KEY"]
    st.sidebar.success("✅ Groq API Key loaded.")
except KeyError:
    st.sidebar.error("❌ GROQ_API_KEY not found. Add it to your Streamlit Secrets.")
    st.stop()
except Exception as e:
    st.sidebar.error(f"Error loading API key: {e}")
    st.stop()


# --- Navigation ---
st.sidebar.title("Navigation")
option = st.sidebar.radio("Select Feature", ["📊 Data Visualization", "🧠 SQL Query Generator", "📄 Demo Data Generator", "🧠 Smart SQL Task Handler"])

# --- Feature 1: Data Visualization ---
if option == "📊 Data Visualization":
    st.header("📊 Data Visualization")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        st.write("### Preview of Data")
        st.dataframe(df.head())

        if st.button("Suggest Visualization"):
            with st.spinner("🧠 Analyzing columns and getting suggestion..."):
                suggestion = get_visualization_suggestion(groq_api_key, list(df.columns))
            
            if suggestion:
                st.write("### Visualization Suggestion")
                st.json(suggestion)
                try:
                    chart_type = suggestion.get("chart_type").strip().lower()
                    x_col = suggestion.get("x", "").strip()
                    y_col = suggestion.get("y", [])

                    if isinstance(y_col, str): y_col = [y_col.strip()]
                    elif isinstance(y_col, list): y_col = [col.strip() for col in y_col]
                    
                    if not all(col in df.columns for col in [x_col] + y_col):
                        st.error("LLM suggested invalid columns. Please check suggestion.")
                    else:
                        st.write(f"### Plot: {chart_type.capitalize()} Chart")
                        if chart_type == "bar":
                            fig = px.bar(df, x=x_col, y=y_col, title=f"{x_col} vs {', '.join(y_col)}")
                        elif chart_type == "line":
                            fig = px.line(df, x=x_col, y=y_col, title=f"{x_col} vs {', '.join(y_col)}")
                        elif chart_type == "scatter":
                            fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {', '.join(y_col)}")
                        elif chart_type == "pie":
                            fig = px.pie(df, names=x_col, values=y_col[0], title=f"{x_col} vs {y_col[0]}")
                        else:
                            st.error(f"Unsupported chart type: {chart_type}")
                            st.stop()
                        
                        st.plotly_chart(fig)
                
                except Exception as e:
                    st.error(f"Error plotting visualization: {e}")

# --- Feature 2: Simple SQL Query Generator ---
elif option == "🧠 SQL Query Generator":
    st.header("🧠 Simple SQL Query Generator")
    st.info("This tool generates a SQL query from plain English but does *not* run it. For execution, use the 'Smart SQL Task Handler'.")
    text_input = st.text_area("Enter your Query here in Plain English:")
    if st.button("Generate SQL Query"):
        with st.spinner("Generating SQL Query..."):
            sql_code = groq_call(
                api_key=groq_api_key,
                system_prompt="You are a SQL generator. Respond ONLY with a single SQL code block.",
                user_prompt=text_input
            )
            st.code(sql_code, language="sql")

# --- Feature 3: Demo Data Generator ---
elif option == "📄 Demo Data Generator":
    st.header("📄 Demo Data Generator")
    user_input = st.text_area("Describe the dataset you want (e.g., '10 employees with name, role, and salary'):")
    num_rows = st.number_input("Number of rows", min_value=1, max_value=50, value=10)
    if st.button("Generate Dataset"):
        with st.spinner("Generating Demo Data..."):
            message, file_path = generate_demo_data(groq_api_key, user_input, num_rows)
        
        st.success(message)
        if file_path:
            with open(file_path, "rb") as f:
                st.download_button("Download CSV", f, file_name="generated_data.csv", mime="text/csv")

# --- Feature 4: Smart SQL Task Handler ---
elif option == "🧠 Smart SQL Task Handler":
    st.header("🧠 Smart SQL Task Handler")
    st.sidebar.header("🗂️ Database Connection")

    # --- CHANGED: Removed all manual input for DB connection ---
    # We now read DIRECTLY from Streamlit Secrets
    try:
        connection_url = st.secrets["DATABASE_URL"]
    except KeyError:
        st.sidebar.error("❌ DATABASE_URL not found. Add it to your Streamlit Secrets.")
        st.error("Database connection URL not configured by the app admin.")
        st.stop()
    except Exception as e:
        st.sidebar.error(f"Error loading DATABASE_URL: {e}")
        st.stop()
    # --- END OF CHANGES ---

    try:
        engine = create_engine(connection_url)
        with engine.connect():
            pass
        st.sidebar.success("✅ Connected to cloud database!")
    except Exception as e:
        st.sidebar.error(f"❌ Connection failed: {e}")
        st.stop()

    # --- SQL Task Input (Unchanged) ---
    user_input = st.text_area("Enter your SQL instruction (e.g., 'Create a table for users' or 'Show all employees'):")
    csv_file = st.file_uploader("Optional: Upload CSV file (for INSERT_CSV tasks)")
    
    if 'sql_code' not in st.session_state:
        st.session_state.sql_code = ""
    if 'task_type' not in st.session_state:
        st.session_state.task_type = ""

    if st.button("Analyze and Generate SQL"):
        if not user_input.strip():
            st.warning("Please enter a valid instruction.")
        else:
            with st.spinner("1. Classifying task..."):
                task_type = classify_sql_task(groq_api_key, user_input)
                st.session_state.task_type = task_type
                st.markdown(f"**🔍 Detected Task:** `{task_type}`")

            if task_type in ["INSERT_CSV_EXISTING", "INSERT_CSV_NEW"]:
                if not csv_file:
                    st.error("This task requires a CSV file. Please upload one.")
                    st.stop()
                table_name = st.text_input("Provide table name for CSV operation:", key="csv_table_name")
                if table_name:
                    if task_type == "INSERT_CSV_EXISTING":
                        msg, _ = insert_csv_existing(table_name, csv_file, engine)
                    else:
                        msg, _ = insert_csv_new(table_name, csv_file, engine)
                    st.info(msg)
                else:
                    st.warning("Please provide a table name.")
            
            elif task_type == "UNKNOWN":
                st.error("Could not determine the task. Please rephrase your request.")

            else:
                with st.spinner("2. Generating SQL..."):
                    sql_code, msg = generate_sql(groq_api_key, user_input, engine, task_type)
                    st.session_state.sql_code = sql_code
                    
                    if sql_code:
                        st.code(sql_code, language="sql")
                        with st.spinner("3. Explaining SQL..."):
                            explanation = explain_sql_command(groq_api_key, sql_code, task_type)
                            st.write("### 💬 AI Explanation")
                            st.markdown(explanation)
                    else:
                        st.error(msg)

    # --- Security Checkpoint (Unchanged) ---
    if st.session_state.sql_code:
        
        if st.session_state.task_type == "SELECT":
            st.write("---")
            if st.button("🚀 Run SELECT Query"):
                with st.spinner("Executing SELECT query..."):
                    msg, data = execute_sql(st.session_state.sql_code, engine)
                    st.info(msg)
                    if data is not None:
                        st.dataframe(data)

        elif st.session_state.task_type in DESTRUCTIVE_TASKS:
            st.write("---")
            st.warning(f"**SECURITY WARNING:** This is a destructive task (`{st.session_state.task_type}`).")
            if st.button("🚨 Confirm and Execute Destructive Query"):
                with st.spinner(f"Executing {st.session_state.task_type} query..."):
                    msg, data = execute_sql(st.session_state.sql_code, engine)
                    st.info(msg)
                    if data is not None:
                        st.dataframe(data)