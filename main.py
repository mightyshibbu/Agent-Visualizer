import os
import json
import logging
import sys
import torch
import tempfile
import streamlit as st
import pandas as pd
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from chromadb import Client, Settings
from chromadb.utils import embedding_functions
import google.generativeai as genai
from more_itertools import batched
from streamlit.components.v1 import html

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Log PyTorch and CUDA info
try:
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info(f"PyTorch classes: {dir(torch.classes) if hasattr(torch, 'classes') else 'No torch.classes'}")
except Exception as e:
    logger.error(f"Error getting PyTorch info: {e}")

# Log environment variables
logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
logger.info(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")

# Check for CUDA_HOME
cuda_home = os.environ.get('CUDA_HOME', 'Not set')
cuda_path = os.environ.get('CUDA_PATH', 'Not set')
logger.info(f"CUDA_HOME: {cuda_home}")
logger.info(f"CUDA_PATH: {cuda_path}")

# --- Load environment variables ---
load_dotenv()

# --- API Configurations ---
if 'api_config' not in st.session_state:
    st.session_state.api_config = {
        'gemini': {
            'api_key': os.getenv('GOOGLE_API_KEY'),
            'model': os.getenv('GEMINI_MODEL', 'gemini-1.5-flash-latest'),
            'temperature': float(os.getenv('GEMINI_TEMPERATURE', '0.5')),
            'max_tokens': int(os.getenv('GEMINI_MAX_TOKENS', '8192')),
            'top_p': float(os.getenv('GEMINI_TOP_P', '0.95')),
        }
    }
    # Configure the Gemini API
    try:
        genai.configure(api_key=st.session_state.api_config['gemini']['api_key'])
    except Exception as e:
        logger.error(f"Failed to configure Gemini: {str(e)}")

# Sidebar: API Config
st.sidebar.title("API Configurations")
with st.sidebar.expander("✨ Gemini"):
    cfg = st.session_state.api_config['gemini']
    st.write(f"**Model:** {cfg['model']}")
    st.write(f"**Temperature:** {cfg['temperature']}")
    st.write(f"**Max Tokens:** {cfg['max_tokens']}")
    st.write(f"**Top P:** {cfg['top_p']}")

# --- ChromaDB Initialization with persistent storage ---
logger.info("Initializing ChromaDB client...")

# Configuration
DATA_DIR = Path("./data")
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize ChromaDB with persistent storage
client = Client(Settings(
    persist_directory=str(EMBEDDINGS_DIR),
    is_persistent=True
))

# Collection name
COLL_NAME = "agent_grapher_data"

def get_file_hash(uploaded_file):
    """Generate a hash for the uploaded file to track if it's been processed."""
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
    return f"{file_hash}_{uploaded_file.name}"

def get_available_features(df):
    """Extract and return available features from the dataframe."""
    return {
        'columns': list(df.columns),
        'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist()
    }

def get_or_create_collection(uploaded_file=None):
    """
    Get existing collection or create a new one if it doesn't exist.
    If uploaded_file is provided, checks if it matches existing embeddings.
    """
    try:
        logger.info(f"Attempting to get existing collection: {COLL_NAME}")
        collection = client.get_collection(COLL_NAME)
        
        if uploaded_file:
            # Check if this file has been processed before
            file_hash = get_file_hash(uploaded_file)
            existing_hashes = set()
            
            # Get all existing file hashes from the collection
            try:
                existing_metadatas = collection.get()['metadatas']
                if existing_metadatas:
                    existing_hashes = {m.get('file_hash') for m in existing_metadatas if m and 'file_hash' in m}
            except Exception as e:
                logger.warning(f"Error getting existing hashes: {str(e)}")
            
            if file_hash not in existing_hashes:
                logger.info("New file detected, will update embeddings")
                return create_new_collection(uploaded_file)
            
            logger.info("Using existing embeddings for this file")
            features = get_available_features(st.session_state.df)
            st.sidebar.success("✅ Using existing embeddings")
            st.sidebar.subheader("Available Features")
            st.sidebar.json(features)
            
        return collection
        
    except Exception as e:
        if "does not exist" in str(e):
            if uploaded_file:
                logger.info(f"Collection {COLL_NAME} not found, creating a new one...")
                return create_new_collection(uploaded_file)
            else:
                st.sidebar.warning("⚠️ No existing embeddings found. Please upload a file to create embeddings.")
                return None
        else:
            logger.error(f"Error accessing collection: {str(e)}", exc_info=True)
            raise

def create_new_collection(uploaded_file):
    """Create a new ChromaDB collection with the specified configuration and process the uploaded file."""
    if not uploaded_file:
        st.error("No file provided for creating embeddings.")
        return None
        
    logger.info("Initializing SentenceTransformer model...")
    
    try:
        # Save the uploaded file to a temporary location
        temp_file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Read the file based on its extension
        try:
            if uploaded_file.name.lower().endswith('.csv'):
                df = pd.read_csv(temp_file_path)
            else:
                # Try Excel first, if it fails, try other formats
                try:
                    df = pd.read_excel(temp_file_path)
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    logger.error(f"Error reading file: {str(e)}", exc_info=True)
                    return None
            
            # Clean the dataframe
            df = df.dropna(how='all')
            if df.empty:
                st.error("The uploaded file is empty or contains no valid data.")
                return None
                
        finally:
            # Clean up the temporary file
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.warning(f"Could not remove temporary file {temp_file_path}: {str(e)}")
        
        # Initialize embedding model
        logger.info("Attempting to import sentence_transformers...")
        import sentence_transformers
        logger.info(f"Successfully imported sentence_transformers version: {sentence_transformers.__version__}")
        
        logger.info("Attempting to load model 'all-MiniLM-L6-v2'...")
        model = sentence_transformers.SentenceTransformer(
            'all-MiniLM-L6-v2',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info("Successfully loaded SentenceTransformer model")
        
        logger.info("Creating embedding function...")
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info("Successfully created embedding function")
        
        # Create or get collection
        logger.info("Creating/Getting ChromaDB collection...")
        collection = client.get_or_create_collection(
            name=COLL_NAME,
            embedding_function=embedding_func,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Prepare embedding data
        file_hash = get_file_hash(uploaded_file)
        texts, metadatas, ids = [], [], []
        
        for idx, row in df.iterrows():
            text = ' | '.join(f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col]))
            texts.append(text)
            metadatas.append({
                'row': idx,
                'file_name': uploaded_file.name,
                'file_hash': file_hash
            })
            ids.append(f"{file_hash}_row_{idx}")
        
        # Add embeddings to collection in batches
        for batch in batched(range(len(texts)), 100):
            collection.add(
                documents=[texts[i] for i in batch],
                metadatas=[metadatas[i] for i in batch],
                ids=[ids[i] for i in batch]
            )
        
        # Show success message and available features
        features = get_available_features(df)
        st.sidebar.success("✅ Created new embeddings")
        st.sidebar.subheader("Available Features")
        st.sidebar.json(features)
        
        return collection
        
    except Exception as e:
        logger.error(f"Error creating collection: {str(e)}", exc_info=True)
        st.error(f"Failed to create collection: {str(e)}")
        return None

def safe_eval_arithmetic(expr):
    """Safely evaluate arithmetic expressions in chart data."""
    if not isinstance(expr, str):
        return expr
        
    # Skip if it's a color code or doesn't contain operators
    if any(char in expr for char in ['rgba', '#']):
        return expr
        
    # Try to evaluate simple arithmetic expressions
    try:
        # First try direct evaluation (safest)
        if all(c in '0123456789+*/-() .' for c in expr):
            return eval(expr, {"__builtins__": {}}, {})
    except:
        pass
        
    return expr

def process_chart_config(config):
    """Recursively process chart config to evaluate arithmetic expressions."""
    if isinstance(config, dict):
        return {k: process_chart_config(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [process_chart_config(item) for item in config]
    elif isinstance(config, str):
        # Check if it's a simple arithmetic expression
        if any(op in config for op in '+-*/') and all(c in '0123456789+*/-() .' for c in config):
            try:
                result = safe_eval_arithmetic(config)
                if isinstance(result, (int, float)):
                    return result
            except:
                pass
    return config

def evaluate_arithmetic_in_json(json_str):
    """Evaluate arithmetic expressions in JSON strings."""
    import re
    import ast
    
    def safe_eval(expr):
        try:
            # Only evaluate if it's a simple arithmetic expression
            if any(op in expr for op in '+-*/') and all(c in '0123456789+*/-() .' for c in expr):
                # Use ast.literal_eval for safer evaluation
                node = ast.parse(expr, mode='eval')
                if all(isinstance(n, (ast.Expression, ast.BinOp, ast.Num, ast.Operator, ast.UnaryOp, ast.USub, ast.UAdd)) for n in ast.walk(node)):
                    result = eval(expr, {"__builtins__": {}}, {})
                    return str(int(result) if isinstance(result, float) and result.is_integer() else result)
        except Exception as e:
            pass
        return None
    
    # First, process all arithmetic expressions in arrays
    def process_arrays(match):
        array_str = match.group(0)
        # Find and evaluate all arithmetic expressions in the array
        def eval_match(m):
            result = safe_eval(m.group(0))
            return result if result is not None else m.group(0)
        
        # Handle both simple arrays and arrays within data objects
        processed = re.sub(r'\b(?:\d+\s*[+\-*/]\s*)+\d+\b', eval_match, array_str)
        return processed
    
    # Process arrays in the JSON string
    # This pattern matches simple arrays (not handling deeply nested arrays for now)
    array_pattern = r'\[[^\[\]]*\]'
    # Process the JSON string in a loop until no more arrays are found
    processed_json = json_str
    while True:
        new_json = re.sub(array_pattern, process_arrays, processed_json, flags=re.DOTALL)
        if new_json == processed_json:  # No more changes
            break
        processed_json = new_json
    
    return processed_json

def render_chartjs(charts_json_str: str):
    try:
        # Clean the JSON string first
        cleaned_json = charts_json_str.strip()
        
        # Remove any markdown code block markers if present
        if cleaned_json.startswith('```json'):
            cleaned_json = cleaned_json[7:]
        if cleaned_json.endswith('```'):
            cleaned_json = cleaned_json[:-3]
            
        # Evaluate arithmetic expressions in the JSON string
        cleaned_json = evaluate_arithmetic_in_json(cleaned_json)
        
        # Try to parse the JSON
        try:
            configs = json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues
            try:
                # Try to evaluate arithmetic expressions in arrays
                import re
                
                # Find all array-like patterns that might contain arithmetic
                def eval_arrays(match):
                    content = match.group(1)
                    try:
                        # Split by commas but respect nested structures
                        parts = []
                        current = ""
                        bracket_level = 0
                        
                        for char in content + ',':
                            if char in '{[':
                                bracket_level += 1
                            elif char in '}]':
                                bracket_level -= 1
                                
                            if char == ',' and bracket_level == 0:
                                parts.append(current.strip())
                                current = ""
                            else:
                                current += char
                        
                        # Evaluate each part
                        evaluated = []
                        for part in parts:
                            try:
                                # Check if it's a simple arithmetic expression
                                if any(op in part for op in '+-*/') and all(c in '0123456789+*/-() .' for c in part):
                                    evaluated.append(str(safe_eval_arithmetic(part)))
                                else:
                                    evaluated.append(part)
                            except:
                                evaluated.append(part)
                        
                        return '[' + ', '.join(evaluated) + ']'
                    except:
                        return match.group(0)
                
                # Apply the array evaluator to all array-like patterns
                fixed_json = re.sub(r'\[([^\[\]{}]*(?:\{[^{}]*\}|\[[^\[\]]*\])?[^\[\]{}]*)\]', 
                                 eval_arrays, 
                                 cleaned_json)
                
                # Try parsing again
                configs = json.loads(fixed_json)
                
            except Exception as e2:
                st.error(f"Failed to parse chart configuration. The generated JSON might be malformed.")
                logger.error(f"JSON Parse Error: {str(e2)}\nOriginal JSON:\n{charts_json_str}")
                logger.error(f"Cleaned JSON:\n{cleaned_json}")
                return
        
        if not isinstance(configs, list):
            configs = [configs]
            
        # Process each config to handle any arithmetic expressions
        configs = [process_chart_config(config) for config in configs]
            
        # Calculate number of charts per row based on number of charts
        num_charts = len(configs)
        if num_charts == 0:
            st.warning("No charts to display.")
            return
            
        # Determine layout: 1, 2, or 3 charts per row
        if num_charts == 1:
            cols = [st.container()]
        elif num_charts == 2:
            col1, col2 = st.columns(2)
            cols = [col1, col2]
        elif num_charts == 3:
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
        else:  # 4 or more charts
            cols = st.columns(2)  # Default to 2 columns for 4+ charts
        
        # Render each chart in its own container
        for i, cfg in enumerate(configs):
            container = cols[i % len(cols)]
            with container:
                container_id = f"chart_{i}"
                chart_config_str = json.dumps(cfg)
                
                # Chart dimensions
                chart_height = 500
                
                chart_html = f"""
                <div style="margin: 20px 0;">
                    <canvas id="{container_id}" height="{chart_height}"></canvas>
                </div>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <script>
                    const ctx_{i} = document.getElementById('{container_id}').getContext('2d');
                    const config = {chart_config_str};
                    
                    // Ensure options exist
                    if (!config.options) config.options = {{}};
                    
                    // Set responsive options
                    config.options.responsive = true;
                    config.options.maintainAspectRatio = false;
                    
                    // Create the chart
                    const chart_{i} = new Chart(ctx_{i}, config);
                </script>
                """
                html(chart_html, height=chart_height + 20, scrolling=False)
                
    except json.JSONDecodeError:
        st.error("Invalid JSON from LLM — cannot render charts.")
    except Exception as e:
        st.error(f"Error rendering charts: {str(e)}")
        logger.error(f"Chart rendering error: {str(e)}", exc_info=True)

# --- Streamlit App ---
st.title("Agent Grapher: AI Chart Generator")

# Initialize collection and dataframe
collection = None
df = None

# File upload and initialization
uploaded_file = st.file_uploader("Upload CSV/Excel data", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Reset file pointer and read the file
    uploaded_file.seek(0)
    try:
        # Read the file based on extension
        if uploaded_file.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            st.stop()
            
        if df.empty:
            st.error("The uploaded file is empty. Please upload a valid file with data.")
            st.stop()
            
        # Store in session state
        st.session_state.df = df
        st.session_state.file_hash = get_file_hash(uploaded_file)
        
        # Get or create collection with the uploaded file
        collection = get_or_create_collection(uploaded_file)
        
    except Exception as e:
        st.error(f"Error reading the uploaded file: {str(e)}")
        logger.error(f"File reading error: {str(e)}", exc_info=True)
        st.stop()
elif 'df' in st.session_state and 'file_hash' in st.session_state:
    # Use the dataframe and collection from session state
    df = st.session_state.df
    try:
        collection = client.get_collection(COLL_NAME)
    except Exception as e:
        st.error("Error loading existing embeddings. Please re-upload your file.")
        logger.error(f"Error loading collection: {str(e)}", exc_info=True)
        st.stop()

# Only show prompt input if we have a valid collection
if collection is not None and df is not None:
    # Display data preview
    st.subheader("Data Preview (First 5 Rows)")
    st.dataframe(df.head(5), use_container_width=True)
    
    # Display embedding status
    if 'file_hash' in st.session_state:
        st.sidebar.info(f"📊 Using embeddings for: {st.session_state.get('last_uploaded_file', 'Current file')}")
    
    user_prompt = st.text_input("Enter your query or request for analysis:")
    
    if st.button("Generate Charts") and user_prompt:
        try:
            # Retrieve context using existing embeddings
            result = collection.query(
                query_texts=[user_prompt],
                n_results=5,
                include=["documents"]
            )
            context = "\n".join(result['documents'][0])
            
            # Use the dataframe from session state
            df = st.session_state.df
            
            # Configure the model
            cfg = st.session_state.api_config['gemini']
            generation_config = {
                'temperature': cfg['temperature'],
                'max_output_tokens': min(cfg['max_tokens'], 4000),
                'top_p': cfg['top_p']
            }
            
            system_msg = """You are a data visualization expert that creates beautiful, insightful charts using Chart.js.

IMPORTANT FORMATTING RULES:
1. Your response MUST be a valid JSON array of chart configurations
2. Each chart configuration MUST include:
   - type: The chart type (bar, line, pie, doughnut, radar, bubble)
   - data: With labels array and datasets array
   - options: Including title with display:true and text
3. DO NOT include any markdown code block markers (```json or ```)
4. DO NOT include any explanatory text outside the JSON
5. For bar/line charts, always include beginAtZero: true in yAxes ticks
6. Include appropriate colors using rgba() format

EXAMPLE RESPONSE FORMAT:
[
  {
    "type": "pie",
    "data": {
      "labels": ["Label 1", "Label 2"],
      "datasets": [{
        "label": "Dataset 1",
        "data": [10, 20],
        "backgroundColor": [
          "rgba(255, 99, 132, 0.2)",
          "rgba(54, 162, 235, 0.2)"
        ]
      }]
    },
    "options": {
      "title": {
        "display": true,
        "text": "Chart Title"
      }
    }
  }
]

IMPORTANT: Your response must be valid JSON that can be directly parsed. Do not include any text outside the JSON array."""
            
            # Get sample data to help with chart generation
            sample_data = df.head().to_dict(orient='records')
            
            user_msg = (f"User query: {user_prompt}\n"
                      f"Available columns: {list(df.columns)}\n"
                      f"Sample data (first 5 rows): {json.dumps(sample_data, default=str)}\n"
                      f"Context from similar queries: {context}\n\n"
                      "Generate 2-4 different visualizations that best represent this data. "
                      "For each visualization, use the most appropriate chart type based on the data. "
                      "Include titles and labels that make the visualizations self-explanatory.")
            
            with st.spinner('Generating visualization...'):
                model = genai.GenerativeModel(
                    model_name=cfg['model'],
                    generation_config=generation_config,
                    system_instruction=system_msg
                )
                
                response = model.generate_content(user_msg)
                chart_json = response.text
                
                st.subheader("Visualized Charts")
                
                # Debug: Show raw response
                with st.expander("Debug: Raw LLM Response"):
                    st.code(chart_json, language="json")
                
                # Check for empty response
                if not chart_json.strip():
                    st.warning("The model returned an empty response. Please try rephrasing your query.")
                    st.stop()
                    
                # Try to clean and parse the response
                try:
                    # Clean the response (in case it's wrapped in markdown code blocks)
                    clean_json = chart_json.strip()
                    if '```json' in clean_json:
                        clean_json = clean_json.split('```json')[1].split('```')[0].strip()
                    elif '```' in clean_json:
                        clean_json = clean_json.split('```')[1].strip()
                        
                    # Try to parse as JSON
                    config = json.loads(clean_json)
                    
                    # Show JSON preview in expander
                    with st.expander(f"🔧 Chart Configuration (Preview)"):
                        st.json(config)
                        
                    # Validate required fields
                    if not isinstance(config, (dict, list)):
                        st.error("Invalid chart configuration: expected an object or array of objects")
                        st.stop()
                        
                    if isinstance(config, list):
                        for i, cfg in enumerate(config):
                            if not isinstance(cfg, dict) or 'type' not in cfg or 'data' not in cfg:
                                st.error(f"Invalid chart configuration at index {i}: missing required fields 'type' or 'data'")
                                st.json(cfg)
                                st.stop()
                    elif isinstance(config, dict):
                        if 'type' not in config or 'data' not in config:
                            st.error("Invalid chart configuration: missing required fields 'type' or 'data'")
                            st.json(config)
                            st.stop()
                    
                    # Render the chart
                    render_chartjs(clean_json)
                    
                except json.JSONDecodeError as je:
                    st.error("Failed to parse the chart configuration as JSON.")
                    st.error(f"JSON Error: {str(je)}")
                    st.text("Raw response:")
                    st.code(chart_json)
                except Exception as e:
                    st.error(f"Error processing chart configuration: {str(e)}")
                    st.exception(e)
                    
        except Exception as e:
            st.error(f"Error generating visualization: {str(e)}")
            logger.error(f"Error in visualization generation: {str(e)}", exc_info=True)
            
        except Exception as e:
            st.error(f"Error calling Gemini API: {str(e)}")
            st.error("Please try again or check your API configuration.")
            logger.error(f"API Error: {str(e)}", exc_info=True)
            st.stop()