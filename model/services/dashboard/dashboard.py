import streamlit as st
import pandas as pd
import numpy as np
import sys
import json
import time
import logging
from sqlalchemy import create_engine

sys.path.append('/app')
from services.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_engine():
    return create_engine(
        f"mysql+mysqlconnector://{Config.MYSQL_USER}:{Config.MYSQL_PASSWORD}@"
        f"{Config.MYSQL_HOST}:{Config.MYSQL_PORT}/{Config.MYSQL_DATABASE}"
    )


def load_metrics():
    path = f"{Config.ACTIVE_MODELS_PATH}/{Config.PREDICTOR_STATE_FILENAME}"
    try:
        with open(path) as f:
            data = json.load(f)
            return data.get('metrics', {})
    except Exception as e:
        logger.error(f"Failed to load metrics: {e}")
        return {}


def get_errors(engine):
    query = """
        SELECT t.PREMIUM - p.PREDICTED_PREMIUM as error
        FROM insurance_records t
        JOIN insurance_records p
            ON t.OBJECT_ID = p.OBJECT_ID
            AND t.TYPE_VEHICLE = p.TYPE_VEHICLE
            AND t.MAKE = p.MAKE
            AND t.USAGE = p.USAGE
            AND t.PROD_YEAR = p.PROD_YEAR
        WHERE t.PREMIUM IS NOT NULL
            AND p.PREDICTED_PREMIUM IS NOT NULL
        LIMIT 10000
    """
    return pd.read_sql(query, engine)['error'].tolist()


# Initialize state
if 'history' not in st.session_state:
    st.session_state.history = {
        'mape': [],
        'zero_init': [],
        'zero_cons': [],
        'else_init': [],
        'else_cons': []
    }
    st.session_state.last_hash = None
    st.session_state.counter = 0

st.set_page_config(layout="wide")
st.title("Insurance Dashboard")

auto = st.sidebar.checkbox("Auto-refresh (5 min)", value=True)
if st.sidebar.button("Refresh Now"):
    st.rerun()

# Load data
metrics = load_metrics()
engine = get_engine()

# Check if metrics changed
current_hash = hash((
    metrics.get('validation_mape'),
    metrics.get('initial_model', {}).get('INSR_ZERO', {}).get('best_score'),
    metrics.get('initial_model', {}).get('ELSE', {}).get('best_score'),
    metrics.get('consequent_model', {}).get('INSR_ZERO', {}).get('best_score'),
    metrics.get('consequent_model', {}).get('ELSE', {}).get('best_score')
))

if current_hash != st.session_state.last_hash:
    st.session_state.last_hash = current_hash
    st.session_state.counter += 1
    idx = st.session_state.counter
    
    mape = metrics.get('validation_mape')
    if mape:
        st.session_state.history['mape'].append({'index': idx, 'value': mape})
    
    initial = metrics.get('initial_model', {})
    consequent = metrics.get('consequent_model', {})
    
    for name, key in [('INSR_ZERO', 'zero_init'), ('ELSE', 'else_init')]:
        if name in initial:
            st.session_state.history[key].append({'index': idx, 'value': initial[name]['best_score']})
    
    for name, key in [('INSR_ZERO', 'zero_cons'), ('ELSE', 'else_cons')]:
        if name in consequent:
            st.session_state.history[key].append({'index': idx, 'value': consequent[name]['best_score']})

# Plot 1: MAPE
st.subheader("Validation MAPE")
if st.session_state.history['mape']:
    df = pd.DataFrame(st.session_state.history['mape'])
    st.line_chart(df.set_index('index'))
else:
    st.info("Waiting for MAPE data...")

# Plot 2: Outliers Removed by Filter
st.subheader("Outliers Removed by Filter")
outliers = metrics.get('data_metrics', {}).get('outliers', {}).get('removed_by_filter', {})
if outliers:
    df = pd.DataFrame([{'filter': k, 'count': v} for k, v in outliers.items()])
    st.bar_chart(df.set_index('filter'))
else:
    st.info("No outlier data available")

# Plot 3: Model Scores (4 lines)
st.subheader("Model RMSE Over Time")
scores = []
for key, name in [('zero_init', 'Zero Initial'), ('else_init', 'Else Initial'),
                  ('zero_cons', 'Zero Consequent'), ('else_cons', 'Else Consequent')]:
    for record in st.session_state.history[key]:
        scores.append({'model': name, 'index': record['index'], 'score': record['value']})

if scores:
    df = pd.DataFrame(scores)
    pivot_df = df.pivot(index='index', columns='model', values='score')
    st.line_chart(pivot_df)
else:
    st.info("Waiting for model scores...")

# Plot 4: Error Histogram (using pandas bins)
st.subheader("Prediction Error Distribution")
errors = get_errors(engine)
if errors:
    # Create histogram bins using pandas
    series = pd.Series(errors)
    hist, bins = np.histogram(errors, bins=50)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    df_hist = pd.DataFrame({'bin_center': bin_centers, 'count': hist})
    st.bar_chart(df_hist.set_index('bin_center'))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Error", f"{np.mean(errors):.0f}")
    with col2:
        st.metric("Median Error", f"{np.median(errors):.0f}")
    with col3:
        st.metric("Std Dev", f"{np.std(errors):.0f}")
else:
    st.info("No error data available")

# Plot 5: Current values summary
st.subheader("Current Model Performance")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Initial Model (GB)**")
    initial = metrics.get('initial_model', {})
    for name in ['INSR_ZERO', 'ELSE']:
        if name in initial:
            st.metric(f"{name}", f"{initial[name]['best_score']:.0f}")
with col2:
    st.markdown("**Consequent Model (NN)**")
    consequent = metrics.get('consequent_model', {})
    for name in ['INSR_ZERO', 'ELSE']:
        if name in consequent:
            st.metric(f"{name}", f"{consequent[name]['best_score']:.0f}")

# Auto-refresh
if auto:
    time.sleep(300)
    st.rerun()