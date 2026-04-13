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


def get_prediction_data(engine):
    query = """
        SELECT
            t.SEX,
            t.INSR_BEGIN,
            t.INSR_END,
            t.EFFECTIVE_YR,
            t.INSR_TYPE,
            t.INSURED_VALUE,
            t.OBJECT_ID,
            t.PROD_YEAR,
            t.SEATS_NUM,
            t.CARRYING_CAPACITY,
            t.TYPE_VEHICLE,
            t.CCM_TON,
            t.MAKE,
            t.USAGE,
            t.CLAIM_PAID,
            t.PREMIUM AS actual,
            p.PREDICTED_PREMIUM AS predicted,
            p.model_version,
            p.ingestion_time
        FROM insurance_records t
        JOIN insurance_records p
            ON t.OBJECT_ID = p.OBJECT_ID
            AND t.SEX = p.SEX
            AND t.INSR_TYPE = p.INSR_TYPE
            AND t.INSURED_VALUE = p.INSURED_VALUE
            AND t.TYPE_VEHICLE = p.TYPE_VEHICLE
            AND t.MAKE = p.MAKE
            AND t.USAGE = p.USAGE
            AND t.INSR_BEGIN = p.INSR_BEGIN
            AND t.INSR_END = p.INSR_END
        WHERE t.PREMIUM IS NOT NULL
            AND p.PREDICTED_PREMIUM IS NOT NULL
        ORDER BY p.ingestion_time DESC
        LIMIT 10000
    """
    df = pd.read_sql(query, engine)
    df['error'] = df['predicted'] - df['actual']
    return df


if 'mape_history' not in st.session_state:
    st.session_state.mape_history = []
    st.session_state.last_mape = None

if 'scores_history' not in st.session_state:
    st.session_state.scores_history = {
        'zero_init': [],
        'zero_cons': [],
        'else_init': [],
        'else_cons': []
    }
    st.session_state.last_params_hash = None

st.set_page_config(layout="wide")
st.title("Insurance Dashboard")

auto = st.sidebar.checkbox("Auto-refresh (5 min)", value=True)
if st.sidebar.button("Refresh Now"):
    st.rerun()

metrics = load_metrics()
engine = get_engine()
df_predictions = get_prediction_data(engine)
now = pd.Timestamp.now()

# update MAPE
current_mape = metrics.get('validation_mape')
if current_mape and current_mape != st.session_state.last_mape:
    st.session_state.mape_history.append({'timestamp': now, 'value': current_mape})
    st.session_state.last_mape = current_mape
    if len(st.session_state.mape_history) > 100:
        st.session_state.mape_history = st.session_state.mape_history[-100:]

# update RMSE
initial = metrics.get('initial_model', {})
consequent = metrics.get('consequent_model', {})

current_params_hash = hash((
    str(initial.get('INSR_ZERO', {}).get('best_params')),
    str(initial.get('ELSE', {}).get('best_params')),
    str(consequent.get('INSR_ZERO', {}).get('best_params')),
    str(consequent.get('ELSE', {}).get('best_params'))
))

if current_params_hash != st.session_state.last_params_hash:
    st.session_state.last_params_hash = current_params_hash

    for name, key in [('INSR_ZERO', 'zero_init'), ('ELSE', 'else_init')]:
        if name in initial:
            score = initial[name]['best_score']
            st.session_state.scores_history[key].append({'timestamp': now, 'value': score})
            if len(st.session_state.scores_history[key]) > 100:
                st.session_state.scores_history[key] = st.session_state.scores_history[key][-100:]

    for name, key in [('INSR_ZERO', 'zero_cons'), ('ELSE', 'else_cons')]:
        if name in consequent:
            score = consequent[name]['best_score']
            st.session_state.scores_history[key].append({'timestamp': now, 'value': score})
            if len(st.session_state.scores_history[key]) > 100:
                st.session_state.scores_history[key] = st.session_state.scores_history[key][-100:]

st.subheader("Validation MAPE Over Time")
if st.session_state.mape_history:
    df = pd.DataFrame(st.session_state.mape_history)
    df = df.set_index('timestamp')
    st.line_chart(df)
else:
    st.info("Waiting for MAPE data...")

st.subheader("Outliers Removed by Filter")
outliers = metrics.get('data_metrics', {}).get('outliers', {}).get('removed_by_filter', {})
if outliers:
    df = pd.DataFrame([{'filter': k, 'count': v} for k, v in outliers.items()])
    st.bar_chart(df.set_index('filter'))
else:
    st.info("No outlier data available")

st.subheader("Model RMSE Over Time")
scores = []
for key, name in [('zero_init', 'Zero Initial'), ('else_init', 'Else Initial'),
                  ('zero_cons', 'Zero Consequent'), ('else_cons', 'Else Consequent')]:
    for record in st.session_state.scores_history[key]:
        scores.append({'model': name, 'timestamp': record['timestamp'], 'score': record['value']})

if scores:
    df = pd.DataFrame(scores)
    pivot_df = df.pivot(index='timestamp', columns='model', values='score')
    st.line_chart(pivot_df)
else:
    st.info("Waiting for model scores...")

st.subheader("Prediction Error Distribution")
if not df_predictions.empty:
    hist, bins = np.histogram(df_predictions['error'], bins=50)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    df_hist = pd.DataFrame({'bin_center': bin_centers, 'count': hist})
    st.bar_chart(df_hist.set_index('bin_center'))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Error", f"{df_predictions['error'].mean():.0f}")
    with col2:
        st.metric("Median Error", f"{df_predictions['error'].median():.0f}")
    with col3:
        st.metric("Std Dev", f"{df_predictions['error'].std():.0f}")
else:
    st.info("No error data available")

st.subheader("Recent Predictions (Last 5)")
if not df_predictions.empty:
    st.dataframe(
        df_predictions.head(5),
        use_container_width=True
    )
else:
    st.info("No prediction data available")

if auto:
    time.sleep(300)
    st.rerun()
