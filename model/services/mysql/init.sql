CREATE DATABASE IF NOT EXISTS insurance_db;
USE insurance_db;

-- Raw data table (stores both training and prediction results)
CREATE TABLE IF NOT EXISTS insurance_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    
    -- Original columns
    SEX INT,
    INSR_BEGIN VARCHAR(10),
    INSR_END VARCHAR(10),
    EFFECTIVE_YR VARCHAR(5),
    INSR_TYPE INT,
    INSURED_VALUE FLOAT,
    OBJECT_ID VARCHAR(10),
    PROD_YEAR INT,
    SEATS_NUM FLOAT,
    CARRYING_CAPACITY FLOAT,
    TYPE_VEHICLE VARCHAR(100),
    CCM_TON FLOAT,
    MAKE VARCHAR(50),
    `USAGE` VARCHAR(100),
    CLAIM_PAID FLOAT,
    
    -- Premium (present in training data, NULL in prediction data)
    PREMIUM FLOAT,
    
    -- Prediction (added by model, NULL in training data)
    PREDICTED_PREMIUM FLOAT,
    
    -- Metadata
    source_topic VARCHAR(10),           -- 'train' or 'validate'
    model_version VARCHAR(50),          -- Which model made the prediction
    ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes for querying
    INDEX idx_object_id (OBJECT_ID),
    INDEX idx_ingestion_time (ingestion_time),
    INDEX idx_source (source_topic),
    INDEX idx_model_version (model_version)
);

-- Quality metrics table (for dashboard)
CREATE TABLE IF NOT EXISTS quality_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    metric_name VARCHAR(100),
    metric_value FLOAT,
    metric_details JSON,
    recorded_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
