import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """
    Load the campaign data set and perform base validation
    Parameters:
        file_path: data file path (should be tab-separated CSV)
    Back:
        pd.DataFrame (on success) or None (on failure)
    """
    REQUIRED_COLUMNS = [
        'ID', 'Year_Birth', 'Income', 'Dt_Customer', 'Recency',
        'MntWines', 'MntFruits', 'MntMeatProducts'
    ]
    
    try:
        df = pd.read_csv(
            file_path,
            sep='\t',
            encoding='utf-8',
            parse_dates=['Dt_Customer'],
            infer_datetime_format=True
        )
        logger.info(f"Data is loaded successfully. Dimensions: {df.shape}")

        if not _validate_data(df, REQUIRED_COLUMNS):
            logger.error("Data verification failed. Please check the data quality")
            return None
            
        return df

    except FileNotFoundError:
        logger.error(f"file not found: {file_path}")
    except pd.errors.ParserError:
        logger.error("File parsing failed. Please check whether the file is tab-separated CSV")
    except Exception as e:
        logger.error(f"unknown error: {str(e)}")
    
    return None

def _validate_data(df, required_columns):
    """
    Perform basic data quality checks
    Return: bool (True indicates that the authentication succeeds)
    """
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required fields: {missing_cols}")
        return False

    if df[['ID', 'Income', 'Dt_Customer']].isnull().any().any():
        logger.warning("Key fields have empty values: ID/Income/Dt_Customer")

    if df['ID'].duplicated().any():
        logger.error("There are duplicate ID values")
        return False

    if (df['Income'] <= 0).any():
        logger.warning("Income contains non-positive values")

    min_date = df['Dt_Customer'].min()
    max_date = df['Dt_Customer'].max()
    if min_date < datetime(2010,1,1) or max_date > datetime.now():
        logger.warning(f"The date is beyond the expected range: {min_date} to {max_date}")

    return True

if __name__ == "__main__":
    data = load_data("dataset/marketing_campaign.csv")
    
    if data is not None:
        print("Example of the first 5 rows of data:")
        print(data.head())

