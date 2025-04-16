import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def clean_data(df):
    """
    Run the full data cleaning process:
        1.Handling missing values
        2.Standardize categorical variables
        3.Fix the outliers
        Generate derived features
    Parameters:
        df: Original DataFrame
    Returns:
        The cleaned DataFrame
    """
    if df is None or df.empty:
        logger.error("Input data is empty")
        return None

    try:
        # ===== 1. Handle key fields =====
        # Ensure that the ID is unique
        df = df.drop_duplicates('ID', keep='first')
        
        # Date field processing
        df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')
        # Remove invalid date records
        df = df[df['Dt_Customer'].notna()]  

        # ===== 2. Missing value processing =====
        # Income filled with median (forward fill)
        df['Income'] = df['Income'].fillna(
            df['Income'].median()
        ).ffill()

        # Product consumption missing fill 0 (assuming no purchase)
        product_cols = [c for c in df.columns if c.startswith('Mnt')]
        df[product_cols] = df[product_cols].fillna(0)

        # ===== 3.Categorical variable standardization =====
        df['Education'] = df['Education'].replace({
            '2n Cycle': 'Undergraduate',
            'Graduation': 'Undergraduate',
            'Master': 'Postgraduate',
            'PhD': 'Postgraduate'
        })

        df['Marital_Status'] = df['Marital_Status'].replace({
            'Married': 'Partner',
            'Together': 'Partner',
            'Divorced': 'Single',
            'Widow': 'Single',
            'Alone': 'Single',
            'Absurd': 'Single',
            'YOLO': 'Single'
        })

        # ===== 4. Outlier processing =====
        # Income truncation (removal of up to 1%)
        # Age rationality Check (18-100 years old)
        income_upper = df['Income'].quantile(0.99)
        df = df[df['Income'] <= income_upper]

        df['Age'] = datetime.now().year - df['Year_Birth']
        df = df[(df['Age'] >= 18) & (df['Age'] <= 100)]

        # ===== 5. Derived feature generation =====
        df['Family_Size'] = df['Kidhome'] + df['Teenhome'] + 1
        df['Total_Spending'] = df[product_cols].sum(axis=1)
        
        df['Activity_Ratio'] = (
            df['NumWebPurchases'] / 
            (df['NumWebVisitsMonth'] + 1e-6)  # Avoid division by zero
        )

        logger.info("✅ Data cleaning complete")
        return df

    except Exception as e:
        logger.error(f"An error occurred during cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    test_data = pd.DataFrame({
        'ID': [1, 2, 3],
        'Year_Birth': [1980, 1990, 2005],
        'Income': [50000, np.nan, 1000000],
        'Dt_Customer': ['2020-01-01', 'invalid', '2021-05-15'],
        'Education': ['Graduation', 'PhD', '2n Cycle'],
        'MntWines': [100, 200, np.nan],
        'Kidhome': [1, 0, 2]
    })
    
    cleaned = clean_data(test_data)
    if cleaned is not None:
        print("\nThe data after cleaning is shown:")
        print(cleaned.head())
