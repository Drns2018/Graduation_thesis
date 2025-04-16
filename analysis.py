import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import logging
from typing import Tuple, Dict, Union


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def segment_customers(df: pd.DataFrame, n_clusters: int = 4) -> Tuple[pd.DataFrame, Dict]:
    """
    Customer segmentation based on RFM model and consumption behavior
    Back:
        - DataFrame with the Segment label
        - Dictionary containing clustering evaluation metrics

    """
    try:
        features = df[[
            'Income',
            'Recency',
            'Total_Spending',
            'Family_Size',
            'Activity_Ratio'
        ]].dropna()
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Segment'] = kmeans.fit_predict(scaled_features)
        
        silhouette_avg = silhouette_score(scaled_features, df['Segment'])
        cluster_metrics = {
            'silhouette_score': silhouette_avg,
            'cluster_sizes': df['Segment'].value_counts().to_dict(),
            'features_used': features.columns.tolist()
        }
        
        logger.info(f"✅ Customer segmentation completed (profile factor: {silhouette_avg:.2f})")
        return df, cluster_metrics
        
    except Exception as e:
        logger.error(f"Cluster analysis failed: {str(e)}")
        return df, {}
def analyze_products(df: pd.DataFrame) -> Dict[str, Union[pd.DataFrame, dict]]:
    """
    Multi-dimensional Product Analysis 
    Returns a dictionary containing the following:
    - segment_demand: indicates the average value of product demand for each segment (DataFrame).
    -total_demand: total product demand (DataFrame)
    -product_correlation: Product correlation matrix (DataFrame)
    - channel_distribution: Purchase channel distribution (DataFrame)
    Parameters:
        df: Must include the 'Segment' column and the product amount column (beginning with Mnt)
    Back:
        Dictionary of structured analysis results
    """
    results = {}
    try:
        product_cols = [
            c for c in df.columns 
            if c.startswith('Mnt') 
            and pd.api.types.is_numeric_dtype(df[c])
        ]
    
        if product_cols:
            segment_demand = df.groupby('Segment')[product_cols].mean()
            results['segment_demand'] = segment_demand.reset_index()
        else:
            logger.warning("Product analysis is skipped because there are no valid numeric product columns")
            
        total_demand = df[product_cols].sum().reset_index()
        total_demand.columns = ['Product', 'Total_Demand']
        results['total_demand'] = total_demand.sort_values('Total_Demand', ascending=False)

        if 'Segment' in df.columns:
            segment_demand = df.groupby('Segment')[product_cols].mean().reset_index()
            
            segment_demand['Top_Product'] = segment_demand[product_cols].idxmax(axis=1)
            results['segment_demand'] = segment_demand
        else:
            logger.warning("Missing 'Segment' column, skipping group analysis")
        
        corr_matrix = df[product_cols].corr().round(2)
        results['product_correlation'] = corr_matrix
        
        if all(col in df.columns for col in ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']):
            channel_data = df[['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']].sum().reset_index()
            channel_data.columns = ['Channel', 'Total_Purchases']
            results['channel_distribution'] = channel_data
        else:
            logger.warning("Lack of purchase channel data, skip channel analysis")
            
        logger.info(f"The product analysis is complete and the {len(results)} item results are generated")
        return results
        
    except Exception as e:
        logger.error(f"Product analysis failure: {str(e)}")
        return {
            'error': str(e),
            'available_columns': list(df.columns) if df is not None else []
        }
        
def forecast_demand(
    df: pd.DataFrame, 
    product: str = 'MntWines',
    periods: int = 6
) -> Tuple[pd.DataFrame, Dict]:
    """
    Product demand forecasting (two methods supported)
    Back:
    - Forecast result DataFrame
    - Dictionary containing evaluation metrics
    """
    
    if df[product].var() < 1e-6:
        raise ValueError("Product demand data fluctuates too little")
    try:
        ts = df.groupby('Dt_Customer')[product].sum().reset_index()
        ts.columns = ['ds', 'y']
        
        model = Prophet(seasonality_mode='multiplicative')
        model.fit(ts)
        future = model.make_future_dataframe(periods=periods, freq='M')
        forecast = model.predict(future)
        val_months = min(12, len(ts))
        if val_months > 3:
            y_true = ts['y'][-val_months:].values
            y_pred = forecast['yhat'][-val_months:].values
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        else:
            mape = np.nan

        last_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        
        metrics = {
            'model': 'Prophet',
            'mape': mape,
            'last_actual': ts['y'].iloc[-1] if len(ts) > 0 else np.nan
        }
        
        logger.info(f"✅ {product}Demand forecast completed (MAPE: {mape:.1f}%)")
        forecast = model.predict(future)
        
        return last_forecast[['ds', 'yhat']], metrics  

        
    except Exception as e:
        logger.error(f"Demand forecasting failure: {str(e)}")
        return pd.DataFrame(), {}

if __name__ == "__main__":
    test_data = pd.DataFrame({
        'Income': np.random.normal(50000, 15000, 100),
        'Recency': np.random.randint(1, 100, 100),
        'Total_Spending': np.random.uniform(100, 2000, 100),
        'Family_Size': np.random.randint(1, 5, 100),
        'Activity_Ratio': np.random.uniform(0.1, 2.5, 100),
        'Dt_Customer': pd.date_range('2020-01-01', periods=100, freq='M'),
        'MntWines': np.random.poisson(50, 100)
    })
    
    segmented_df, metrics = segment_customers(test_data)
    print("Clustering Results:\n", segmented_df['Segment'].value_counts())
    product_results = analyze_products(segmented_df)
    print("\nTotal product demand:\n", product_results['total_demand'])
    forecast, forecast_metrics = forecast_demand(test_data)
    print("\nPredicting Outcomes:\n", forecast.head())
    
    

