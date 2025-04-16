import os
import logging
import argparse
import pandas as pd
from datetime import datetime
from data_processing import load_data
from cleaning import clean_data
from analysis import segment_customers, analyze_products, forecast_demand
from reporting import (
    plot_customer_segments,
    plot_product_demand,
    plot_purchase_channels,
    generate_ai_report,
    save_report_as_pdf
)
###

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", 
                       choices=["default", "ablation"],
                       default="default",
                       help="Operation mode: default- conventional analysis, ablation-ablation experiment")
    return parser.parse_args()
###

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def create_output_dir():
    """ Create output directory structure """
    os.makedirs("output/figures", exist_ok=True)
    os.makedirs("output/tables", exist_ok=True)

def save_analysis_results(results: dict, prefix: str = ""):
    """Save analysis results to CSV"""
    for name, data in results.items():
        if isinstance(data, pd.DataFrame):
            path = f"output/tables/{prefix}{name}.csv"
            data.to_csv(path, index=False)
            logger.info(f"==== Save analysis results ====: {path}")

def run_analysis(data_path: str):
    try:
        # ===== Initialization =====
        create_output_dir()
        start_time = datetime.now()
        logger.info(f"==== Start the analysis process ==== ({start_time.strftime('%Y-%m-%d %H:%M')})")

        # ===== Data processing =====
        logger.info("==== Load raw data ====")
        raw_df = load_data(data_path)
        if raw_df is None:
            raise ValueError("==== ❌ Data loading failure ====")

        logger.info("==== Cleaning data ====")
        cleaned_df = clean_data(raw_df)
        if cleaned_df is None:
            raise ValueError("==== ❌ Data cleaning failure ====")

        # ===== Analysis process =====
        logger.info("==== Customer segmentation analysis and Product demand analysis ====")
        segmented_df, seg_metrics = segment_customers(cleaned_df)
        save_analysis_results({"customer_segments": segmented_df}, "seg_")

        product_results = analyze_products(segmented_df)
        save_analysis_results(product_results, "prod_")

        logger.info("==== Demand forecasting ====")
        forecast, forecast_metrics = forecast_demand(
            segmented_df, 
            product='MntWines',
            periods=6
        )
        save_analysis_results({"wine_forecast": forecast}, "fcst_")

        # ===== Visual generation =====
        logger.info("==== Generate visual charts ====")
        fig_paths = [
            "output/figures/customer_segments.png",
            "output/figures/product_demand.png",
            "output/figures/purchase_channels.png"
        ]
        
        plot_customer_segments(segmented_df, fig_paths[0])
        plot_product_demand(product_results['segment_demand'], fig_paths[1])
        
        if 'channel_distribution' in product_results:
            plot_purchase_channels(
                product_results['channel_distribution'], 
                fig_paths[2]
            )

        logger.info("==== Generate analysis report ====")
        report_text = generate_ai_report(
            segmented_df,
            product_results,
            forecast_metrics
        )
        forecast_data = forecast.to_dict('records') if forecast is not None else {}
        report_text = generate_ai_report(
            segmented_df,
            product_results,
            {
                'last_forecast': {
                    'yhat_max': forecast['yhat'].max() if not forecast.empty else None,
                    'period': forecast['ds'].max() if not forecast.empty else None
                }
            }
        )
        
        save_report_as_pdf(
            report_text=report_text,
            charts=fig_paths,
            filename="output/marketing_analysis_report.pdf"
        )

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ The analysis process is complete, taking {duration:.2f} seconds")
        logger.info(f"==== The results are saved in: {os.path.abspath('output')}====")

    except Exception as e:
        logger.error(f"❌ Analysis flow interruption: {str(e)}")
        raise
###
def run_ablation_experiments(data_path):
    df = load_data(data_path)
    df = clean_data(df)
    
    experiments = {
        "baseline": {"segment": True, "full_features": True, "prophet": True},
        "groupA": {"segment": False, "full_features": True, "prophet": True},
        "groupB": {"segment": True, "full_features": False, "prophet": True},
        "groupC": {"segment": True, "full_features": True, "prophet": False}
    }
    
    results = {}
    for name, config in experiments.items():
        try:
            if config["segment"]:
                df_seg, _ = segment_customers(df)
            else:
                df_seg = df.copy()
                df_seg["Segment"] = pd.qcut(df["Income"], q=4, labels=False)

            if config["prophet"]:
                forecast, metrics = forecast_demand(df_seg)
                forecast_values = forecast['yhat'].tolist() if not forecast.empty else []
            else:
                forecast_values = df_seg["MntWines"].rolling(3).mean().dropna().iloc[-6:].tolist()
            
            results[name] = {
                "segmentation": df_seg["Segment"].nunique(),
                "forecast_values": forecast_values,
                "mape": metrics.get('mape', None) if config["prophet"] else None
            }
            
        except Exception as e:
            results[name] = {"error": str(e)}
    
    pd.DataFrame(results).T.to_csv("output/ablation_results.csv")
    return results
###


if __name__ == "__main__":
    args = parse_args()
    DATA_PATH = "dataset/marketing_campaign.csv"
    
    if args.mode == "ablation":
        run_ablation_experiments(DATA_PATH)
    else:
        run_analysis(DATA_PATH)  

