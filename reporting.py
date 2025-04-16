import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from fpdf import FPDF
import dashscope
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from datetime import datetime

# Initializes the common sense query API
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

def plot_customer_segments(segmented_df, output_path):
    """
    Draw a comparison chart of customer segmentation group characteristics
    Includes: income distribution, age distribution, total consumption distribution
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.boxplot(x='Segment', y='Income', data=segmented_df)
    plt.title('Income Distribution by Segment')
    plt.xticks(rotation=45)
    plt.subplot(1, 3, 2)
    sns.violinplot(x='Segment', y='Age', data=segmented_df)
    plt.title('Age Distribution by Segment')
    plt.subplot(1, 3, 3)
    sns.barplot(x='Segment', y='Total_Spending', 
                data=segmented_df.groupby('Segment')['Total_Spending'].mean().reset_index())
    plt.title('Average Spending by Segment')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Customer breakdown map saved to: {output_path}")

def plot_product_demand(product_demand, output_path):

    heatmap_data = product_demand.set_index('Segment')
    product_cols = [col for col in heatmap_data.columns if col.startswith('Mnt')]
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data[product_cols], 
                annot=True, fmt=".0f", 
                cmap="YlGnBu", 
                linewidths=.5)
    plt.title('Product Demand Across Segments')
    plt.xlabel('Products')
    plt.ylabel('Customer Segment')
    plt.xticks(rotation=45)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Product demand heat map has been saved to: {output_path}")

def plot_purchase_channels(channel_data, output_path):
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Channel', y='Total_Purchases', 
                data=channel_data.sort_values('Total_Purchases', ascending=False))
    plt.title('Purchase Channel Distribution')
    plt.ylabel('Total Purchases')
    plt.xticks(rotation=30)
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Purchase channel distribution map has been saved to: {output_path}")

def clean_report_text(raw_report):
    patterns = [
        (r'^\s*#+\s*', ''),       
        (r'\*{2,}', ''),           
        (r'^-{3,}$', '\n'),        
        (r'^\s*[-*+]\s+', '  • '), 
    ]
    cleaned = raw_report
    for pattern, repl in patterns:
        cleaned = re.sub(pattern, repl, cleaned, flags=re.MULTILINE)
    return cleaned
def generate_ai_report(segmented_df, product_results, forecast_metrics):
    """
    Generate intelligent analysis reports with Tongyi
    Returns formatted report text
    """
    try:
        segment_summary = segmented_df.groupby('Segment').agg({
            'Income': 'mean',
            'Age': 'mean',
            'Total_Spending': 'sum'
        }).to_markdown()
        
        if 'segment_demand' in product_results:
            demand_df = product_results['segment_demand']

            numeric_cols = demand_df.select_dtypes(include=['number']).columns.tolist()
            if 'Segment' in demand_df.columns:
                numeric_cols = [c for c in numeric_cols if c != 'Segment']  
            
            if numeric_cols:
                top_products = demand_df.set_index('Segment')[numeric_cols].idxmax(axis=1)
                products_str = top_products.to_markdown()
        

        
        prompt = f"""
        Please generate a professional market report based on the following data analysis results:
        
        === Customer segmentation analysis ===
        {segment_summary}
        
        === Key point ===
        1. Highest income group: Segment {segmented_df.groupby('Segment')['Income'].mean().idxmax()}
        2. Youngest group: Segment {segmented_df.groupby('Segment')['Age'].mean().idxmin()}
        3. Groups prefer products:
           {products_str}
        
        === Demand forecasting ===
        Forecast peak sales in the next six months: {forecast_metrics.get('last_forecast', {}).get('yhat_max', 'N/A')}
        
        Reporting requirements:
        1. In English, it is divided into two parts: "Core findings" and "strategic recommendations"
        2. Contain specific data references
        3. Give actionable marketing suggestions
        """
        
        response = dashscope.Generation.call(
            model="qwen-turbo",
            #model="qwen-plus",
            prompt=prompt,
            temperature=0.3
        )
        
        if response.status_code == 200:
            raw_report = response.output['text']
            return clean_report_text(raw_report) 
        return "Report generation failed. Please check API configuration"
            
    except Exception as e:
        print(f"❌ An error occurred while generating the report: {str(e)}")
        return f"Report generation exception: {str(e)}"
def save_report_as_pdf(report_text, charts, filename):
    """
    Save text reports and charts as PDFS
    :param charts: Chart path list [segment_path, product_path, channel_path]
    """
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Marketing Analysis Report', 0, 1, 'C')
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    
    for line in report_text.split('\n'):
        #pdf.multi_cell(0, 5, txt=line.encode('latin-1', 'replace').decode('latin-1'))
        if line.strip().startswith('Top Product:'):  
            pdf.set_font('', 'B')
            pdf.multi_cell(0, 5, line)
            pdf.set_font('', '')
        elif line.startswith('  •'):
            pdf.cell(10)
            pdf.multi_cell(0, 5, line[3:])
        else:
            pdf.multi_cell(0, 5, line)
    for chart_path in charts:
        if os.path.exists(chart_path):
            pdf.add_page()
            pdf.image(chart_path, x=10, w=190)

    pdf.output(filename)
    print(f"✅ PDF report has been generated: {filename}")

###
def plot_ablation_results(results):
    """ Mapping ablation experiments """
    df = pd.DataFrame(results).T
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    for name, data in results.items():
        plt.plot(data["forecast_values"], label=name)
    plt.title("Forecast Comparison")
    plt.legend()

    plt.subplot(132)
    df["segmentation"].plot(kind="bar")
    plt.title("Segment Count")

    plt.subplot(133)
    df["features_used"].str.len().plot(kind="bar")
    plt.title("Features Used")
    
    plt.tight_layout()
    plt.savefig("output/ablation_comparison.png")
###



if __name__ == "__main__":
    test_data = pd.DataFrame({
        'Segment': [0,0,1,1],
        'Income': [50000, 60000, 80000, 75000],
        'Age': [35, 40, 28, 32],
        'Total_Spending': [1200, 1500, 2000, 1800]
    })
    
    plot_customer_segments(test_data, "test_segments.png")

    save_report_as_pdf(
        "Test report content",
        ["test_segments.png"],
        "test_report.pdf"
    )

