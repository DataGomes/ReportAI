import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the necessary components directly
from report_generator.report_generator import ReportGenerator
from report_generator.config import set_api_keys

def run_test():
    print(f"ReportGenerator module: {ReportGenerator.__module__}")
    print(f"set_api_keys module: {set_api_keys.__module__}")

    # Set your API keys
    set_api_keys(
        pybliometrics_key="your_pybliometrics_key",
        voyage_key="your_voyage_key",
        together_key="your_together_key"
    )

    # Create a ReportGenerator instance
    generator = ReportGenerator()

    # Generate a report
    query = "manufacturing management and artificial intelligence in business"
    output_dir = os.path.join(os.path.dirname(__file__), 'Results')
    html_output, query, html_pdf = generator.run_report(query, output_dir=output_dir)

    if html_pdf == "failure":
        print(f"Error: {html_output}")
    else:
        print("Report generated successfully.")
        print(f"HTML output: {html_output[:100]}...")  # Print the first 100 characters of the HTML output

if __name__ == "__main__":
    run_test()