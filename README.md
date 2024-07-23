# Report Generator

This library generates reports based on scientific literature queries using Scopus, Cohere, and Together AI. You can use the query to automatically extract data from Scopus, or provide a csv downloaded from Scopus.

## Installation

You can install the Report Generator directly from GitHub:

```
pip install git+https://github.com/DataGomes/ReportAI

## API Keys

Before using the library, you need to set up your API keys. You can do this in two ways:

1. Set environment variables:
   - CO_API_KEY
   - TOGETHER_API_KEY
   - Pybliometrics will ask to set the api key using Scopus.
2. Use the `set_api_keys()` function:

```python
from report_generator import set_api_keys

set_api_keys(
    cohere_key="your_cohere_key",
    together_key="your_together_key"
)
```

Ensure you have the necessary permissions and subscriptions for Scopus API, Cohere, and Together AI.

## Usage

### As a library

```python
from report_generator import ReportAI, set_api_keys

# Set your API keys
set_api_keys(
    cohere_key="your cohere key",
    together_key="your together key"
)

# Create a ReportAI instance
generator = ReportAI()

# Generate a report
query = "artificial intelligence in healthcare and genetics"
html_output, query, html_pdf = generator.run_report(query, output_dir="your output directory")

if html_pdf == "failure":
    print(f"Error: {html_output}")
    
else:
    print("Report generated successfully.")
    print("\nSummary of results:")
    print(html_output)
```
If you want to use a csv downloaded from Scopus change html_pdf = generator.run_report(query, output_dir="your output directory", csv_path = "your csv path")
