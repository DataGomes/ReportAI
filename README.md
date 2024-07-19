# Report Generator

This library generates reports based on scientific literature queries using Scopus, Voyage AI, and Together AI.

## Installation

You can install the Report Generator directly from GitHub:

```
pip install git+https://github.com/DataGomes/ReportAI

## API Keys

Before using the library, you need to set up your API keys. You can do this in two ways:

1. Set environment variables:
   - VOYAGE_API_KEY
   - TOGETHER_API_KEY
   - Pybliometrics will guide to set the api key using Scopus.
2. Use the `set_api_keys()` function:

```python
from report_generator import set_api_keys

set_api_keys(
    pybliometrics_key="your_pybliometrics_key",
    voyage_key="your_voyage_key",
    together_key="your_together_key"
)
```

Ensure you have the necessary permissions and subscriptions for Scopus API, Voyage AI, and Together AI.

## Usage

### As a library

```python
from report_generator import ReportAI, set_api_keys

# Set your API keys
set_api_keys(
    voyage_key="your voyage key",
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


### As a command-line tool

After installation, you can use the `generate_report` command:

```
generate_report "cars manufacturing management artificial intelligence" --output /path/to/o```
