#### Overview
This Python script is designed to process radiology reports using PySpark and extract key information related to lung nodules. It employs regular expressions (regex) to identify and extract relevant details such as size, location, descriptors, progression, and follow-up recommendations. The script is intended for use in a distributed computing environment like Azure, where it can be integrated into larger data pipelines for medical text analysis.

#### Features
Regex Extraction: Utilizes advanced regex patterns to identify key terms and phrases related to lung nodules in radiology reports.
Negation Handling: Includes a basic negation filter to improve the accuracy of term matching.
Lung-RADS Classification: Extracts and standardizes Lung-RADS categories from the reports, enabling further analysis.
Flagging System: Generates flags indicating whether matches were found in the impression text, report text, or both.
Data Integration: Combines extracted information into a structured format, suitable for downstream processing.

#### Dependencies
Python 3.x
PySpark
Azure ML SDK
Pandas
Regular Expressions (re module)

#### Installation
Install Python 3.x and ensure it is added to your system's PATH.
Install PySpark and Azure ML SDK via pip:
sh
Copy code
pip install pyspark azureml-sdk pandas
Ensure you have access to an Azure ML Workspace and the necessary credentials to connect.

#### Usage
1. Setup
Before running the script, ensure you have access to a Spark session. This script is intended to be run in a distributed environment, so make sure your Spark session is configured correctly.

2. Running the Script
Input: The script expects a Spark DataFrame containing radiology reports with columns for ImpressionText, ReportText, and diagnostic codes.
Execution: Call the main function run_step_2(df) with your input DataFrame. The function will return a new DataFrame with extracted information.

3. Key Functions
find_match_index(): Splits input text into phrases and applies regex patterns to extract matches and their indices.
extract_most_serious_lungrad(): Identifies the most serious Lung-RADS classification from extracted categories.
create_flags(): Generates flags based on the presence of specific terms in the text.
run_step_2(): Main function that orchestrates the extraction process and combines results.

4. Output
The script returns a Spark DataFrame with additional columns for each category of extracted information (e.g., size, location, descriptors) and corresponding flags. These columns can be used for further analysis or machine learning applications.

#### Improvements
The script includes various improvements over earlier versions, such as enhanced regex patterns for better accuracy, additional terms for progression, and a basic negation filter. Future improvements may include more sophisticated negation handling and expanded regex coverage.
