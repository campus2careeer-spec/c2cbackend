import pandas as pd

# Load Excel file
input_file = "raw/jharkhand_jobs.xlsx"
output_file = "jharkhand_jobs.csv"

# Read Excel file (default: first sheet)
df = pd.read_excel(input_file)

# Save as CSV
df.to_csv(output_file, index=False)

print("Conversion completed!")