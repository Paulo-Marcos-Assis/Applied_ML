import pandas as pd
import re
import unicodedata

def normalize_text(text):
    """
    Normalize text for comparison by:
    - Converting to lowercase
    - Removing accents
    - Removing extra whitespace
    - Removing special characters
    """
    if pd.isna(text):
        return ""
    
    # Convert to string
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove accents
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

# Load datasets
print("Loading datasets...")
test_df = pd.read_csv('/home/paulo/CascadeProjects/Applied_ML/test/dataset/TEST_CLEAN_NO_DUPLICATES.csv')
positive_df = pd.read_csv('/home/paulo/CascadeProjects/Applied_ML/dataset/cleaned_data_no_bias/POSITIVE_DF_COMPANIES_REDUZIDO.csv')

print(f"Test dataset shape: {test_df.shape}")
print(f"Positive dataset shape: {positive_df.shape}")

# Filter test dataset to only label == 1
print("\nFiltering test dataset for label == 1...")
test_filtered = test_df[test_df['label'] == 1].copy()
print(f"Filtered test dataset shape: {test_filtered.shape}")
print(f"Label distribution in test dataset:\n{test_df['label'].value_counts()}")

# Normalize text columns for comparison
print("\nNormalizing text columns...")
test_filtered['text_normalized'] = test_filtered['text'].apply(normalize_text)
positive_df['text_normalized'] = positive_df['text'].apply(normalize_text)

# Find matches
print("\nSearching for matches...")
matches = []

for idx, test_row in test_filtered.iterrows():
    test_text_norm = test_row['text_normalized']
    
    # Check if this text exists in positive_df
    match = positive_df[positive_df['text_normalized'] == test_text_norm]
    
    if not match.empty:
        # Get the first match (in case of duplicates)
        match_row = match.iloc[0]
        
        # Create new row with data from test dataset + empresa(s) and fraude from positive dataset
        new_row = {
            'title': test_row['title'],
            'text': test_row['text'],
            'label': test_row['label'],
            'empresa(s)': match_row['empresa(s)'],
            'fraude': match_row['fraude']
        }
        matches.append(new_row)

print(f"\nFound {len(matches)} matches!")

# Create new dataframe with matches
if matches:
    result_df = pd.DataFrame(matches)
    
    # Save to new CSV
    output_path = '/home/paulo/CascadeProjects/Applied_ML/test/dataset/TEST_CLEAN_NO_DUPLICATES_WITH_MATCHES.csv'
    result_df.to_csv(output_path, index=False)
    print(f"\nNew dataset saved to: {output_path}")
    print(f"Shape: {result_df.shape}")
    print(f"\nColumns: {list(result_df.columns)}")
    print(f"\nFirst few rows:")
    print(result_df.head())
else:
    print("\nNo matches found between the datasets.")
