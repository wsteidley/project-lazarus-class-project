import pandas as pd
from datetime import datetime, timezone
from response_format import funding_rounds
import json
import re

utc_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
print(utc_timestamp)

#-------------------
funding_rounds_list=list(map(lambda s:s.strip(),funding_rounds.split(',')))
def fix_round_name(current_name):
    for name in funding_rounds_list:
        if current_name and name.lower() in current_name.lower():
            return name
    return 'Unknown'

def check_round_names(funding_rounds):
    new_funding_rounds_list = []
    for funding_string in funding_rounds:
        funding_obj = json.loads(funding_string)
        new_funding_obj = []
        for funding in funding_obj:
            if not (funding.get("type") and funding.get("properties")):
                round_name = funding.get("round_name")
                if round_name not in funding_rounds_list:
                    round_name = fix_round_name(round_name)
                funding["round_name"] = round_name
                new_funding_obj.append(funding)
        new_funding_rounds_list.append(new_funding_obj)
    return new_funding_rounds_list

def standardize_date(date_string):
    """
    Parse various date formats into MM/YY format.
    
    Supported input formats include:
    - MM/DD/YYYY
    - MM-DD-YYYY
    - YYYY-MM
    - YY/MM
    - YY-MM
    - Full date strings
    - Datetime objects
    - Timestamps
    
    Args:
        date_string (str or datetime): Input date to be parsed
    
    Returns:
        str: Date in MM/YY format, or None if parsing fails
    """
    # Handle different input types
    if isinstance(date_string, datetime):
        return date_string.strftime("%m/%y")
    
    if not isinstance(date_string, str):
        try:
            # Convert to string if possible
            date_string = str(date_string)
        except:
            return None
    
    # Remove any leading/trailing whitespace
    date_string = date_string.strip()
    
    # List of parsing attempts with different patterns
    parsing_attempts = [
        # Direct MM/YY or MM/YYYY formats
        (r'^(\d{1,2})[/.-](\d{2}(?:\d{2})?)$', lambda m: f"{m.group(1).zfill(2)}/{m.group(2)[-2:]}"),
        
        (r'^(\d{4})$', lambda m: f"01/{m.group(1)[-2:]}"),

        # YYYY-MM format
        (r'^(\d{4})[/.-](\d{1,2})$', lambda m: f"{m.group(2).zfill(2)}/{m.group(1)[-2:]}"),
        
        # YY/MM or YY-MM format
        (r'^(\d{2})[/.-](\d{1,2})$', lambda m: f"{m.group(2).zfill(2)}/{m.group(1)}"),
        
        # MM/DD/YYYY or MM-DD-YYYY
        (r'^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})$', lambda m: f"{m.group(1).zfill(2)}/{m.group(3)[-2:]}"),
        
        # YYYY-MM-DD
        (r'^(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})$', lambda m: f"{m.group(2).zfill(2)}/{m.group(1)[-2:]}"),
        
        # Textual dates with comma
        (r'(\w+)\s+(\d{1,2}),\s*(\d{4})$', lambda m: datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%B %d %Y").strftime("%m/%y")),
        
        # Textual dates without comma
        (r'(\w+)\s+(\d{1,2})\s+(\d{4})$', lambda m: datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%B %d %Y").strftime("%m/%y"))
    ]
    
    # Try each parsing pattern
    for pattern, formatter in parsing_attempts:
        match = re.match(pattern, date_string, re.IGNORECASE)
        if match:
            try:
                return formatter(match)
            except:
                continue
    
    # If all parsing attempts fail, try datetime parsing as a last resort
    try:
        return datetime.fromisoformat(date_string).strftime("%m/%y")
    except Exception as e:
        print(f"Exception parsing date: {str(e)}")
        return None
    
def standardize_funding_data(funding_rounds):
    new_funding_rounds = check_round_names(funding_rounds)
    funding_round_dict = {}
    standardized_funding_rounds = []
    for funding_round_list in new_funding_rounds:
        for funding_round in funding_round_list:
            round_name = funding_round["round_name"]
            exiting_round_obj = funding_round_dict.get(round_name, {})
            if exiting_round_obj.get("currency_symbol") is None and funding_round.get("currency_symbol") is not None:
                exiting_round_obj["currency_symbol"] = funding_round["currency_symbol"]
            if exiting_round_obj.get("date") is None and funding_round.get("date") is not None:
                exiting_round_obj["date"] = standardize_date(funding_round["date"])
            if exiting_round_obj.get("amount") is None:
                exiting_round_obj["amount"] = 0
            if not isinstance(funding_round.get("amount"), (int, float)):
               funding_round["amount"] = 0
            exiting_round_obj["amount"] = int(max(funding_round.get("amount"), exiting_round_obj.get('amount')))
            exiting_round_obj["round_name"] = round_name
            if exiting_round_obj.get("date") is not None:
                funding_round_dict[round_name] = exiting_round_obj
    return list(funding_round_dict.values())
        # if not amount and funding_round.get('amount')
    
all_funding_rounds = []
def reconcile_rows(group):
    funding_rounds = group["funding_rounds"].tolist()
    standardized_funding_rounds = standardize_funding_data(funding_rounds)
    row = group.iloc[0]
    all_funding_rounds.append(standardized_funding_rounds)
    row["funding_rounds"] = str(standardized_funding_rounds)
    # Example handling different columns differently
    # result['numeric_col'] = group['numeric_col'].sum()  # Sum numbers
    # result['text_col'] = ' | '.join(group['text_col'].unique())  # Join unique text
    # result['latest_date'] = group['date_col'].max()  # Get latest date
    return row

def save_article_data_to_csv(data_frame: pd.DataFrame, filename: str) -> None:
    """Save processed data to CSV"""
    if not filename:
        raise ValueError('Need to provide a filename to save')
    data_frame.to_csv(filename, index=False, encoding='utf-8')
    print(f"Data enriched with funding rounds successfully saved to {filename}")

def plot_data(data):
    import matplotlib.pyplot as plt
    import plotly.express as px
    import pandas as pd

    # Your dict of years and amounts

    # Matplotlib version with y-axis in millions
    plt.figure(figsize=(10,6))
    plt.plot(list(sorted(data.keys())), 
            [x/1000000 for x in list(sorted(data.values()))], 
            marker='o')
    plt.title('Amount by Year')
    plt.xlabel('Year')
    plt.ylabel('Millions of Dollars')
    plt.show()

    # Plotly version with y-axis in millions
    df = pd.DataFrame.from_dict(data, orient='index', columns=['Amount'])
    df.index.name = 'Year'
    df = df.reset_index().sort_values('Year')
    df['Amount_Millions'] = df['Amount'] / 1000000

    import plotly.express as px
    fig = px.line(df, x='Year', y='Amount_Millions', 
                title='Amount by Year',
                labels={'Amount_Millions': 'Millions of Dollars'},
                markers=True)
    fig.show()

def main():
    filename = ''
    if not filename:
        raise ValueError("Filename should be output file from step2")
    utc_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    try:
        funding_data_df = pd.read_csv(filename)
        funding_data_df = funding_data_df
        funding_data_df = funding_data_df[funding_data_df['funding_rounds'].notna()]
        funding_data_df = funding_data_df.groupby('company_name').apply(reconcile_rows)
        funding_data_df = funding_data_df.drop(['json_schema', 'type'], axis=1)
        save_article_data_to_csv(funding_data_df, f"standardized_funding_{utc_timestamp}.csv")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    main()