import pandas as pd
def find_date_col(data):
    """
    This function finds the datetime column from the all columns
    and sets the datetime column as index
    Args:
        data: the input file
    Returns:
        date_col (str): name of the datetime column
        data: after the conversion from string to datetime object
    """
    date_col = []
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                data[col] = pd.to_datetime(data[col])
                date_col.append(col)
            except ValueError:
                pass

    df_datetime = data.select_dtypes(["datetime","datetimetz","timedelta"])

    if not date_col:
        date_col = list(df_datetime.columns)
    return date_col, data