import numpy as np
import pandas as pd


def load_the_csvs(loc, data=['p1'], verbose=True):
    '''
    For loading in the Crunchbase CSV files for this project.
    Returns a list of dataframes in order of input CSV list.

    Parameters
    ------------
    loc: str
        Path to folder where the CSVs are
    
    data : list of str
        The names of CSV files to load in. There are default values 
        but any of the following are also accepted.
        
        INPUT Crunchbase CSVs:
        acquisitions       funds                    organizations
        category_groups    investment_partners      org_parents
        checksum           investments              p1
        degrees            investors                people
        event_appearances  ipos                     people_descriptions
        events             jobs
        funding_rounds     organization_descriptions
        
        OUTPUT Crunchbase CSVs for model:
        organizations_merged     p1_by_category_group
        p1_funding_rounds        organizations_by_category_group
        p1_investments           baseline               
        p1_investment_partners   baseline_impute_linear                
        p1_jobs                  baseline_impute_complete    
    
    verbose : Bool
        For printing to stdout.

    Return
    ------------
    dfs : list
        List of dataframes containing CSV data.
    '''
    dfs = []
    for csv_name in data:
        path = loc + csv_name + '.csv'
        if 'baseline' in csv_name:
            temp = reduce_memory_usage(pd.read_csv(path, sep=';'), verbose=False)
        else:
            temp = reduce_memory_usage(pd.read_csv(path), verbose=False)
        dfs.append(temp)
        if verbose:
            print(f'{path.upper()}')
            print(f'{csv_name.upper()} shape: {temp.shape}')
            print(f'{csv_name.upper()} columns: {temp.columns.to_list()}')
            print()
    if len(dfs) == 1:
        return dfs[0]
    return dfs

def column_formatter(df):
    '''
    Method to avoid copying and pasting the same lines of pandas code.
    '''
    for col in df.columns:
        # Convert boolean to binary
        if col=='p1_tag':
            df['p1_tag'] = df['p1_tag'].apply(lambda x: 1 if x == True else 0)
        # P1 datetime column
        if col=='p1_date':
            df['p1_date'] = pd.to_datetime(df['p1_date'])
        # Convert employee_count 'unknown' to np.NaN to get accurate missing value count
        if col=='employee_count':
            df['employee_count'] = df['employee_count'].apply(lambda x: np.NaN if x == 'unknown' else x)
        # CB datetime columns-- OutOfBoundsDatetime error if do not coerce for CB native timestamp columns 
        if col in ['founded_on','closed_on','announced_on','started_on','ended_on']:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def reduce_memory_usage(df, verbose=True):
    '''
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    
    Sourced from: https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
    '''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def reformat_large_tick_values(tick_val, pos):
    '''
    Turns large tick values (in the billions, millions and thousands) 
    such as 4500 into 4.5K and also appropriately turns 4000 into 4K 
    (no zero after the decimal).
    '''
    if tick_val >= 1000000000:
        val = round(tick_val/1000000000, 1)
        new_tick_format = '${:}B'.format(val)
    elif tick_val >= 1000000:
        val = round(tick_val/1000000, 1)
        new_tick_format = '${:}M'.format(val)
    elif tick_val >= 1000:
        val = round(tick_val/1000, 1)
        new_tick_format = '${:}K'.format(val)
    elif tick_val < 1000:
        new_tick_format = round(tick_val, 1)
    else:
        new_tick_format = tick_val

    # make new_tick_format into a string value
    new_tick_format = str(new_tick_format)
    
    # code below will keep 4.5M as is but change values such as 4.0M to 4M since that zero after the decimal isn't needed
    index_of_decimal = new_tick_format.find('.')
    
    if index_of_decimal != -1:
        value_after_decimal = new_tick_format[index_of_decimal+1]
        if value_after_decimal == '0':
            # remove the 0 after the decimal point since it's not needed
            new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal+2:]
            
    return new_tick_format