phys_columns = ['vibration_g',
                'temperature_c',
                'power_factor']
target1 = 'machine_status'
nan_cols = ['temperature_c']
def drop_nans(df):
    print('строк до геноцида:', len(df))
    df.dropna(subset=nan_cols, inplace=True)
    print('неугодные ликвидированы. Строк после:', len(df))
def prepare_data_for_model1(df):
    drop_nans(df)
    X = df[phys_columns]
    y = df[target1]
    return X, y
