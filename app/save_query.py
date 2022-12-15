
def save_query(env_dict):
    column = env_dict['Column Name']

    strings = []
    strings.append('#####################################')
    strings.append(f'#### {column} ')
    strings.append('#####################################')

    if env_dict['Column Type'] == 'Numerical':
        strings.append( f"data['{column}'] = data['{column}'].astype('float32')" )

        if env_dict['Missing'] == 'Fill Zero':
            strings.append( f"data['{column}'] = data['{column}'].fillna(0)" )
        elif env_dict['Missing'] == 'Fill Mean':
            strings.append( f"data['{column}'] = data['{column}'].fillna(data['{column}'].mean())" )

        if env_dict['Scale'] == 'Log Scale':
            strings.append( f"data['{column}'] = np.log(data['{column}'])" )

        if env_dict['Normalization'] == 'Standardization':
            strings.append( f"data['{column}'] = (data['{column}'] - data['{column}'].mean()) / data['{column}'].std()" )
        elif env_dict['Normalization'] == 'Normalization':
            strings.append( f"data['{column}'] = (data['{column}'] - data['{column}'].min()) / (data['{column}'].max() - data['{column}'].min())" )

        # if env_dict['Lambda x'] != '':
            # strings.append( f"data['{column}'] = data['{column}'].apply( lambda x: {env_dict['Lambda x']} )" )

    return '\n'.join(strings)
