import pandas as pd
import json



def serialization_json_new(df, test=False):
    """
    JSON-format: {"0": {"feature1": value1, "feature2": value2}}
    """
  
    def process_row(row, index=None):
        row_dict = {}
        for col in row.index:
            value = row[col]

            if pd.api.types.is_integer(value):
                value = int(value)
            elif pd.api.types.is_float(value):
                value = float(value)
            elif pd.api.types.is_bool(value):
                value = bool(value)
                
            if col == 'label' and test:
                row_dict[col] = ' '
            else:
                row_dict[col] = value
        return {str(index if index is not None else 0): row_dict}

    if isinstance(df, (pd.Series, pd.DataFrame)):

        if isinstance(df, pd.Series):
            df = df.to_frame().T
            
        result = {}
        for idx, (_, row) in enumerate(df.iterrows()):
            result.update(process_row(row, idx))
            
        final = json.dumps(result, separators=(', ', ': '))
        if test:
            final = final[:-4]
        return final
    
    raise TypeError("Input must be pandas Series or DataFrame")


def serialization1_old(df, test=False):
    
    """
    Features are: age = 39, education = Bachelor, gain = 2174. Answer is 0.
    """

    def process_row(row):
        features = []
        for col in row.index:
            if col != 'label':
                value = row[col]
                if isinstance(value, float):
                    value = int(value)
                features.append(f"{col} = {value}")
        
        features_str = ", ".join(features)
        label = ' ' if test else row.get('label', 'unknown')
        return f"Features are: {features_str}. Answer is {label}."


    if isinstance(df, pd.Series):
        return process_row(df)
    
    prompt_lines = []
    for _, row in df.iterrows():
        prompt_lines.append(process_row(row))

    final = "\n".join(prompt_lines)
    if test:
        final = final[:-2]
    return final


def serialization2_old(df, test=False):
    
    """
    Features are: age is 39, education is Bachelor, gain is 2174. Answer is 0.
    """
    def process_row(row):
        features = []
        for col in row.index:
            if col != 'label':
                value = row[col]
                if isinstance(value, float):
                    value = int(value)
                features.append(f"the {col.replace('_', ' ')} is {value}")

        features_str = ", ".join(features)
        label = ' ' if test else row.get('label', 'unknown')
        return f"Features are: {features_str}. Answer is {label}."

    if isinstance(df, pd.Series):
        return process_row(df)
    
    prompt_lines = []
    for _, row in df.iterrows():
        prompt_lines.append(process_row(row))

    final = "\n".join(prompt_lines)

    if test:

        final = final[:-2]
    
    return final


def serialization3_old(df, test=False):
    
    """
    Features are: x_1 = 39, x_2 = Bachelor, x_3 = 2174. Answer is: y = 0. 
    """
    
    def process_row(row, columns):
        features = []
        for i, col in enumerate([c for c in columns if c != 'label'], start=1):
            value = row[col]
            if isinstance(value, float):
                value = int(value)
            features.append(f"x_{i} = {value}")

        features_str = ", ".join(features)
        label = ' ' if test else row.get('label', 'unknown')
        return f"Features are: {features_str}. Answer is: y = {label}."
    
    if isinstance(df, pd.Series):
        row_dict = df.to_dict()
        columns = df.index.tolist()
        return process_row(row_dict, columns)
    
    prompt_lines = []
    columns = [c for c in df.columns]
    for _, row in df.iterrows():
        prompt_lines.append(process_row(row, columns))

    final = "\n".join(prompt_lines)

    if test:

        final = final[:-2]
        
    return final


def serialization_datamatrix_new(df, test=False):
    
    """
    [['name', 'age'], ['helen', 47]]
    """
    
    if isinstance(df, pd.Series):
        header = list(df.index)
        row_values = list(df.values)
        
        if test and 'label' in df.index:
            label_pos = list(df.index).index('label')
            row_values[label_pos] = ' '

        final = str([header, row_values])
    
    elif isinstance(df, pd.DataFrame):
        header = list(df.columns)
        data = []
        
        for row in df.itertuples(index=False):
            row_values = list(row)
            if test and 'label' in df.columns:
                label_pos = list(df.columns).index('label')
                row_values[label_pos] = ' '
            data.append(row_values)

        final = str([header] + data)

    if test:
        
        final = final[:-5]

    return final


def serialization_markdown_new(df, test=False):
    
    """
        | name | age | label |
        |:-----|:-----|:-----|
        | helen | 47 | A |
        | john | 32 | B |
    """

    if isinstance(df, pd.Series):
        data = df.to_frame().T
    else:
        data = df.copy()
    
    if test and 'label' in data.columns:
        data['label'] = ' '
    
    markdown_lines = []
    
    headers = list(data.columns)
    markdown_lines.append('|' + '|'.join(headers) + '|')
    
    alignments = ['|:-----' for _ in data.columns]
    markdown_lines.append(''.join(alignments) + '|')
    
    for _, row in data.iterrows():
        row_values = [str(x) for x in row.values]
        markdown_lines.append('|' + '|'.join(row_values) + '|')

    final = '\n'.join(markdown_lines)

    if test:
        final = final[:-2]
    
    return final


def serialization_latex_new(df, test=False):
    
    """
        name & age & label \\
        \hline
        \hline
        helen & 47 & A \\
        \hline
    """
    
    if isinstance(df, pd.Series):
        data = df.to_frame().T
    else:
        data = df.copy()

    if test and 'label' in data.columns:
        data['label'] = r' '
    
    latex_rows = []
    
    latex_rows.append(' & '.join(data.columns) + r' \\')
    latex_rows.append(r'\hline')
    latex_rows.append(r'\hline') 
    for _, row in data.iterrows():
        row_values = [str(x) for x in row.values]
        latex_rows.append(' & '.join(row_values) + r' \\')
        latex_rows.append(r'\hline')

    final = '\n'.join(latex_rows)

    if test:
        final = final[:-11]
    
    return final


def serialization_csv_new(df, test=False, separator=', '):
    
    """
        name, age, label
        helen, 47, A
        john, 32, B
    """
    
    if isinstance(df, pd.Series):
        data = df.to_frame().T
    else:
        data = df.copy()
    
    if test and 'label' in data.columns:
        data['label'] = ' '
    
    csv_lines = []
    csv_lines.append(separator.join(data.columns))
    for _, row in data.iterrows():
        row_values = [str(x) for x in row.values]
        csv_lines.append(separator.join(row_values))

    final = '\n'.join(csv_lines)
    if test:
        final = final[:-1]
    
    return final


def serialization_html_new(df, test=False):
    
    """
        <table><thead><tr><th>name</th><th>age</th></tr></thead>
        <tbody><tr><td>helen</td><td>47</td></tr></tbody></table>
    """    
    if isinstance(df, pd.Series):
        data = df.to_frame().T
    else:
        data = df.copy()
    
    if test and 'label' in data.columns:
        data['label'] = ' '
    
    parts = ['<table>']
    
    # header
    parts.append('<thead><tr>')
    parts.extend(f'<th>{col}</th>' for col in data.columns)
    parts.append('</tr></thead>')

    # body
    parts.append('<tbody>')
    for _, row in data.iterrows():
        parts.append('<tr>')
        parts.extend(f'<td>{val}</td>' for val in row.values)
        parts.append('</tr>')
    parts.append('</tbody>')
    
    parts.append('</table>')
    final = ''.join(parts)
    if test:
        final = final[:-26]
    return final


def serialization_table_new(df, test=False):
    
    """
    
    | val1 | val2 | ... | val20 | target
    
    """
    
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    else:
        df = df.copy()
        
    features_names = df.columns.tolist()
    features_names = [name for name in features_names if name != 'label']
    features = df[features_names]
    target = df['label']
    
    serialized_rows = []
    for i in range(len(df)):
        
        feature_str = " | ".join(f" {str(val)} " for val in features.iloc[i])
        target_val = " " if test else str(target.iloc[i])
        row_str = f"|{feature_str} | {target_val} |"
        
        serialized_rows.append(row_str)

    #columns = " | ".join(df.columns.tolist())
    #concat_shots = [columns] + serialized_rows

    final = "\n".join(serialized_rows)
    if test:
        final = final[:-2]
    
    return final


def serialization_dict_new(df, test=False):
    """
    Сериализует DataFrame в формате:
    {
        'age': [22, 33, 44, 55, 66],
        'status': [1, 2, 3, 4, 5, 6], 
        'label': [1, 0, 1, 0, 1, ''] 
    }
    """

    if isinstance(df, pd.Series):
        df = df.to_frame().T
    else:
        df = df.copy()
        
    features_names = [col for col in df.columns if col != 'label']
    result = {}
    
    for feature in features_names:
        result[feature] = df[feature].tolist()
    
    if 'label' in df.columns:
        if test:
            labels = df['label'].tolist()
            if labels:
                labels[-1] = ''
            result['label'] = labels
        else:
            result['label'] = df['label'].tolist()

    serialized = ",\n ".join(f"{key}: {value}" for key, value in result.items())
    return f"{{\n {serialized} \n}}"


def serialization_dict_structure(df, test=False):
    """
    Возвращает словарь с сериализованными данными в виде Python-структуры:
    {
        'feature1': [val1, val2, ...],
        'feature2': [val1, val2, ...],
        'label': [val1, val2, ...]  # Последний элемент будет '' при test=True
    }
    """
    
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    else:
        df = df.copy()
        
    result = {
        col: df[col].tolist() 
        for col in df.columns 
        if col != 'label'
    }
    
    if 'label' in df.columns:
        if test:
            labels = df['label'].tolist()
            if labels:
                labels[-1] = None
            result['label'] = labels
        else:
            result['label'] = df['label'].tolist()
    
    return result


def append_to_serialized_dict(serialized_data, new_df):

    new_serialized = {k: v.copy() if isinstance(v, list) else v for k, v in serialized_data.items()}
    
    if isinstance(new_df, pd.Series):
        new_df = new_df.to_frame().T
    
    for col in new_serialized:
        if col == 'label':
            last_empty = len(new_serialized[col]) > 0 and new_serialized[col][-1] in (None, '')
            
            if last_empty:
                new_serialized[col][-1] = new_df[col].iloc[0]
                new_serialized[col].extend(new_df[col].iloc[1:].tolist())
            else:
                new_serialized[col].extend(new_df[col].tolist())

            new_serialized[col][-1] = None
        else:
            new_serialized[col].extend(new_df[col].tolist())
    
    return new_serialized


def dict_to_string(data_dict):

    lines = []
    for key, values in data_dict.items():

        formatted_values = []
        for v in values:
            if v is None:
                formatted_values.append('None')
            elif isinstance(v, str):
                formatted_values.append(f"'{v}'")
            else:
                formatted_values.append(str(v))
        
        line = f" '{key}': [{', '.join(formatted_values)}]"
        lines.append(line)
    
    result = "{\n" + ",\n".join(lines) + "\n}"
    
    return result


def serialization_markdown_masked_new(df, test=False):
    
    """
        | feature_1 | feature_2 | feature_3 |
        |:----------|:----------|:----------|
        | helen     | 47        | A         |
        | john      | 32        | B         |
    """
    
    if isinstance(df, pd.Series):
        data = df.to_frame().T
    else:
        data = df.copy()
    
    new_columns = [f'feature_{i+1}' for i in range(len(data.columns))]
    new_columns[-1] = 'label'
    data.columns = new_columns
    
    if test and 'label' in data.columns:  # feature_3 теперь соответствует label
        data['label'] = ' '
    
    markdown_lines = []
    
    headers = list(data.columns)
    markdown_lines.append('|' + '|'.join(headers) + '|')
    
    alignments = ['|:-----' for _ in data.columns]
    markdown_lines.append(''.join(alignments) + '|')
    
    for _, row in data.iterrows():
        row_values = [str(x) for x in row.values]
        markdown_lines.append('|' + '|'.join(row_values) + '|')

    final = '\n'.join(markdown_lines)

    if test:
        final = final[:-1]
    
    return final


def serialization_natural_language(df, test=False):

    column_name = 'natural_language_data_test' if test else 'natural_language_data'
    

    if isinstance(df, pd.Series):

        if column_name in df.index:
            data_value = df[column_name]
            if test:
                return data_value
            else:
                return str(data_value)
        elif hasattr(df, 'name') and df.name == column_name:

            if test:
                return df
            else:
                return '\n'.join(df.astype(str).values)
        else:

            available_keys = df.index.tolist() if hasattr(df, 'index') else []
            if column_name in available_keys:
                data_value = df[column_name]
                if test:
                    return data_value
                else:
                    return str(data_value)
            else:

                if test:
                    return df
                else:
                    return str(df) if len(df) == 1 else '\n'.join(df.astype(str).values)
    
    elif isinstance(df, pd.DataFrame):

        if column_name not in df.columns:
            available_columns = ", ".join(df.columns.tolist())
            raise ValueError(f"There is no column '{column_name}' in the dataset. "
                           f"Available columns: {available_columns}. "
                           f"\nPreprocessing for natural language serialization required.")
        
        if test:
            return df[column_name]
        else:
            return '\n'.join(df[column_name].astype(str).values)
    
    else:
        raise TypeError(f"Expected pandas.DataFrame or pandas.Series, got {type(df)}")