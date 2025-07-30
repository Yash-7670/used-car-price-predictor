def get_models_for_brand(df, brand):
    return df[df['brand'] == brand]['model'].unique().tolist()

def get_options_for_model(df, brand, model):
    subset = df[(df['brand'] == brand) & (df['model'] == model)]
    return {
        'year': sorted(subset['year'].unique()),
        'fuel': subset['fuel'].unique().tolist(),
        'transmission': subset['transmission'].unique().tolist(),
        'seller_type': subset['seller_type'].unique().tolist(),
        # etc.
    }
