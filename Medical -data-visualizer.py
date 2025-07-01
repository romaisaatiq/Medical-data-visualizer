import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def draw_cat_plot(df):
    # 1. Create df_cat with categorical variables converted to 0/1 for good visualization
    df_cat = df.copy()
    df_cat['overweight'] = (df_cat['weight'] / ((df_cat['height']/100) ** 2) > 25).astype(int)
    
    # Normalize 'cholesterol' and 'gluc' to 0/1
    df_cat['cholesterol'] = (df_cat['cholesterol'] > 1).astype(int)
    df_cat['gluc'] = (df_cat['gluc'] > 1).astype(int)
    
    # 2. Melt and group data for catplot
    df_cat = pd.melt(df_cat, id_vars=['cardio'], value_vars=[
        'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # Count the values grouped by 'cardio', 'variable', and 'value'
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
    # 3. Create the categorical plot
    fig = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar'
    )
    
    # 4. Return the figure for further use
    return fig.fig

def draw_heat_map(df):
    # 7. Clean the data according to instructions
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]
    
    # 8. Calculate correlation matrix
    corr = df_heat.corr()
    
    # 9. Generate mask for upper triangle
    mask = pd.np.triu(np.ones_like(corr, dtype=bool))
    
    # 10. Set up matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 11. Plot heatmap
    sns.heatmap(
        corr,
        annot=True,
        fmt='.1f',
        mask=mask,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5},
        ax=ax,
        center=0
    )
    
    # 12. Return the figure
    return fig