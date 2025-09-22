import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_data(file_path):
    return pd.read_csv(file_path)


def clean_data(df):
    df = df.drop_duplicates()
    # Converting created_date and canceled_date to datetime
    df['created_date'] = pd.to_datetime(df['created_date'])
    df['canceled_date'] = pd.to_datetime(df['canceled_date'], errors='coerce')
    return df

def main():
    
    file_path = os.path.join('data', 'Subscription Cohort Analysis Data.csv')
    df = load_data(file_path)
    df = clean_data(df)
    
    # Basic stats
    # -----------
    
    # Finding earliest and latest created_date
    print('Dataset tiemframe start:', df['created_date'].min())  # Earliest: 2022-09-01
    print('Dataset timeframe end:', df['created_date'].max())  # Latest: 2023-09-08
    
    '''
    So, this dataset shows the data for a year. This won't necissarily
    give a full idea of the company's overall trends, but might reflect
    seasonal trends and developments from the 2022 - 2023 year.
    '''
    
    print("\n------------------------\n")
    print('Total subscribers:', len(df['canceled_date']))
    print('Current subscribers:', len(df['canceled_date'].dropna()))
    
    '''
    3069 total entries, 2004 non-null canceled_date entries. This means
    1065 entries are still active subscriptions (about a third of total entries).
    A two-thirds drop in subscriptions over a year is quite significant.
    Seeing as this dataset's timeframe includes the end of COVID lockdowns,
    it is not necessarily surprising to see a large drop off in subscriptions
    as people return to in-person activities.
    
    I will be reflecting the distribution of cancel dates over time.
    With the above in mind, I predict the cancellations will skew towards
    The end of the datase's timeframe.
    '''
    
    print("\n------------------------\n")
    
    
    # Find average subscription duration
    df['subscription_duration'] = (df['canceled_date'] - df['created_date']).dt.days
    avg_duration = df['subscription_duration'].mean()
    print(f"Average subscription duration: {avg_duration:.1f} days")
    
    # Finding variance and standard deviation of subscription duration
    duration_variance = df['subscription_duration'].var()
    duration_std = df['subscription_duration'].std()
    print(f"Variance of subscription duration: {duration_variance:.2f}")
    print(f"Standard deviation of subscription duration: {duration_std:.2f}")
    # The average subscription duration is about 100 days, with a high variance
    # and standard deviation. This indicates a wide range of subscription lengths,
    # with some users canceling quickly and others staying subscribed for much longer.    

    print("\n------------------------\n")
    # Visualize cancellations over time
    
    # Give false date for null canceled_date values
    df['canceled_date'] = df['canceled_date'].dropna()
    plt.figure(figsize=(12, 6))
    sns.histplot(df['canceled_date'], bins=30, kde=False)
    plt.title('Distribution of Subscription Cancellation Dates')
    plt.xlabel('Canceled Date')
    plt.ylabel('Number of Cancellations')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    # The cancellations show a clear increase over time, with a peak
    # towards the end of the dataset's timeframe. This supports the hypothesis
    # that cancellations increased as people returned to in-person activities.
    
    # heatmap of cancellations by month
        # Only look at rows where a cancellation actually happened

    # Extract year-month as a period
    df['canceled_month'] = df['canceled_date'].dt.to_period('M')

    # Count cancellations per month
    cancellations_by_month = df.groupby('canceled_month').size().reset_index(name='count')
    cancellations_by_month['canceled_month'] = cancellations_by_month['canceled_month'].dt.to_timestamp()
    
    data = cancellations_by_month['count'].to_numpy().reshape(1, -1)
    
    # Since September 2023 has only 8 days of the month, we can multiply 
    # its number of entries by ~3.75 to give a more accurate representation
    # of what a full month might look like.
    if cancellations_by_month['canceled_month'].iloc[-1].month == 9:
        data[0, -1] = int(data[0, -1] * (30 / 8))

    # Plot as heatmap
    plt.figure(figsize=(12, 2))
    sns.heatmap(data, annot=True, fmt='d', cmap='YlGnBu', cbar_kws={'label': 'Cancellations'})

    # Put month labels along the x-axis
    plt.xticks(
        ticks=np.arange(len(cancellations_by_month)) + 0.5,
        labels=cancellations_by_month['canceled_month'].dt.strftime('%b %Y'),
        rotation=45,
        ha='right'
    )

    plt.yticks([], [])  # hide y-axis (since it’s just 1 row)
    plt.title('Cancellations by Month')
    plt.tight_layout()
    plt.show()

    '''
    The heatmap shows a clear increase in cancellations over the months,
    with a significant spike in the last month (September 2023). This aligns
    with the earlier histogram and supports the idea that cancellations
    increased as people returned to normal activities post-pandemic.
    '''

if __name__ == "__main__":
    main()