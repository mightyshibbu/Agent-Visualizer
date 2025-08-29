import argparse
import random
import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()

# Helper to randomly inject missing and corrupt values
def inject_edge_cases(df: pd.DataFrame, missing_rate=0.01, corrupt_rate=0.005):
    df = df.copy()
    total = df.size
    # Missing values
    n_missing = int(total * missing_rate)
    for _ in range(n_missing):
        i = random.randrange(df.shape[0])
        j = random.randrange(df.shape[1])
        df.iat[i, j] = np.nan
    # Corrupt values (random text)
    n_corrupt = int(total * corrupt_rate)
    for _ in range(n_corrupt):
        i = random.randrange(df.shape[0])
        j = random.randrange(df.shape[1])
        df.iat[i, j] = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    return df

# 1. Campaign Data: impressions, clicks, conversions, spend, cpm, cpc, cpa
def generate_campaign_data(n=1000):
    data = []
    for _ in range(n):
        impressions = random.randint(100, 100000)
        clicks = random.randint(0, impressions)
        conversions = random.randint(0, clicks)
        spend = round(random.uniform(10, 10000), 2)
        cpm = round((spend / impressions) * 1000, 2) if impressions else np.nan
        cpc = round((spend / clicks), 2) if clicks else np.nan
        cpa = round((spend / conversions), 2) if conversions else np.nan
        date = fake.date_between(start_date='-90d', end_date='today')
        data.append({
            'date': date,
            'platform': random.choice(['Google Ads', 'Meta Ads', 'LinkedIn', 'Twitter']),
            'campaign_id': fake.uuid4(),
            'impressions': impressions,
            'clicks': clicks,
            'conversions': conversions,
            'spend': spend,
            'cpm': cpm,
            'cpc': cpc,
            'cpa': cpa
        })
    df = pd.DataFrame(data)
    return inject_edge_cases(df)

# 2. Web & Social Analytics: page views, sessions, bounce rate, engagement metrics

def generate_web_social_analytics(n=1000):
    data = []
    for _ in range(n):
        page_views = random.randint(0, 50000)
        sessions = random.randint(0, page_views) if page_views else 0
        bounce_rate = round(random.uniform(0, 100), 2)
        likes = random.randint(0, 10000)
        shares = random.randint(0, 5000)
        comments = random.randint(0, 3000)
        date = fake.date_between(start_date='-90d', end_date='today')
        data.append({
            'date': date,
            'source': random.choice(['Website', 'Facebook', 'Twitter', 'Instagram', 'LinkedIn']),
            'page_views': page_views,
            'sessions': sessions,
            'bounce_rate': bounce_rate,
            'likes': likes,
            'shares': shares,
            'comments': comments
        })
    df = pd.DataFrame(data)
    return inject_edge_cases(df)

# 3. Audience Segments: demographics, interests, geolocation, device/platform

def generate_audience_segments(n=500):
    data = []
    for _ in range(n):
        age = random.randint(18, 65)
        gender = random.choice(['Male', 'Female', 'Other'])
        interest = random.choice(['Tech', 'Finance', 'Health', 'Travel', 'Food', 'Sports'])
        country = fake.country()
        city = fake.city()
        device = random.choice(['Desktop', 'Mobile', 'Tablet'])
        data.append({
            'user_id': fake.uuid4(),
            'age': age,
            'gender': gender,
            'interest': interest,
            'country': country,
            'city': city,
            'device': device
        })
    df = pd.DataFrame(data)
    return inject_edge_cases(df)

# 4. Time Series: daily/hourly metrics over months for campaign performance

def generate_time_series(n_days=90, freq='D'):
    idx = pd.date_range(end=datetime.today(), periods=n_days, freq=freq)
    data = {
        'date': idx,
        'metric_value': np.random.randint(100, 10000, size=len(idx))
    }
    df = pd.DataFrame(data)
    return inject_edge_cases(df)

# 5. Media Schedules: GRPs, reach, frequency, channel schedules

def generate_media_schedule(n=500):
    data = []
    for _ in range(n):
        date = fake.date_between(start_date='-90d', end_date='today')
        channel = random.choice(['TV', 'Radio', 'Print', 'Online'])
        schedule_item = fake.sentence(nb_words=4)
        grps = round(random.uniform(10, 200), 2)
        reach = round(random.uniform(0, 100), 2)
        frequency = round(random.uniform(1, 10), 2)
        data.append({
            'date': date,
            'channel': channel,
            'schedule_item': schedule_item,
            'GRPs': grps,
            'reach': reach,
            'frequency': frequency
        })
    df = pd.DataFrame(data)
    return inject_edge_cases(df)

# CLI Entry

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic datasets for LLM fine-tuning')
    parser.add_argument('--type', choices=['campaign', 'web_social', 'audience', 'time_series', 'media'], required=True)
    parser.add_argument('--count', type=int, default=1000, help='Number of records or days to generate')
    parser.add_argument('--format', choices=['csv', 'json'], default='csv')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    args = parser.parse_args()

    if args.type == 'campaign':
        df = generate_campaign_data(args.count)
    elif args.type == 'web_social':
        df = generate_web_social_analytics(args.count)
    elif args.type == 'audience':
        df = generate_audience_segments(args.count)
    elif args.type == 'time_series':
        df = generate_time_series(args.count)
    elif args.type == 'media':
        df = generate_media_schedule(args.count)

    output = args.output or f'synthetic_{args.type}.{args.format}'
    if args.format == 'csv':
        df.to_csv(output, index=False)
    else:
        df.to_json(output, orient='records', date_format='iso')

    print(f'Dataset saved to {output}')

if __name__ == '__main__':
    main()
