import os
import django
import pandas as pd
from datetime import datetime

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'soccer_project.settings')
django.setup()

from prediction_app.models import Match
from sklearn.preprocessing import LabelEncoder

def load_data_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    print("Loading data into the database...")
    Match.objects.all().delete()
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['result'])
    def encode_result(result):
        if result == 'W': return 1
        elif result == 'D': return 0
        else: return -1
    df['target'] = df['result'].apply(encode_result)
    le = LabelEncoder()
    df['team_code'] = le.fit_transform(df['team'])
    df['opp_code'] = le.fit_transform(df['opponent'])
    df['venue_code'] = le.fit_transform(df['venue'])
    for _, row in df.iterrows():
        Match.objects.create(
            date=row['date'],
            time=datetime.strptime(row['time'], '%H:%M').time(),
            comp=row['comp'],
            round=row['round'],
            day=row['day'],
            venue=row['venue'],
            team=row['team'],
            opponent=row['opponent'],
            result=row['result'],
            gf=row['gf'],
            ga=row['ga'],
            sh=row['sh'],
            sot=row['sot'],
            xg=row['xg'],
            xga=row['xga'],
            poss=row['poss'],
            season=row['season'],
            captain=row['captain'],
            formation=row['formation'],
            opp_formation=row['opp formation'],
            referee=row['referee'],
            team_code=row['team_code'],
            opp_code=row['opp_code'],
            venue_code=row['venue_code'],
            target=row['target']
        )
    print("Data loading complete!")
if __name__ == '__main__':
    file_path = 'matches_full.csv'
    load_data_from_csv(file_path)
