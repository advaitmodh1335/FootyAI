import os
import django
import pandas as pd
import numpy as np

from django.core.management.base import BaseCommand
from django.conf import settings

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Django init
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'soccer_project.settings')
django.setup()

from prediction_app.models import Match

class Command(BaseCommand):
    help = 'Trains and saves the soccer prediction model (time-split, calibrated).'

    def handle(self, *args, **options):
        self.stdout.write("Fetching data from the database...")

        # Safer existence check
        if not Match.objects.exists():
            self.stdout.write(self.style.ERROR("Error: No data found. Please run load_data.py first."))
            return

        # Pull rows and build DataFrame
        qs = Match.objects.all().values()
        df = pd.DataFrame(list(qs))
        if df.empty:
            self.stdout.write(self.style.ERROR("Error: dataframe is empty."))
            return

        # --- Basic sanity checks ---
        required_cols = [
            'date', 'team', 'opponent', 'gf', 'ga', 'sh', 'sot',
            'venue_code', 'team_code', 'opp_code', 'target'
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            self.stdout.write(self.style.ERROR(f"Error: missing columns: {missing}"))
            return

        # Ensure datetype + sort
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # --- Rolling features for the team (no leakage via closed='left') ---
        rolling_cols = ['gf', 'ga', 'sh', 'sot']
        for col in rolling_cols:
            df[f'{col}_rolling'] = (
                df.groupby('team')[col]
                  .rolling(window=5, min_periods=1, closed='left')
                  .mean()
                  .reset_index(level=0, drop=True)
            )

        # --- Opponent rolling via self-join: join where left.opponent == right.team (same date) ---
        # Build a compact table of "team rolling stats" keyed by (date, team)
        team_roll = df[['date', 'team'] + [f'{c}_rolling' for c in rolling_cols]].copy()
        team_roll = team_roll.rename(columns={
            'team': 'opp_team',
            'gf_rolling': 'opp_gf_rolling',
            'ga_rolling': 'opp_ga_rolling',
            'sh_rolling': 'opp_sh_rolling',
            'sot_rolling': 'opp_sot_rolling'
        })

        # Merge: left has columns (..., opponent), right maps opponent->opp rolling (by same date)
        df = df.merge(
            team_roll,
            left_on=['date', 'opponent'],
            right_on=['date', 'opp_team'],
            how='left'
        ).drop(columns=['opp_team'])

        # Fill NaNs arising early in season or unmatched merges
        for col in [f'{c}_rolling' for c in rolling_cols] + \
                   [f'opp_{c}_rolling' for c in ['gf', 'ga', 'sh', 'sot']]:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        # --- Features / Target ---
        feature_cols = [
            'venue_code', 'team_code', 'opp_code',
            'gf_rolling', 'ga_rolling', 'sh_rolling', 'sot_rolling',
            'opp_gf_rolling', 'opp_ga_rolling', 'opp_sh_rolling', 'opp_sot_rolling'
        ]
        target_col = 'target'

        missing = [c for c in feature_cols + [target_col] if c not in df.columns]
        if missing:
            self.stdout.write(self.style.ERROR(f"Error: Missing columns after feature engineering: {missing}"))
            return

        # Optional: ensure target has no NaNs
        df = df.dropna(subset=[target_col]).reset_index(drop=True)

        # --- Time-based split (no leakage): last ~20% by date as validation ---
        # You can adjust the split ratio here.
        split_date = df['date'].quantile(0.8)
        train_df = df[df['date'] <= split_date].copy()
        val_df   = df[df['date']  > split_date].copy()

        if train_df.empty or val_df.empty:
            self.stdout.write(self.style.ERROR("Error: time-based split produced empty train/val. Consider adjusting split."))
            return

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_val   = val_df[feature_cols].values
        y_val   = val_df[target_col].values

        self.stdout.write(f"Train size: {len(train_df)} | Val size: {len(val_df)} (split_date={split_date.date()})")

        # --- Base model ---
        base = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=1
        )
        base.fit(X_train, y_train)

        # --- Calibrate on validation slice (probability quality matters) ---
        cal = CalibratedClassifierCV(base, method='isotonic', cv='prefit')
        cal.fit(X_val, y_val)

        # --- Metrics ---
        # If target is 3-class (e.g., home/draw/away), log_loss/brier are multiclass safe.
        proba_val = cal.predict_proba(X_val)
        pred_val  = np.argmax(proba_val, axis=1)

        # Brier (multiclass) – implement one-vs-all average
        classes = np.unique(y_train)
        # Build one-hot for y_val
        y_val_ohe = np.zeros((len(y_val), len(classes)))
        for i, c in enumerate(classes):
            y_val_ohe[:, i] = (y_val == c).astype(float)
        brier = np.mean(np.sum((proba_val - y_val_ohe) ** 2, axis=1))

        ll = log_loss(y_val, proba_val, labels=classes)
        acc = accuracy_score(y_val, pred_val)

        self.stdout.write(self.style.SUCCESS(
            f"Validation — Accuracy: {acc:.3f} | Brier: {brier:.4f} | LogLoss: {ll:.4f}"
        ))

        # --- Persist artifacts ---
        le = LabelEncoder()
        le.fit(df['team'])  # keep for inference helpers if you need name<->id mapping

        out_model = os.path.join(settings.BASE_DIR, 'soccer_predictor_model.joblib')
        out_cal   = os.path.join(settings.BASE_DIR, 'soccer_predictor_calibrated.joblib')
        out_le    = os.path.join(settings.BASE_DIR, 'label_encoder.joblib')

        joblib.dump(base, out_model)
        joblib.dump(cal,  out_cal)
        joblib.dump(le,   out_le)

        self.stdout.write(self.style.SUCCESS(
            f"Saved model to {out_model}, calibrated model to {out_cal}, and LabelEncoder to {out_le}"
        ))
