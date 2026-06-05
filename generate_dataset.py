import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

study_hours = np.random.normal(5, 2, n).clip(0, 12)
attendance = np.random.normal(75, 15, n).clip(40, 100)
prev_grade = np.random.normal(65, 15, n).clip(30, 100)
sleep_hours = np.random.normal(7, 1.5, n).clip(3, 10)
extracurricular = np.random.randint(0, 2, n)
internet_access = np.random.randint(0, 2, n)
family_support = np.random.randint(1, 5, n)
absences = np.random.randint(0, 20, n)

score = (
    0.30 * study_hours / 12 +
    0.25 * attendance / 100 +
    0.25 * prev_grade / 100 +
    0.10 * sleep_hours / 10 +
    0.05 * family_support / 4 +
    0.05 * internet_access -
    0.10 * absences / 20 +
    np.random.normal(0, 0.05, n)
).clip(0, 1)

passed = (score >= 0.5).astype(int)

df = pd.DataFrame({
    'study_hours': np.round(study_hours, 1),
    'attendance_pct': np.round(attendance, 1),
    'prev_grade': np.round(prev_grade, 1),
    'sleep_hours': np.round(sleep_hours, 1),
    'extracurricular': extracurricular,
    'internet_access': internet_access,
    'family_support': family_support,
    'absences': absences,
    'passed': passed
})

df.to_csv('student_data.csv', index=False)
print(f"Dataset saved: {len(df)} records, {df['passed'].mean()*100:.1f}% pass rate")
