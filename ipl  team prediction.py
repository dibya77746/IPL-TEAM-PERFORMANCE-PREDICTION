import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load datasets
matches = pd.read_csv('matches.csv')
deliveries = pd.read_csv('deliveries.csv')

# --- Data Cleaning ---
# Remove matches with no result or missing winner
matches = matches.dropna(subset=['winner', 'team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'season'])
matches = matches[matches['result'].str.lower() != 'no result']

# Standardize team names (optional, for older seasons)
team_name_map = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
    'Deccan Chargers': 'Sunrisers Hyderabad'
}
for col in ['team1', 'team2', 'winner', 'toss_winner']:
    matches[col] = matches[col].replace(team_name_map)

# --- IPL Champions (Most Cups) ---
finals = matches.drop_duplicates(subset=['season'], keep='last')
champions = finals['winner'].value_counts()
most_cups_team = champions.idxmax()
most_cups = champions.max()

# --- Most Finals Played ---
finalists = pd.concat([finals['team1'], finals['team2']])
finals_played = finalists.value_counts()
most_finals_team = finals_played.idxmax()
most_finals = finals_played.max()

# --- Most Playoffs Played ---
# Playoff matches: Use 'eliminator' or 'final' columns if available, else fallback to last 4 matches per season
if 'eliminator' in matches.columns:
    playoffs = matches[matches['eliminator'] == 'Y']
elif 'final' in matches.columns:
    playoffs = matches[matches['final'] == 'Y']
else:
    playoffs = matches.groupby('season').tail(4)

playoff_teams = pd.concat([playoffs['team1'], playoffs['team2']])
playoffs_played = playoff_teams.value_counts()
most_playoffs_team = playoffs_played.idxmax()
most_playoffs = playoffs_played.max()

# --- Prepare features and target for prediction ---
for col in ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision']:
    matches[col] = matches[col].astype('category')

X = pd.get_dummies(matches[['team1', 'team2', 'venue', 'toss_winner', 'toss_decision']])
y = matches['winner'].astype('category').cat.codes
winner_mapping = dict(enumerate(matches['winner'].astype('category').cat.categories))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Prediction Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=winner_mapping.values()))

# --- IPL Champions ---
print(f"\nMost IPL Cups (Champions) till 2024: {most_cups_team} ({most_cups} times)")
print("Champions by year:")
print(finals[['season', 'winner']].set_index('season'))

# --- Most Finals Played ---
print(f"\nTeam with Most Finals Played: {most_finals_team} ({most_finals} times)")
print(finals_played)

# --- Most Playoffs Played ---
print(f"\nTeam with Most Playoffs Played: {most_playoffs_team} ({most_playoffs} times)")
print(playoffs_played)

# --- Figure 1: Confusion Matrix ---
plt.figure(figsize=(12,8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=winner_mapping.values(), yticklabels=winner_mapping.values())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Winner')
plt.ylabel('Actual Winner')
plt.tight_layout()
plt.show()

# --- Figure 2: Team Win Counts ---
plt.figure(figsize=(10,5))
matches['winner'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Total Wins by Team')
plt.xlabel('Team')
plt.ylabel('Number of Wins')
plt.tight_layout()
plt.show()

# --- Figure 3: Venue-wise Match Count ---
plt.figure(figsize=(12,5))
matches['venue'].value_counts().head(15).plot(kind='bar', color='orange')
plt.title('Top 15 Venues by Number of Matches')
plt.xlabel('Venue')
plt.ylabel('Number of Matches')
plt.tight_layout()
plt.show()

# --- Figure 4: Toss Decision Impact ---
plt.figure(figsize=(6,4))
sns.countplot(x='toss_decision', hue='winner', data=matches)
plt.title('Toss Decision vs Winner')
plt.xlabel('Toss Decision')
plt.ylabel('Count')
plt.legend(title='Winner', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- Figure 5: Year-wise Matches Played ---
matches['year'] = matches['season']
plt.figure(figsize=(10,4))
matches['year'].value_counts().sort_index().plot(kind='bar', color='green')
plt.title('Matches Played Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Matches')
plt.tight_layout()
plt.show()

# --- Figure 6: Team1 vs Team2 Win Heatmap ---
pivot = pd.crosstab(matches['team1'], matches['winner'])
plt.figure(figsize=(12,8))
sns.heatmap(pivot, cmap='YlGnBu', annot=False)
plt.title('Team1 vs Winner Heatmap')
plt.xlabel('Winner')
plt.ylabel('Team1')
plt.tight_layout()
plt.show()

# --- Figure 7: Most Playoff Appearances ---
plt.figure(figsize=(10,5))
playoffs_played.head(10).plot(kind='bar', color='purple')
plt.title('Top 10 Teams by Playoff Appearances')
plt.xlabel('Team')
plt.ylabel('Playoff Matches Played')
plt.tight_layout()
plt.show()
# --- Figure 8: IPL Finals Wins by Team ---
plt.figure(figsize=(10,6))
finals['winner'].value_counts().plot(kind='bar', color='lightcoral')
plt.title('Most IPL Finals Wins by Team')
plt.xlabel('Team')
plt.ylabel('Number of IPL Titles')
plt.tight_layout()
plt.show()

# --- Figure: Season-wise IPL Champions ---
plt.figure(figsize=(12,5))
sns.pointplot(
    data=finals.sort_values('season'),
    x='season',
    y='winner',
    color='blue',
    join=False
)
plt.title('IPL Champions by Season')
plt.xlabel('Season')
plt.ylabel('Champion Team')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# --- Example Prediction ---
sample = {
    'team1': 'Mumbai Indians',
    'team2': 'Chennai Super Kings',
    'venue': 'Wankhede Stadium',
    'toss_winner': 'Chennai Super Kings',
    'toss_decision': 'bat'
}
sample_df = pd.DataFrame([sample])
sample_encoded = pd.get_dummies(sample_df)
sample_encoded = sample_encoded.reindex(columns=X.columns, fill_value=0)
predicted = model.predict(sample_encoded)[0]
print("\nSample Prediction: Likely winner is", winner_mapping[predicted])