import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


dataset = dataset.drop_duplicates().dropna()


categorical_columns = [
    'BusinessTravel', 'Department', 'Education', 'Gender', 
    'OverTime', 'JobRole', 'MaritalStatus'
]
dataset = pd.get_dummies(dataset, columns=categorical_columns)


dataset['Attrition'] = dataset['Attrition'].map({'Yes': 1, 'No': 0})


features = dataset.drop('Attrition', axis=1)
target = dataset['Attrition']


X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"ROC AUC Score: {roc_auc:.2f}")


dataset['TurnoverLikelihood'] = model.predict_proba(features)[:, 1]


fig, axs = plt.subplots(1, 3, figsize=(24, 6))


feature_importances = pd.DataFrame(model.feature_importances_, index=features.columns, columns=['Importance']).sort_values('Importance', ascending=False)
sns.barplot(x='Importance', y=feature_importances.index, data=feature_importances, ax=axs[0])
axs[0].set_title('Feature Importance')
axs[0].set_xlabel('Importance')
axs[0].set_ylabel('Feature')

sns.histplot(dataset['TurnoverLikelihood'], bins=30, kde=True, ax=axs[1])
axs[1].set_title('Distribution of Turnover Likelihood')
axs[1].set_xlabel('Turnover Likelihood')
axs[1].set_ylabel('Frequency')


def simulate_scenario(data, satisfaction_change=0, compensation_change=0):

    scenario_data = data.drop(['Attrition', 'TurnoverLikelihood'], axis=1).copy()

    
    if 'JobSatisfaction' in scenario_data.columns:
        scenario_data['JobSatisfaction'] = scenario_data['JobSatisfaction'] + satisfaction_change
    if 'MonthlyIncome' in scenario_data.columns:
        scenario_data['MonthlyIncome'] = scenario_data['MonthlyIncome'] * (1 + compensation_change)

    
    scenario_data['ScenarioTurnoverLikelihood'] = model.predict_proba(scenario_data)[:, 1]

    return scenario_data


scenario_result = simulate_scenario(dataset, satisfaction_change=1, compensation_change=0.1)


sns.histplot(scenario_result['ScenarioTurnoverLikelihood'], bins=30, kde=True, ax=axs[2])
axs[2].set_title('Distribution of Turnover Likelihood after Scenario')
axs[2].set_xlabel('Scenario Turnover Likelihood')
axs[2].set_ylabel('Frequency')


plt.tight_layout()


plt.show()

