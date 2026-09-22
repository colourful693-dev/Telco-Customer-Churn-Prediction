import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve,accuracy_score

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("C:/Users/浩翔/archive/WA_Fn-UseC_-Telco-Customer-Churn.csv",encoding="utf-8")
print(df.info())
print(df.describe())
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True))

df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

print(df.isnull().sum())
df = df.dropna()
df = df.drop("customerID", axis=1)
print(df.duplicated().sum())
df = df.drop_duplicates()
df["Churn"] = df["Churn"].map({"Yes":1, "No":0})
print(df["Churn"].value_counts())

#EDA分析
plt.figure(figsize=(6,4))
sns.countplot(x="Churn",data=df)
plt.title("客户流失分布")
plt.savefig("C:/Users/浩翔/archive/churn_dist.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8,4))
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("合约类型 vs 客户流失")
plt.savefig("C:/Users/浩翔/archive/contract_churn.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10,4))
sns.countplot(x="PaymentMethod", hue="Churn", data=df)
plt.title("支付方式 vs 客户流失")
plt.xticks(rotation=30)
plt.savefig("C:/Users/浩翔/archive/payment_churn.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8,4))
sns.histplot(data=df, x="MonthlyCharges", hue="Churn", kde=True)
plt.title("月度消费分布")
plt.savefig("C:/Users/浩翔/archive/monthly_charge.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10,8))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("特征相关性热力图")
plt.savefig("C:/Users/浩翔/archive/corr_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

#特征编码数据处理
X = df.iloc[:,0:19]
y = df["Churn"]

print(df.shape)            # (行数, 列数)
print(df.columns.tolist()) # 所有列名


#独热编码
num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_features = [col for col in X.columns if col not in num_features]
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_features)
    ],
    remainder="passthrough"
)

#划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
#逻辑回归
lr_pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("lr", LogisticRegression(max_iter=1000))
])
lr_pipe.fit(X_train, y_train)
y_pred_lr = lr_pipe.predict(X_test)
y_pred_prob_lr = lr_pipe.predict_proba(X_test)[:, 1]

print("="*20)
print("逻辑回归模型评估")
print(classification_report(y_test, y_pred_lr))
print("AUC =", roc_auc_score(y_test, y_pred_prob_lr))
print("混淆矩阵：")
print(confusion_matrix(y_test, y_pred_lr))

#随机森林
from sklearn.ensemble import RandomForestClassifier
rf_pipe = Pipeline([
    ("preprocess", preprocessor),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])
rf_pipe.fit(X_train, y_train)
y_pred_rf = rf_pipe.predict(X_test)
y_prob_rf = rf_pipe.predict_proba(X_test)[:,1]
print("===== 随机森林 =====")
print(f"准确率Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
print(f"AUC: {roc_auc_score(y_test, y_prob_rf):.3f}")
print(classification_report(y_test, y_pred_rf))

#ROC曲线
plt.figure(figsize=(6,6))
# 逻辑回归
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_prob_lr)
plt.plot(fpr_lr, tpr_lr, label=f"逻辑回归 AUC={roc_auc_score(y_test,y_pred_prob_lr):.3f}")
# 随机森林
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
plt.plot(fpr_rf, tpr_rf, label=f"随机森林 AUC={roc_auc_score(y_test,y_prob_rf):.3f}")

plt.plot([0,1],[0,1],"k--")
plt.xlabel("FPR 假阳性率")
plt.ylabel("TPR 真阳性率")
plt.title("逻辑回归 vs 随机森林 ROC曲线对比")
plt.legend()
plt.savefig("C:/Users/浩翔/archive/roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()

#随机森林重要性特征
plt.figure(figsize=(8,5))
ohe = preprocessor.named_transformers_["cat"]
cat_names = ohe.get_feature_names_out(cat_features).tolist()
all_feat_names = num_features + cat_names

rf_model = rf_pipe.named_steps["rf"]
imp_df = pd.DataFrame({
    "feature": all_feat_names,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False).head(10)

sns.barplot(x="importance", y="feature", data=imp_df)
plt.title("随机森林 Top10 特征重要性")
plt.savefig("C:/Users/浩翔/archive/rf_feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()
