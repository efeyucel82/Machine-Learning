# Gerekli kütüphaneleri import ettim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from scipy import stats

# 1. VERİ SETİ OLUŞTURMA
np.random.seed(0)
df = pd.DataFrame({
    'Age': np.random.randint(20, 70, 100),
    'Income': np.random.normal(50000, 15000, 100),
    'Education_Level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 100),
    'Spending_Score': np.random.normal(50, 10, 100)
})

# 2. EKSİK VERİ OLUŞTURMA & DOLDURMA
df.loc[np.random.choice(df.index, 10), 'Income'] = np.nan
df.loc[:, 'Income'] = df['Income'].fillna(df['Income'].mean())

# 3. KATEGORİK VERİYİ SAYIYA ÇEVİRME
df['Education_Level_Cat'] = df['Education_Level'].astype('category').cat.codes

# 4. KORELASYON ANALİZİ & GRAFİĞİ
correlation_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Korelasyon Matrisi")
plt.show()

# 5. BASİT DOĞRUSAL REGRESYON 
X = df[['Age']]
y = df['Spending_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

print("\n BASİT REGRESYON ")
print("Eğim:", model.coef_[0])
print("Sabit:", model.intercept_)
print("R^2 Skoru:", round(model.score(X_test, y_test), 4))

# 6. BASİT REGRESYON GRAFİĞİ
y_pred = model.predict(X_test)
plt.scatter(X_test, y_test, color='blue', label='Gerçek Değer')
plt.plot(X_test, y_pred, color='red', label='Tahmin')
plt.title("Yaş vs Harcama Skoru")
plt.xlabel("Yaş")
plt.ylabel("Spending Score")
plt.legend()
plt.show()

# 7. ÇOKLU DOĞRUSAL REGRESYON
X_multi = df[['Age', 'Income', 'Education_Level_Cat']]
y_multi = df['Spending_Score']
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)

print("\n--- ÇOKLU REGRESYON ---")
print("Katsayılar:", model_multi.coef_)
print("Sabit:", model_multi.intercept_)
print("R^2 Skoru:", round(model_multi.score(X_test_m, y_test_m), 4))

# 8. NORMALİZASYON + D'AGOSTINO K^2 TESTİ
scaler = StandardScaler()
spending_scaled = scaler.fit_transform(df[['Spending_Score']])
p_value = stats.normaltest(spending_scaled.flatten()).pvalue

print("\n--- NORMAL DAĞILIM TESTİ (D’Agostino) ---")
print("p-değeri:", round(p_value, 4))
print("Sonuç:", "Başarılı" if p_value > 0.05 else "Başarısız")

# 9. DÖNÜŞÜM TÜRLERİ (Box-Cox, Log, Karekök)
transformers = {
    'Box-Cox': lambda x: PowerTransformer(method='box-cox').fit_transform(x) if (x > 0).all() else np.nan,
    'Log': lambda x: np.log(x) if (x > 0).all() else np.nan,
    'Square Root': lambda x: np.sqrt(x) if (x >= 0).all() else np.nan
}

print("\n DÖNÜŞÜM KARŞILAŞTIRMASI ")
for name, func in transformers.items():
    try:
        transformed = func(df[['Spending_Score']].values)
        pval = stats.normaltest(transformed.flatten()).pvalue
        sonuc = "Başarılı" if pval > 0.05 else "BAşarısız"
        print(f"{name}: p-değeri={round(pval, 4)} → {sonuc}")
    except Exception as e:
        print(f"{name}: Uygulanamadı {e}")
