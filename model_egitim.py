import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. CSV dosyasını oku
df = pd.read_csv("traffic_light_summary.csv")  # Önceki oluşturduğumuz CSV

# 2. Etiketleri sayısallaştır
label_map = {"Düşük": 0, "Orta": 1, "Yüksek": 2}
df["Yoğunluk_Label"] = df["Yoğunluk"].map(label_map)

# 3. Özellik ve hedef sütunlarını ayır
X = df[["Ortalama Araç Sayısı"]]
y = df["Yoğunluk_Label"]

# 4. Eğitim ve test setine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Modeli eğit
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 6. Doğruluk değerlendirmesi
y_pred = model.predict(X_test)
print("Doğruluk:", accuracy_score(y_test, y_pred))

# 7. Modeli kaydet
joblib.dump(model, "trafik_yogunluk_modeli.joblib")


#test
# Eğittiğimiz modeli yükle
model = joblib.load("trafik_yogunluk_modeli.joblib")

ortalama_arac_sayisi = 12.3  # Örneğin
tahmin = model.predict([[ortalama_arac_sayisi]])

reverse_map = {0: "Düşük", 1: "Orta", 2: "Yüksek"}
print(f"Tahmin edilen trafik yoğunluğu: {reverse_map[tahmin[0]]}")

