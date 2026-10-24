#%%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

DATA_URL = "https://raw.githubusercontent.com/AbhishekDataAnalyst/Credit-Card-Approval/main/Credit_card.csv"

try:
    IMG_DIR = Path(__file__).parent / "images"
except NameError:
    IMG_DIR = Path.cwd() / "images"
IMG_DIR.mkdir(exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

df = pd.read_csv(DATA_URL)

print(df.shape)
print(df.head())
#%%
#чистка датасета от id
df = df.drop("Ind_ID", axis=1)
#таргет
y = df["Car_Owner"]
X = df.drop("Car_Owner", axis=1)

print(y.value_counts())
#%%
from sklearn.model_selection import train_test_split

#сплит до любой обработки, чтобы статистики считались только по train
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=32,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
#%%
#заполнение пропусков (статистики считаем по train и применяем к test)

X_train = X_train.copy()
X_test = X_test.copy()

gender_mode = X_train["GENDER"].mode()[0]
income_median = X_train["Annual_income"].median()
birthday_median = X_train["Birthday_count"].median()

X_train["GENDER"] = X_train["GENDER"].fillna(gender_mode)
X_test["GENDER"] = X_test["GENDER"].fillna(gender_mode)
#профессия
X_train["Type_Occupation"] = X_train["Type_Occupation"].fillna("Unknown")
X_test["Type_Occupation"] = X_test["Type_Occupation"].fillna("Unknown")
#годовой доход
X_train["Annual_income"] = X_train["Annual_income"].fillna(income_median)
X_test["Annual_income"] = X_test["Annual_income"].fillna(income_median)
#возраст
X_train["Birthday_count"] = X_train["Birthday_count"].fillna(birthday_median)
X_test["Birthday_count"] = X_test["Birthday_count"].fillna(birthday_median)

print(X_train.isna().sum().sum(), "пропусков в train")
print(X_test.isna().sum().sum(), "пропусков в test")
#%%
#кодировка категоральных признаков
#test приводим к колонкам train, неизвестные категории игнорируются

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

X_train = X_train.astype(int)
X_test = X_test.astype(int)

print("Количество признаков:", X_train.shape[1])
#%%
#кодировка классов
y_train = y_train.map({"Y": 1, "N": -1}).values
y_test = y_test.map({"Y": 1, "N": -1}).values

print(pd.Series(y_train).value_counts())
print(pd.Series(y_test).value_counts())
#%%
#стандартизация (KNN считает расстояния, поэтому масштаб важен)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#%%
#гауссово ядро
def gauss_kernel(r):
    return np.exp(-2 * r ** 2)
#%%
#KNN с окном Парзена переменной ширины
class ParzenKNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        return self

    def predict(self, X):
        dist = cdist(X, self.X)

        #если эталонов мало, k не может быть больше (число объектов - 1)
        k = min(self.k, len(self.y) - 1)

        #ширина окна - расстояние до (k+1)-го соседа
        h = np.maximum(np.sort(dist, axis=1)[:, k], 1e-10)

        #вес каждого объекта обучения, класс получает сумму весов своих объектов
        weights = gauss_kernel(dist / h[:, None])
        scores = np.stack(
            [np.sum(weights[:, self.y == c], axis=1) for c in self.classes],
            axis=1
        )

        return self.classes[np.argmax(scores, axis=1)]
#%%
#скользящий контроль LOO
def loo_risk(X, y, k_values):
    dist = cdist(X, X)
    np.fill_diagonal(dist, np.inf)
    dist_sorted = np.sort(dist, axis=1)

    risks = []
    for k in k_values:
        h = np.maximum(dist_sorted[:, k], 1e-10)
        weights = gauss_kernel(dist / h[:, None])

        score_pos = np.sum(weights[:, y == 1], axis=1)
        score_neg = np.sum(weights[:, y == -1], axis=1)

        pred = np.where(score_pos > score_neg, 1, -1)
        risks.append(np.mean(pred != y))

    return np.array(risks)


K_MAX = 30
k_values = np.arange(1, K_MAX + 1)

risks = loo_risk(X_train, y_train, k_values)

best_k = int(k_values[np.argmin(risks)])

print("k с минимальным LOO-риском:", best_k)
print("минимальный LOO-риск:", risks.min())
print("LOO-риск при k=1:", risks[0])
#%%
#график эмпирического риска
plt.figure(figsize=(8, 5))

plt.plot(k_values, risks)
plt.scatter([best_k], [risks.min()], color="red", zorder=3, label=f"лучший k = {best_k}")

plt.xlabel("k")
plt.ylabel("Эмпирический риск (LOO)")
plt.title("Эмпирический риск в зависимости от k")

plt.legend()
plt.grid()

plt.savefig(IMG_DIR / "loo_risk.png", dpi=150, bbox_inches="tight")
plt.show()
#%%
#метрики
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_metrics(name, y_true, y_pred):
    return {
        "Модель": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label=1),
        "Recall": recall_score(y_true, y_pred, pos_label=1),
        "F1-score": f1_score(y_true, y_pred, pos_label=1)
    }
#%%
#сравнение с эталонной реализацией
from sklearn.neighbors import KNeighborsClassifier

knn = ParzenKNN(best_k).fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

sk_knn = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
sk_knn.fit(X_train, y_train)
y_pred_sk = sk_knn.predict(X_test)

comparison = pd.DataFrame([
    get_metrics("KNN + окно Парзена (наш)", y_test, y_pred_knn),
    get_metrics("sklearn KNN", y_test, y_pred_sk),
])

print(comparison)
#%%
#отбор эталонов
def select_prototypes(X, y):
    N = len(y)

    dist = cdist(X, X)
    #объект не может быть эталоном сам для себя
    np.fill_diagonal(dist, np.inf)

    #первый объект каждого класса
    omega = [np.where(y == c)[0][0] for c in (1, -1)]

    #расстояние до ближайшего эталона и его класс для каждого объекта
    def nearest(omega):
        d = dist[:, omega]
        j = np.argmin(d, axis=1)
        return d[np.arange(N), j], y[omega][j]

    nd, nl = nearest(omega)
    current_ccv = np.mean(nl != y)
    start_ccv = current_ccv

    while True:
        candidates = np.setdiff1d(np.arange(N), omega)

        #если добавить кандидата, объект перейдет к нему, когда он ближе текущего эталона
        closer = dist[:, candidates] < nd[:, None]
        pred = np.where(closer, y[candidates][None, :], nl[:, None])
        ccv = np.mean(pred != y[:, None], axis=0)

        best = np.argmin(ccv)

        #останавливаемся, когда CCV больше не уменьшается
        if ccv[best] >= current_ccv:
            break

        omega.append(candidates[best])
        nd, nl = nearest(omega)
        current_ccv = ccv[best]

    return np.array(omega), start_ccv, current_ccv


omega, start_ccv, final_ccv = select_prototypes(X_train, y_train)

print("Объектов в обучении:", len(y_train))
print("Эталонов:", len(omega))
print("Доля эталонов:", len(omega) / len(y_train))
print("CCV:", start_ccv, "->", final_ccv)
#%%
#KNN на эталонах
knn_etalon = ParzenKNN(best_k).fit(X_train[omega], y_train[omega])
y_pred_etalon = knn_etalon.predict(X_test)

comparison_etalon = pd.DataFrame([
    get_metrics("KNN без отбора эталонов", y_test, y_pred_knn),
    get_metrics("KNN с отбором эталонов", y_test, y_pred_etalon),
])

comparison_etalon["Объектов в обучении"] = [len(y_train), len(omega)]

print(comparison_etalon)
#%%
#визуализация отбора эталонов
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
Z = pca.fit_transform(X_train)

plt.figure(figsize=(8, 6))

for c, name in ((1, "Y (владеет авто)"), (-1, "N (не владеет)")):
    m = y_train == c
    plt.scatter(Z[m, 0], Z[m, 1], s=20, alpha=0.3, label=name)

plt.scatter(Z[omega, 0], Z[omega, 1], facecolors="none", edgecolors="black",
            s=100, linewidths=1.5, label="Эталоны")

plt.xlabel("Главная компонента 1")
plt.ylabel("Главная компонента 2")
plt.title("Отобранные эталоны")

plt.legend()
plt.grid()

plt.savefig(IMG_DIR / "etalons.png", dpi=150, bbox_inches="tight")
plt.show()
#%%
#диаграмма сравнения качества всех трех вариантов
metric_names = ["Accuracy", "Precision", "Recall", "F1-score"]

rows = [comparison.iloc[0], comparison.iloc[1], comparison_etalon.iloc[1]]
labels = ["KNN + окно Парзена", "sklearn KNN", "KNN + окно Парзена на эталонах"]

x = np.arange(len(metric_names))
width = 0.25

plt.figure(figsize=(9, 6))

for i, (row, label) in enumerate(zip(rows, labels)):
    plt.bar(x + (i - 1) * width, [row[m] for m in metric_names], width, label=label)

plt.xticks(x, metric_names)
plt.ylabel("Значение метрики")
plt.ylim(0, 1)
plt.title("Сравнение качества классификации на тесте")

plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.savefig(IMG_DIR / "metrics_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
#%%

#%%

#%%
