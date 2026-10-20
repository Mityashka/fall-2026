#%%
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

#данные из внешнего источника
DATA_URL = "https://raw.githubusercontent.com/AbhishekDataAnalyst/Credit-Card-Approval/main/Credit_card.csv"

#папка для картинок рядом с файлом
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
#ядро и решающее правило

#гауссово ядро
def gauss_kernel(r):
    return np.exp(-0.5 * r ** 2)


#веса классов для соседей, которые уже отсортированы по расстоянию
#dist_sorted - расстояния до соседей по возрастанию, labels_sorted - их классы
def parzen_scores(dist_sorted, labels_sorted, k):
    #ширина окна - расстояние до (k+1)-го соседа (переменная для каждого объекта)
    h = np.maximum(dist_sorted[:, k], 1e-12)

    #веса k ближайших соседей
    w = gauss_kernel(dist_sorted[:, :k] / h[:, None])
    labels = labels_sorted[:, :k]

    score_pos = np.sum(w * (labels == 1), axis=1)
    score_neg = np.sum(w * (labels == -1), axis=1)

    return score_pos, score_neg


#выбор класса, при равенстве берем более частый класс обучения
def decide(score_pos, score_neg, default):
    return np.where(
        score_pos > score_neg,
        1,
        np.where(score_pos < score_neg, -1, default)
    )


#отступ: (вес своего класса - вес чужого) / сумма весов, лежит в [-1, 1]
def margin_from_scores(y, score_pos, score_neg):
    own = np.where(y == 1, score_pos, score_neg)
    other = np.where(y == 1, score_neg, score_pos)
    return (own - other) / (own + other)
#%%
#KNN с окном Парзена переменной ширины
class ParzenKNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.default = 1 if np.sum(y) >= 0 else -1
        return self

    def scores(self, X):
        dist = cdist(X, self.X)
        order = np.argsort(dist, axis=1, kind="stable")

        #если эталонов мало, k не может быть больше (число объектов - 1)
        k = min(self.k, len(self.y) - 1)

        dist_sorted = np.take_along_axis(dist, order, axis=1)
        labels_sorted = self.y[order]

        return parzen_scores(dist_sorted, labels_sorted, k)

    def predict(self, X):
        score_pos, score_neg = self.scores(X)
        return decide(score_pos, score_neg, self.default)
#%%
#скользящий контроль LOO
#каждый объект по очереди убираем из обучения и классифицируем по остальным
def loo_sorted(X):
    dist = cdist(X, X)
    #сам объект не должен быть себе соседом - ставим бесконечность, он уйдет в конец
    np.fill_diagonal(dist, np.inf)

    order = np.argsort(dist, axis=1, kind="stable")[:, :-1]
    dist_sorted = np.take_along_axis(dist, order, axis=1)

    return dist_sorted, order


def loo_risk(X, y, k_values):
    dist_sorted, order = loo_sorted(X)
    labels_sorted = y[order]
    default = 1 if np.sum(y) >= 0 else -1

    risks = []
    for k in k_values:
        score_pos, score_neg = parzen_scores(dist_sorted, labels_sorted, k)
        pred = decide(score_pos, score_neg, default)
        risks.append(np.mean(pred != y))

    return np.array(risks)


K_MAX = 200
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
#наша реализация на тесте
knn = ParzenKNN(best_k).fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_knn))
#%%
#сравнение с эталонной реализацией
from sklearn.neighbors import KNeighborsClassifier

#обычный KNN из sklearn, тот же k
sk_knn = KNeighborsClassifier(n_neighbors=best_k)
sk_knn.fit(X_train, y_train)
y_pred_sk = sk_knn.predict(X_test)


#KNN из sklearn с нашими весами: берем k+1 соседей, у последнего вес 0,
#он нужен только чтобы задать ширину окна
def parzen_weights(dist):
    h = np.maximum(dist[:, -1:], 1e-12)
    w = gauss_kernel(dist / h)
    w[:, -1] = 0
    return w


sk_parzen = KNeighborsClassifier(n_neighbors=best_k + 1, weights=parzen_weights)
sk_parzen.fit(X_train, y_train)
y_pred_sk_parzen = sk_parzen.predict(X_test)

print("Совпадение предсказаний с sklearn (окно Парзена):", np.mean(y_pred_knn == y_pred_sk_parzen))
print("Совпадение предсказаний с sklearn (обычный KNN):", np.mean(y_pred_knn == y_pred_sk))

comparison = pd.DataFrame([
    get_metrics("KNN + окно Парзена (наш)", y_test, y_pred_knn),
    get_metrics("sklearn KNN (обычный)", y_test, y_pred_sk),
    get_metrics("sklearn KNN (веса Парзена)", y_test, y_pred_sk_parzen),
])

print(comparison)
#%%
#отбор эталонов (алгоритм STOLP)
#delta - объекты с отступом меньше delta считаем шумом и выбрасываем
#max_errors - сколько ошибок на обучении допускаем, чтобы остановиться
def stolp(X, y, k, delta, max_errors):
    dist = cdist(X, X)

    #1. отступы всех объектов относительно всей выборки (сам объект исключаем)
    dist_sorted, order = loo_sorted(X)
    score_pos, score_neg = parzen_scores(dist_sorted, y[order], k)
    margins = margin_from_scores(y, score_pos, score_neg)

    #2. убираем шум
    noise = np.where(margins < delta)[0]
    clean = np.where(margins >= delta)[0]

    #3. по одному эталону от каждого класса - с максимальным отступом
    omega = []
    for c in (1, -1):
        idx = clean[y[clean] == c]
        omega.append(idx[np.argmax(margins[idx])])

    errors_history = []

    #4. добавляем эталоны, пока ошибок больше допустимого
    while True:
        rest = np.setdiff1d(clean, omega)

        #классифицируем оставшиеся объекты только по эталонам
        d = dist[np.ix_(rest, omega)]
        o = np.argsort(d, axis=1, kind="stable")
        d_sorted = np.take_along_axis(d, o, axis=1)
        labels_sorted = y[omega][o]

        k_eff = min(k, len(omega) - 1)
        sp, sn = parzen_scores(d_sorted, labels_sorted, k_eff)

        default = 1 if np.sum(y[omega]) >= 0 else -1
        pred = decide(sp, sn, default)
        rest_margins = margin_from_scores(y[rest], sp, sn)

        wrong = np.where(pred != y[rest])[0]
        errors_history.append(len(wrong))

        if len(wrong) <= max_errors:
            break

        #добавляем ошибочный объект с самым маленьким отступом
        worst = wrong[np.argmin(rest_margins[wrong])]
        omega.append(rest[worst])

    return np.array(omega), noise, margins, errors_history
#%%
#распределение отступов на обучении (по нему выбираем delta)
dist_sorted, order = loo_sorted(X_train)
score_pos, score_neg = parzen_scores(dist_sorted, y_train[order], best_k)
margins_train = margin_from_scores(y_train, score_pos, score_neg)

for d in (0.0, -0.2, -0.4, -0.6, -0.8):
    print("delta =", d, "-> шум:", np.sum(margins_train < d), "объектов")
#%%
#запуск отбора эталонов
DELTA = 0.0
MAX_ERRORS = int(0.05 * len(y_train))

omega, noise, margins_stolp, errors_history = stolp(
    X_train,
    y_train,
    k=best_k,
    delta=DELTA,
    max_errors=MAX_ERRORS
)

print("Объектов в обучении:", len(y_train))
print("Шум:", len(noise))
print("Эталонов:", len(omega))
print("Доля эталонов:", len(omega) / len(y_train))
print("Класс 1 / класс -1 среди эталонов:", np.sum(y_train[omega] == 1), "/", np.sum(y_train[omega] == -1))
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
#визуализация отбора эталонов (проекция на 2 главные компоненты)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
Z = pca.fit_transform(X_train)

is_noise = np.zeros(len(y_train), dtype=bool)
is_noise[noise] = True
is_etalon = np.zeros(len(y_train), dtype=bool)
is_etalon[omega] = True
is_other = ~is_noise & ~is_etalon

colors = {1: "tab:blue", -1: "tab:orange"}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

#до отбора
for c in (1, -1):
    m = y_train == c
    axes[0].scatter(Z[m, 0], Z[m, 1], s=15, color=colors[c], alpha=0.6,
                    label="Y (владеет авто)" if c == 1 else "N (не владеет)")
axes[0].set_title("Вся обучающая выборка")
axes[0].legend()

#после отбора
axes[1].scatter(Z[is_other, 0], Z[is_other, 1], s=15, color="lightgray", label="обычные объекты")
axes[1].scatter(Z[is_noise, 0], Z[is_noise, 1], s=40, color="black", marker="x", label="шум")
for c in (1, -1):
    m = is_etalon & (y_train == c)
    axes[1].scatter(Z[m, 0], Z[m, 1], s=60, color=colors[c], edgecolor="black",
                    label="эталоны Y" if c == 1 else "эталоны N")
axes[1].set_title("Результат отбора эталонов")
axes[1].legend()

for ax in axes:
    ax.set_xlabel("Главная компонента 1")
    ax.set_ylabel("Главная компонента 2")
    ax.grid()

plt.savefig(IMG_DIR / "etalons.png", dpi=150, bbox_inches="tight")
plt.show()
#%%
#сколько ошибок на обучении при росте числа эталонов
plt.figure(figsize=(8, 5))

plt.plot(range(2, 2 + len(errors_history)), errors_history)
plt.axhline(MAX_ERRORS, linestyle="--", color="red", label=f"допустимо ошибок = {MAX_ERRORS}")

plt.xlabel("Число эталонов")
plt.ylabel("Ошибок на обучающих объектах")
plt.title("Работа алгоритма отбора эталонов")

plt.legend()
plt.grid()

plt.savefig(IMG_DIR / "stolp_errors.png", dpi=150, bbox_inches="tight")
plt.show()