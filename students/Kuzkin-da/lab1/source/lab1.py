#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Credit_card.csv")

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
#заполнение пропусков

X["GENDER"] = X["GENDER"].fillna(
    X["GENDER"].mode()[0]
)
#профессия
X["Type_Occupation"] = X["Type_Occupation"].fillna(
    "Unknown"
)
#годовой
X["Annual_income"] = X["Annual_income"].fillna(
    X["Annual_income"].median()
)

X["Birthday_count"] = X["Birthday_count"].fillna(
    X["Birthday_count"].median()
)

print(X.isna().sum().sum(), "пропусков")
#%%
#кодировка категоральных признаков

X = pd.get_dummies(X, drop_first=True)
X = X.astype(int)

print("Количество признаков:", X.shape[1])
#%%
from sklearn.model_selection import train_test_split

#x-признак
#y-правильный ответ

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=32,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape) #205
#%%
#кодировка классов

y_train = y_train.map({"Y": 1, "N": -1})
y_test = y_test.map({"Y": 1, "N": -1})

print(y_train.value_counts())
print(y_test.value_counts())
#%%
#стандартизация
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#%%
#математические функции
def predict(X, w, b):
    scores = X @ w + b
    return np.where(scores >= 0, 1, -1)


def margin(X, y, w, b):
    return y * (X @ w + b)


def quadratic_loss(X, y, w, b):
    predictions = X @ w + b
    return np.mean((y - predictions) ** 2) / 2
#%%
#начальная ошибка при 0
w = np.zeros(X_train.shape[1])
b = 0.0

loss = quadratic_loss(
    X_train,
    y_train.values,
    w,
    b
)

print("Начальная ошибка:", loss)
#%%
#градиент
def gradient(X, y, w, b):
    N = len(y)

    predictions = X @ w + b
    errors = predictions - y

    grad_w = (X.T @ errors) / N
    grad_b = np.mean(errors)

    return grad_w, grad_b


grad_w, grad_b = gradient(
    X_train,
    y_train.values,
    w,
    b
)

print("Размер градиента:", grad_w.shape)
print("Градиент b:", grad_b)
#%%
#обучение спуском
def train_gradient_descent(
    X,
    y,
    learning_rate=0.01,
    epochs=1000
):
    N, n_features = X.shape

    w = np.zeros(n_features)
    b = 0.0

    losses = []

    for epoch in range(epochs):
        grad_w, grad_b = gradient(X, y, w, b)

        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b

        loss = quadratic_loss(X, y, w, b)
        losses.append(loss)

    return w, b, losses
#%%
#запуск
w_gd, b_gd, losses_gd = train_gradient_descent(
    X_train,
    y_train.values,
    learning_rate=0.01,
    epochs=1000
)

print("Начальная ошибка:", losses_gd[0])
print("Конечная ошибка:", losses_gd[-1])
#%%
#L2
def gradient_l2(X, y, w, b, l2=0.001):
    N = len(y)

    predictions = X @ w + b
    errors = predictions - y

    grad_w = (X.T @ errors) / N
    grad_b = np.mean(errors)

    grad_w += l2 * w

    return grad_w, grad_b
#%%
def recurrent_quality(old_quality, new_loss, n):
    return (
        ((n - 1) / n) * old_quality
        + (1 / n) * new_loss
    )
#%%
#стохастический спуск
def train_sgd_momentum(
    X,
    y,
    learning_rate=0.001,
    momentum=0.9,
    l2=0.001,
    epochs=100,
    batch_size=32,
    random_state=42
):
    rng = np.random.default_rng(random_state)

    N, n_features = X.shape

    w = np.zeros(n_features)
    b = 0.0

    velocity_w = np.zeros(n_features)
    velocity_b = 0.0

    losses = []

    #история рекуррентной оценки
    q_history = []
    q = 0.0

    for epoch in range(epochs):

        indices = rng.permutation(N)

        for start in range(0, N, batch_size):

            batch_indices = indices[start:start + batch_size]

            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            grad_w, grad_b = gradient_l2(
                X_batch,
                y_batch,
                w,
                b,
                l2
            )

            velocity_w = (
                momentum * velocity_w
                + learning_rate * grad_w
            )

            velocity_b = (
                momentum * velocity_b
                + learning_rate * grad_b
            )

            w = w - velocity_w
            b = b - velocity_b

        #ошибка после эпохи
        loss = quadratic_loss(X, y, w, b)
        losses.append(loss)

        #рекуррентная оценка
        q = recurrent_quality(
            q,
            loss,
            epoch + 1
        )

        q_history.append(q)

    return w, b, losses, q_history
#%%
#обучение
w_sgd, b_sgd, losses_sgd, q_sgd = train_sgd_momentum(
    X_train,
    y_train.values,
    learning_rate=0.001,
    momentum=0.9,
    l2=0.001,
    epochs=100,
    batch_size=32,
    random_state=42
)

print("Начальная ошибка:", losses_sgd[0])
print("Конечная ошибка:", losses_sgd[-1])
print("Конечная рекуррентная оценка q:", q_sgd[-1])
#%%
#оптимальный шаг для обучения
def optimal_step(X, grad_w):
    N = X.shape[0]

    H_grad = (X.T @ (X @ grad_w)) / N

    numerator = np.sum(grad_w ** 2)
    denominator = np.sum(grad_w * H_grad)

    if denominator == 0:
        return 0.01

    return numerator / denominator
#%%
#наискорейший гд
def train_steepest_descent(X, y, epochs=1000):
    N, n_features = X.shape

    w = np.zeros(n_features)
    b = 0.0

    losses = []
    learning_rates = []

    for epoch in range(epochs):

        grad_w, grad_b = gradient(X, y, w, b)

        lr = optimal_step(X, grad_w)

        w = w - lr * grad_w
        b = b - lr * grad_b

        loss = quadratic_loss(X, y, w, b)

        losses.append(loss)
        learning_rates.append(lr)

    return w, b, losses, learning_rates
#%%
#запуск
w_steepest, b_steepest, losses_steepest, learning_rates_steepest = (
    train_steepest_descent(
        X_train,
        y_train.values,
        epochs=1000
    )
)

print("Начальная ошибка:", losses_steepest[0])
print("Конечная ошибка:", losses_steepest[-1])
print("Первые 10 шагов:", learning_rates_steepest[:10])
#%%
#инит весов
def correlation_initialization(X, y):
    correlations = []

    for j in range(X.shape[1]):

        if np.std(X[:, j]) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(X[:, j], y)[0, 1]

        correlations.append(corr)

    w = np.array(correlations)
    b = 0.0

    return w, b
#%%
#обучение
def train_gradient_descent_with_init(
    X,
    y,
    w,
    b,
    learning_rate=0.01,
    epochs=1000
):
    losses = []

    for epoch in range(epochs):

        grad_w, grad_b = gradient(
            X,
            y,
            w,
            b
        )

        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b

        losses.append(
            quadratic_loss(X, y, w, b)
        )

    return w, b, losses
#%%
w_corr, b_corr = correlation_initialization(
    X_train,
    y_train.values
)

w_corr_gd, b_corr_gd, losses_corr_gd = (
    train_gradient_descent_with_init(
        X_train,
        y_train.values,
        w_corr,
        b_corr,
        learning_rate=0.01,
        epochs=1000
    )
)

print("Конечная ошибка:", losses_corr_gd[-1])
#%%
#обучение со случайными весами
def train_random_init(
    X,
    y,
    learning_rate=0.01,
    epochs=1000,
    random_state=42
):
    rng = np.random.default_rng(random_state)

    w = rng.normal(0, 0.1, X.shape[1])
    b = rng.normal(0, 0.1)

    losses = []

    for epoch in range(epochs):

        grad_w, grad_b = gradient(
            X,
            y,
            w,
            b
        )

        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

        losses.append(
            quadratic_loss(X, y, w, b)
        )

    return w, b, losses
#%%
##%%
w_random, b_random, losses_random = train_random_init(
    X_train,
    y_train.values,
    random_state=42
)

print("Начальная ошибка:", losses_random[0])
print("Конечная ошибка:", losses_random[-1])
#%%
#многократный запуск гс со случайными весами
def multistart_gradient_descent(
    X,
    y,
    n_starts=10,
    learning_rate=0.01,
    epochs=1000
):
    best_w = None
    best_b = None
    best_losses = None
    best_loss = float("inf")

    for seed in range(n_starts):

        w, b, losses = train_random_init(
            X,
            y,
            learning_rate=learning_rate,
            epochs=epochs,
            random_state=seed
        )

        if losses[-1] < best_loss:
            best_loss = losses[-1]
            best_w = w
            best_b = b
            best_losses = losses

    return best_w, best_b, best_losses
#%%
#запуск
w_multi, b_multi, losses_multi = multistart_gradient_descent(
    X_train,
    y_train.values,
    n_starts=10,
    learning_rate=0.01,
    epochs=1000
)

print("Лучшая конечная ошибка:", losses_multi[-1])
#%%
#сорт по модулю
margins = margin(
    X_train,
    y_train.values,
    w_multi,
    b_multi
)

order_by_margin = np.argsort(
    np.abs(margins)
)

print("первые 10 индексов:")
print(order_by_margin[:10])

print("абсолютные маржи:")
print(
    np.abs(
        margins[order_by_margin[:10]]
    )
)
#%%
#обучение сгд с учетом сортировки по марже
def train_sgd_with_order(
    X,
    y,
    order,
    learning_rate=0.001,
    epochs=100
):
    N, n_features = X.shape

    w = np.zeros(n_features)
    b = 0.0

    losses = []

    for epoch in range(epochs):

        for i in order:

            X_i = X[i:i + 1]
            y_i = y[i:i + 1]

            grad_w, grad_b = gradient(
                X_i,
                y_i,
                w,
                b
            )

            w -= learning_rate * grad_w
            b -= learning_rate * grad_b

        losses.append(
            quadratic_loss(X, y, w, b)
        )

    return w, b, losses
#%%
#запуск обучения
w_margin, b_margin, losses_margin = train_sgd_with_order(
    X_train,
    y_train.values,
    order_by_margin,
    learning_rate=0.001,
    epochs=100
)

print("Конечная ошибка:", losses_margin[-1])
#%%
#обучение в случайном порядке
rng = np.random.default_rng(32)

random_order = rng.permutation(
    len(y_train)
)

w_random_order, b_random_order, losses_random_order = (
    train_sgd_with_order(
        X_train,
        y_train.values,
        random_order,
        learning_rate=0.001,
        epochs=100
    )
)

print("Конечная ошибка:", losses_random_order[-1])
#%%
#метрики
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

y_pred_steepest = predict(
    X_test,
    w_steepest,
    b_steepest
)

print("Accuracy:", accuracy_score(
    y_test,
    y_pred_steepest
))

print("Precision:", precision_score(
    y_test,
    y_pred_steepest,
    pos_label=1
))

print("Recall:", recall_score(
    y_test,
    y_pred_steepest,
    pos_label=1
))

print("F1-score:", f1_score(
    y_test,
    y_pred_steepest,
    pos_label=1
))
#%%
#сравнение моделей
models = {
    "Steepest GD": (w_steepest, b_steepest),
    "SGD + Momentum + L2": (w_sgd, b_sgd),
    "Margin order": (w_margin, b_margin),
    "Random order": (w_random_order, b_random_order),
    "Multistart": (w_multi, b_multi),
    "Correlation init": (w_corr_gd, b_corr_gd)
}

results = []

for name, (w_model, b_model) in models.items():

    y_pred = predict(
        X_test,
        w_model,
        b_model
    )

    results.append({
        "NAme": name,
        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),
        "Precision": precision_score(
            y_test,
            y_pred,
            pos_label=1
        ),
        "Recall": recall_score(
            y_test,
            y_pred,
            pos_label=1
        ),
        "F1": f1_score(
            y_test,
            y_pred,
            pos_label=1
        )
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1",
    ascending=False
).reset_index(drop=True)

print(results_df)
#%%
#матрица ошибок
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    y_pred_steepest,
    labels=[-1, 1]
)

print("матрица ошибок:")
print(cm)

plt.figure(figsize=(5, 4))

plt.imshow(cm)

plt.xticks(
    [0, 1],
    ["N", "Y"]
)

plt.yticks(
    [0, 1],
    ["N", "Y"]
)

plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.title("Матрица ошибок")

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.show()
#%%
#анализ отступов на тестовых данных
margins_test = margin(
    X_test,
    y_test.values,
    w_steepest,
    b_steepest
)

print("Минимальный отступ:", margins_test.min())
print("Максимальный отступ:", margins_test.max())
print("Средний отступ:", margins_test.mean())

print(
    "Отрицательных отступов:",
    np.sum(margins_test < 0)
)

print(
    "Положительных отступов:",
    np.sum(margins_test > 0)
)

print(
    "Объектов с отступом < 0.1:",
    np.sum(np.abs(margins_test) < 0.1)
)

print(
    "Объектов с отступом < 0.2:",
    np.sum(np.abs(margins_test) < 0.2)
)

print(
    "Объектов с отступом < 0.5:",
    np.sum(np.abs(margins_test) < 0.5)
)
#%%
#график отступов
plt.figure(figsize=(10, 5))

plt.scatter(
    range(len(margins_test)),
    margins_test,
    s=15
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Номер объекта")
plt.ylabel("Отступ")
plt.title("Отступы объектов тестовой выборки")

plt.show()
#%%
#auc
from sklearn.metrics import roc_curve, roc_auc_score

scores = X_test @ w_steepest + b_steepest

y_test_binary = (
    y_test.values == 1
).astype(int)

fpr, tpr, thresholds = roc_curve(
    y_test_binary,
    scores
)

auc = roc_auc_score(
    y_test_binary,
    scores
)

print("AUC:", auc)
#%%
#roc
plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Steepest GD (AUC = {auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    "--",
    label="Случайный классификатор"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривая")

plt.legend()
plt.grid()

plt.show()
#%%
#сравнение с эталоном
from sklearn.linear_model import RidgeClassifier

L2 = 0.001

ridge_alpha = (
    L2 * len(y_train) / 2.0
)

print("alpha:", ridge_alpha)

ridge = RidgeClassifier(
    alpha=ridge_alpha
)

ridge.fit(
    X_train,
    y_train.values
)

y_pred_ridge = ridge.predict(
    X_test
)

print("Accuracy:", accuracy_score(
    y_test,
    y_pred_ridge
))

print("Precision:", precision_score(
    y_test,
    y_pred_ridge
))

print("Recall:", recall_score(
    y_test,
    y_pred_ridge
))

print("F1-score:", f1_score(
    y_test,
    y_pred_ridge
))
#%%
comparison = pd.DataFrame([
    [
        "Steepest Gradient Descent",
        accuracy_score(y_test, predict(X_test, w_steepest, b_steepest)),
        precision_score(y_test, predict(X_test, w_steepest, b_steepest)),
        recall_score(y_test, predict(X_test, w_steepest, b_steepest)),
        f1_score(y_test, predict(X_test, w_steepest, b_steepest))
    ],
    [
        "RidgeClassifier",
        accuracy_score(y_test, y_pred_ridge),
        precision_score(y_test, y_pred_ridge),
        recall_score(y_test, y_pred_ridge),
        f1_score(y_test, y_pred_ridge)
    ]
], columns=[
    "Модель",
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score"
])

print(comparison)