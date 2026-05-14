"""Модуль обучения и оценки моделей."""

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report


def evaluate_model(model, x_train, y_train, x_val, y_val, name="Model"):
    """Обучает модель и выводит метрики на train и val."""
    model.fit(x_train, y_train)

    y_pred_train = model.predict(x_train)
    y_pred_val = model.predict(x_val)

    print(f"=== {name} ===")
    print(f"{'Метрика':<15} {'Train':>8} {'Val':>8}")
    print(f"{'Accuracy':<15} {accuracy_score(y_train, y_pred_train):>8.4f} {accuracy_score(y_val, y_pred_val):>8.4f}")
    print(f"{'F1-score':<15} {f1_score(y_train, y_pred_train):>8.4f} {f1_score(y_val, y_pred_val):>8.4f}")
    print(f"{'ROC-AUC':<15} {roc_auc_score(y_train, y_pred_train):>8.4f} {roc_auc_score(y_val, y_pred_val):>8.4f}")
    print("\nClassification Report (Val):")
    print(classification_report(y_val, y_pred_val))

    return model