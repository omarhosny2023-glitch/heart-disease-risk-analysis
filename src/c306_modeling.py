from __future__ import annotations

from docx import Document
from docx.shared import Inches
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.tree import plot_tree

RANDOM_STATE = 42

# =========================================================
# HELPERS
# =========================================================
BEST_MODELS = {}
def _to_series(y) -> pd.Series:
    if isinstance(y, pd.Series):
        return y.copy()
    return pd.Series(y)


def _resolve_positive_label(y) -> object | None:
    """
    Return a reasonable positive label for binary classification.
    """
    y = _to_series(y).dropna()
    unique = list(pd.unique(y))

    if len(unique) != 2:
        return None

    preferred = ["Yes", "yes", 1, "1", True]
    for p in preferred:
        if p in unique:
            return p

    return unique[1]


def _make_scorer_for_y(y_train):
    pos_label = _resolve_positive_label(y_train)
    if pos_label is None:
        return make_scorer(f1_score, average="macro"), None
    return make_scorer(f1_score, pos_label=pos_label, zero_division=0), pos_label


# =========================================================
# CONFUSION MATRIX PLOT
# =========================================================
def plot_confusion_matrix(y_true, y_pred, title: str) -> None:
    """
    Plot a confusion matrix with real class labels.
    """
    y_true_s = _to_series(y_true)
    y_pred_s = _to_series(y_pred)

    labels = list(pd.unique(pd.concat([y_true_s.astype(object), y_pred_s.astype(object)], ignore_index=True)))
    cm = confusion_matrix(y_true_s, y_pred_s, labels=labels)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="black" if cm[i, j] > thresh else "white",
            )

    plt.tight_layout()
    plt.show()


# =========================================================
# ROC CURVE PLOT
# =========================================================
def plot_roc_curve(model, X_test, y_test, title="ROC Curve"):
    y_test = _to_series(y_test)

    positive_label = _resolve_positive_label(y_test)

    if not hasattr(model, "predict_proba") or positive_label is None:
        print("ROC curve cannot be plotted (no predict_proba or not binary).")
        return

    classes = list(model.classes_)
    if positive_label not in classes:
        print("Positive label not found in classes.")
        return

    pos_idx = classes.index(positive_label)
    y_score = model.predict_proba(X_test)[:, pos_idx]

    y_test_bin = (y_test == positive_label).astype(int)

    fpr, tpr, _ = roc_curve(y_test_bin, y_score)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid()
    plt.show()

# =========================================================
# MODEL EVALUATION (UPDATED FOR CROSS-VALIDATION
# =========================================================
def evaluate_model(model, X, y, title: str) -> dict:
    global BEST_MODELS
    BEST_MODELS[title] = model
    
    y = _to_series(y)
    cv = get_cv(y)

    y_pred_arr = cross_val_predict(model, X, y, cv=cv)
    y_pred = pd.Series(y_pred_arr, index=y.index)

    positive_label = _resolve_positive_label(y)

    if hasattr(model, "predict_proba") and positive_label is not None:
        y_score_matrix = cross_val_predict(model, X, y, cv=cv, method='predict_proba')
        classes = list(model.classes_)
        if positive_label in classes:
            pos_idx = classes.index(positive_label)
            y_score = y_score_matrix[:, pos_idx]
        else:
            y_score = None
    else:
        y_score = None

    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y, y_pred),
    }

    if positive_label is not None and len(pd.unique(y)) == 2:
        metrics["Precision"] = precision_score(y, y_pred, pos_label=positive_label, zero_division=0)
        metrics["Recall"] = recall_score(y, y_pred, pos_label=positive_label, zero_division=0)
        metrics["F1-score"] = f1_score(y, y_pred, pos_label=positive_label, zero_division=0)

        if y_score is not None:
            y_test_bin = (y == positive_label).astype(int)
            metrics["ROC-AUC"] = roc_auc_score(y_test_bin, y_score)
            metrics["PR-AUC"] = average_precision_score(y_test_bin, y_score)
    else:
        metrics["Precision"] = precision_score(y, y_pred, average="macro", zero_division=0)
        metrics["Recall"] = recall_score(y, y_pred, average="macro", zero_division=0)
        metrics["F1-score"] = f1_score(y, y_pred, average="macro", zero_division=0)

    print(f"\n==================== {title} ====================")
    for name, value in metrics.items():
        print(f"{name:18s}: {value:.6f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))

    print("\nClassification Report:")
    print(classification_report(y, y_pred, zero_division=0))

    plot_confusion_matrix(y, y_pred, f"Confusion Matrix - {title}")
    plot_roc_curve(model, X, y, f"ROC Curve - {title}")
    return metrics


# =========================================================
# SAFE CROSS-VALIDATION
# =========================================================
def get_cv(y_train: pd.Series) -> StratifiedKFold:
    """
    Create a safe stratified CV object.
    """
    y_train = _to_series(y_train)
    min_class_count = int(y_train.value_counts().min())

    if min_class_count < 2:
        raise ValueError(
            "Not enough samples in the smallest class to run stratified cross-validation."
        )

    n_splits = min(3, min_class_count)

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


# =========================================================
# GRID SEARCH
# =========================================================

def tune_knn(X_train, y_train):
    model = KNeighborsClassifier(metric='hamming')

    param_grid = {
        "n_neighbors": [3, 5, 7, 9], 
        "weights": ["uniform", "distance"]
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=get_cv(y_train),
        scoring=_make_scorer_for_y(y_train)[0],
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid


def tune_naive_bayes(X_train, y_train):
    model = CategoricalNB()

    param_grid = {
        "alpha": [0.1, 0.5, 1.0, 2.0]
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=get_cv(y_train),
        scoring=_make_scorer_for_y(y_train)[0],
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    return grid


# =========================================================
# FIT + EVALUATE
# =========================================================
def fit_and_evaluate_model(grid_search: GridSearchCV, X_test, y_test, title: str) -> dict:
    """
    Fit time is measured by GridSearchCV internally + evaluation on test set.
    """
    fit_time = grid_search.refit_time_ if hasattr(grid_search, "refit_time_") else 0
    best_model = grid_search.best_estimator_

    metrics = evaluate_model(best_model, X_test, y_test, title)
    metrics["Training Time (sec)"] = fit_time
    return metrics


# =========================================================
# COMPARISON
# =========================================================
def print_model_parameters(knn_grid, nb_grid, stage=""):
    print(f"\n==================== MODEL PARAMETERS ({stage}) ====================\n")

    # -------- KNN --------
    best_knn = knn_grid.best_params_
    
    print("(KNN Model)")
    print(f"1. Best number of neighbors (k) = {best_knn.get('n_neighbors', 'N/A')}")
    print(f"2. Weight type used = {best_knn.get('weights', 'N/A')}")
    print("3. Distance metric = hamming")

    # -------- NAIVE BAYES --------
    best_nb = nb_grid.best_params_
    
    print("(Naïve Bayes Model)")
    print("1. Type of NB used = CategoricalNB")
    print(f"2. Best alpha value = {best_nb.get('alpha', 'N/A')}")

def compare_knn_and_nb(X_train, y_train, X_test, y_test, balance_fn=None, target_col=None, balance_kwargs=None, show_before=True) -> pd.DataFrame:
    results = []

    if show_before:
        print("\n===== BEFORE BALANCING =====")
        knn_grid_before = tune_knn(X_train, y_train)
        knn_metrics = evaluate_model(knn_grid_before.best_estimator_, X_test, y_test, "KNN (Before)")
        knn_metrics["Model"] = "KNN (Before)"
        results.append(knn_metrics)

        nb_grid_before = tune_naive_bayes(X_train, y_train)
        nb_metrics = evaluate_model(nb_grid_before.best_estimator_, X_test, y_test, "Naive Bayes (Before)")
        nb_metrics["Model"] = "Naive Bayes (Before)"
        results.append(nb_metrics)
        print_model_parameters(knn_grid_before, nb_grid_before, stage="Before Balancing")

    if balance_fn is not None and target_col is not None:
        print("\n===== AFTER BALANCING =====")
        train_df = pd.concat([X_train, y_train], axis=1)
        train_balanced = balance_fn(train_df, target_col=target_col, **(balance_kwargs or {}))
        X_train_bal = train_balanced.drop(columns=[target_col])
        y_train_bal = train_balanced[target_col]

        knn_grid_after = tune_knn(X_train_bal, y_train_bal)
        knn_metrics = evaluate_model(knn_grid_after.best_estimator_, X_test, y_test, "KNN (After)")
        knn_metrics["Model"] = "KNN (After)"
        results.append(knn_metrics)

        nb_grid_after = tune_naive_bayes(X_train_bal, y_train_bal)
        nb_metrics = evaluate_model(nb_grid_after.best_estimator_, X_test, y_test, "Naive Bayes (After)")
        nb_metrics["Model"] = "Naive Bayes (After)"
        results.append(nb_metrics)
        print_model_parameters(knn_grid_after, nb_grid_after, stage="After Balancing")

    cols = ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1-score"]
    if not results:
        return pd.DataFrame(columns=["Model"] + cols).set_index("Model")

    results_df = pd.DataFrame(results).set_index("Model")
    print(results_df[cols].round(4))
    return results_df


#=======================================================================================

# =========================================================
# DECISION TREE + ENSEMBLE METHODS
# =========================================================

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier

# =========================================================
# PLOT DECISION TREE
# =========================================================

def plot_decision_tree_model(
    tree_model,
    feature_names,
    class_names=None,
    title="Decision Tree",
    max_depth=3
):

    plt.figure(figsize=(22, 10))

    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=8
    )

    plt.title(title)

    plt.tight_layout()

    plt.show()

# --------------------------------------------------------------------

def _make_bagging(base_estimator=None, **kwargs):

    try:
        if base_estimator is None:
            return BaggingClassifier(**kwargs)

        return BaggingClassifier(
            estimator=base_estimator,
            **kwargs
        )

    except TypeError:

        return BaggingClassifier(
            base_estimator=base_estimator,
            **kwargs
        )


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def feature_importance_table(model, feature_names):

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance (%)": model.feature_importances_ * 100
    })

    importance_df = importance_df.sort_values(
        by="Importance (%)",
        ascending=False
    )

    return importance_df


def plot_feature_importance(df_imp, title):

    top = df_imp.head(10)

    plt.figure(figsize=(10, 6))

    plt.barh(
        top["Feature"],
        top["Importance (%)"]
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Importance (%)")
    plt.title(title)

    plt.tight_layout()
    plt.show()


# =========================================================
# PARAMETER PRINT
# =========================================================

def print_best_params(search, stage):

    print(f"\n==================== {stage} PARAMETERS ====================")

    for k, v in search.best_params_.items():
        print(f"{k} = {v}")


# =========================================================
# DECISION TREE
# =========================================================

def tune_decision_tree(X_train, y_train):

    model = DecisionTreeClassifier(
        random_state=RANDOM_STATE
    )

    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 7],
        "min_samples_leaf": [5, 10]
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=get_cv(y_train),
        scoring=_make_scorer_for_y(y_train)[0],
        n_jobs=1
    )

    grid.fit(X_train, y_train)

    return grid


# =========================================================
# BAGGING
# =========================================================

def tune_bagging(X_train, y_train):

    model = _make_bagging(
        base_estimator=DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        random_state=RANDOM_STATE
    )

    param_grid = {
        "n_estimators": [50, 100]
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=get_cv(y_train),
        scoring=_make_scorer_for_y(y_train)[0],
        n_jobs=1
    )

    grid.fit(X_train, y_train)

    return grid


# =========================================================
# RANDOM FOREST
# =========================================================

def tune_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        random_state=RANDOM_STATE
    )

    param_grid = {
        "n_estimators": [100],
        "max_depth": [None, 10]
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=get_cv(y_train),
        scoring=_make_scorer_for_y(y_train)[0],
        n_jobs=1
    )

    grid.fit(X_train, y_train)

    return grid


# =========================================================
# MAIN COMPARISON
# =========================================================
def compare_tree_and_ensemble(
    X_train,
    y_train,
    X_test,
    y_test,
    balance_fn=None,
    target_col=None,
    balance_kwargs=None,
    show_before=True
):
    results = []

    def run_models(X_tr, y_tr, label):
        # ======================
        # DECISION TREE
        # ======================
        tree_grid = tune_decision_tree(X_tr, y_tr)
        tree_model = tree_grid.best_estimator_
        plot_decision_tree_model(
            tree_model,
            feature_names=X_tr.columns,
            class_names=["Low", "High"],
            title=f"Decision Tree ({label})",
            max_depth=3
        )
        tree_metrics = evaluate_model(
            tree_model,
            X_test,
            y_test,
            f"Decision Tree ({label})"
        )
        tree_metrics["Model"] = f"Decision Tree ({label})"
        results.append(tree_metrics)
        print_best_params(tree_grid, f"Decision Tree ({label})")

        # ======================
        # BAGGING
        # ======================
        bag_grid = tune_bagging(X_tr, y_tr)
        bag_model = bag_grid.best_estimator_
        bag_metrics = evaluate_model(
            bag_model,
            X_test,
            y_test,
            f"Bagging ({label})"
        )
        bag_metrics["Model"] = f"Bagging ({label})"
        results.append(bag_metrics)
        print_best_params(bag_grid, f"Bagging ({label})")

        # ======================
        # RANDOM FOREST
        # ======================
        rf_grid = tune_random_forest(X_tr, y_tr)
        rf_model = rf_grid.best_estimator_
        rf_metrics = evaluate_model(
            rf_model,
            X_test,
            y_test,
            f"Random Forest ({label})"
        )
        rf_metrics["Model"] = f"Random Forest ({label})"
        results.append(rf_metrics)
        print_best_params(rf_grid, f"Random Forest ({label})")

        # ======================
        # FEATURE IMPORTANCE
        # ======================
        feature_names = list(X_tr.columns)
        tree_imp = feature_importance_table(tree_model, feature_names)
        rf_imp = feature_importance_table(rf_model, feature_names)

        print("\n========== DECISION TREE FEATURE IMPORTANCE ==========")
        print(tree_imp.round(3).to_string(index=False))

        print("\n========== RANDOM FOREST FEATURE IMPORTANCE ==========")
        print(rf_imp.round(3).to_string(index=False))

        plot_feature_importance(tree_imp, f"Decision Tree Feature Importance ({label})")
        plot_feature_importance(rf_imp, f"Random Forest Feature Importance ({label})")

    # =====================================================
    # BEFORE BALANCING
    # =====================================================
    if show_before:
        print("\n===== BEFORE BALANCING =====")
        run_models(X_train, y_train, "Before")

    # =====================================================
    # AFTER BALANCING
    # =====================================================
    if balance_fn is not None and target_col is not None:
        print("\n===== AFTER BALANCING =====")
        train_df = pd.concat([X_train, y_train], axis=1)
        balanced_df = balance_fn(
            train_df,
            target_col=target_col,
            **(balance_kwargs or {})
        )
        X_bal = balanced_df.drop(columns=[target_col])
        y_bal = balanced_df[target_col]
        run_models(X_bal, y_bal, "After")

    # =====================================================
    # FINAL TABLE
    # =====================================================
    cols = ["Model", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1-score"]
    
    if not results:
        return pd.DataFrame(columns=cols)

    results_df = pd.DataFrame(results)
    print("\n================ FINAL TABLE ================")
    print(results_df[cols].round(4).to_string(index=False))

    return results_df

def export_and_plot_roc(before_table, after_table, X, y):
    print("\n" + "=" * 60)
    print("EXPORTING RESULTS & GENERATING ROC CURVE")
    print("=" * 60)

    excel_filename = "final_models_comparison.xlsx"
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        before_table.to_excel(writer, sheet_name='Before Balancing', index=False)
        after_table.to_excel(writer, sheet_name='After Balancing', index=False)
    
    print(f"Final results for all models successfully saved to {excel_filename}")

    plt.figure(figsize=(10, 8))
    
    cv = get_cv(y)
    
    models_to_plot = {
        'KNN': BEST_MODELS.get('KNN (Before)'),
        'Naive Bayes': BEST_MODELS.get('Naive Bayes (Before)'),
        'Decision Tree': BEST_MODELS.get('Decision Tree (Before)'),
        'Random Forest': BEST_MODELS.get('Random Forest (Before)'),
        'Bagging': BEST_MODELS.get('Bagging (Before)')
    }

    from sklearn.model_selection import cross_val_predict

    for name, model in models_to_plot.items():
        if model is None:
            continue 
            
        y_proba_cv = cross_val_predict(model, X, y, cv=cv, method='predict_proba')
        
        model.fit(X, y)       
        classes_list = list(model.classes_)
        
        if 'diseased' in classes_list:
            pos_class_index = classes_list.index('diseased')     
            y_scores = y_proba_cv[:, pos_class_index]
            
            fpr, tpr, _ = roc_curve(y, y_scores, pos_label='diseased')
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curve for All Models (Cross-Validated)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    image_filename = "combined_roc_curve.png"
    plt.savefig(image_filename, dpi=300, bbox_inches='tight')
    print(f"ROC curve plot successfully saved to {image_filename}")

### Export all remaining plots (including ROC) to a Word document ###

_word_doc = None
_plot_counter = 0
_original_show = None

def init_word_plot_capture():
    global _word_doc, _plot_counter, _original_show
    
    print("\n" + "=" * 60)
    print("INITIALIZING WORD DOCUMENT HOOK")
    print("=" * 60)

    _word_doc = Document()
    _word_doc.add_heading('Modeling Plots Report', 0)
    _plot_counter = 0
    _original_show = plt.show

    def hooked_show(*args, **kwargs):
        global _plot_counter
        
        for n in plt.get_fignums():
            fig = plt.figure(n)
            img_path = f"temp_captured_{_plot_counter}.png"
            fig.savefig(img_path, dpi=300, bbox_inches='tight')
            
            _word_doc.add_picture(img_path, width=Inches(6))
            
            if os.path.exists(img_path):
                os.remove(img_path)
            _plot_counter += 1
        
        _original_show(*args, **kwargs)
        plt.close('all')

    plt.show = hooked_show

def save_captured_plots_to_word(filename="Additional_Plots_Report.docx"):
    global _word_doc, _plot_counter
    
    if _word_doc is not None:
        _word_doc.save(filename)
        print(f"\n[INFO] Successfully captured {_plot_counter} plots into {filename}!")
    else:
        print("\n[ERROR] Word document hook was not initialized.")
