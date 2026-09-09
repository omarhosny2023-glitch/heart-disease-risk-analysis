import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import c306_data_understanding as du
import c306_data_preprocess as dp
import c306_modeling as modeling
# =========================================================
# Heart DATASET MAPPING
# =========================================================

def classify_variables(df):
    if df is None or df.empty:
        return [], [], []

    # Heart Disease features
    numeric_vars = ['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']
    ordinal_vars = ["ST_Slope"]
    categorical_vars = ['Sex','ChestPainType','FastingBS','RestingECG','ExerciseAngina','HeartDisease']

    numeric_vars = [c for c in numeric_vars if c in df.columns]
    ordinal_vars = [c for c in ordinal_vars if c in df.columns]
    categorical_vars = [c for c in categorical_vars if c in df.columns]

    return numeric_vars, ordinal_vars, categorical_vars

du.classify_variables = classify_variables

# =========================================================
# HELPERS
# =========================================================
def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_list(title, items):
    print(f"\n{title}")
    if items:
        for i in items:
            print(f"  - {i}")
    else:
        print("  - None")


def print_distribution(y, title):
    print_section(title)
    dist = y.value_counts(normalize=True) * 100
    for k, v in dist.items():
        print(f"  Class {k}: {v:.2f}%")


def log_stage(name, df, count_outliers=True):
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    print(f"\n[{name}] shape = {df.shape}", flush=True)

    if count_outliers:
        num_cols = df.select_dtypes(include="number").columns

        outlier_mask = pd.DataFrame(False, index=df.index, columns=num_cols)

        for col in num_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outlier_mask[col] = (df[col] < lower) | (df[col] > upper)

        total_outliers = int(outlier_mask.any(axis=1).sum())

        print(f"Missing = {missing} | Duplicates = {duplicates} | Outliers = {total_outliers}", flush=True)
    else:
        print(f"Missing = {missing} | Duplicates = {duplicates}", flush=True)


# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv("heart.csv")

target = "HeartDisease"
if target not in df.columns:
    raise ValueError("Target column not found")

# ALWAYS split after
y = df[target].copy()
X = df.drop(columns=[target]).copy()

# X["chas"] = X["chas"].astype("category")
# X["rad"] = X["rad"].astype("category")

# =========================================================
# EDA
# =========================================================
print_section("EDA REPORT")

with PdfPages("heart disease quality report.pdf") as pdf:
    overview = du.data_overview(df, pdf)
    du.univariate_plot(df, pdf)
    
    du.relations_num(df, vars="all", chart_type="heatmap", pdf=pdf)
    du.relations_num(df, vars="all", chart_type="pair", pdf=pdf)
    du.correlation_num_ordinal(df, pdf)
    
    cats = df.select_dtypes(include=["object", "category"]).columns
    for i in range(len(cats) - 1):
        du.relations_cat(df, cats[i], cats[i + 1], pdf=pdf)

    du.relations_cat_num(df, "all", "all", "box", pdf)
    du.relations_cat_num(df, "all", "all", "violin", pdf)


# =========================================================
# PREPROCESSING
# =========================================================
print_section("PREPROCESSING")

numeric_vars, ordinal_vars, categorical_vars = du.classify_variables(df)

X_clean = dp.delete_duplicates(X)
log_stage("After delete_duplicates", X_clean)

X_clean = dp.repair_missing(X_clean, method="median")
log_stage("After repair_missing", X_clean)

dp.print_outlier_placeholder(X_clean)

X_clean = dp.repair_outliers(X_clean, detection="IQR", method="cap")
log_stage("After repair_outliers", X_clean)

# remove duplicates again
X_clean = dp.delete_duplicates(X_clean)
log_stage("After final deduplication", X_clean)

dp.print_data_quality_summary(X_clean, "AFTER PREPROCESSING")

X_clean = dp.fix_skewness(X_clean)
log_stage("After fix_skewness", X_clean)

X_clean = dp.scale_numeric(X_clean)
log_stage("After scale_numeric", X_clean, count_outliers=False)

# ===== MERGE BACK =====
df_clean = pd.concat([X_clean, y.loc[X_clean.index]], axis=1)


print("\nOriginal shape:", df.shape, flush=True)
print("Cleaned shape:", df_clean.shape, flush=True)

# =========================================================
# FEATURE SELECTION
# =========================================================
print_section("FEATURE SELECTION")


# 1) FILTER
df_fs = dp.filter_selection(df_clean, target)
print("After filter selection:", df_fs.shape, flush=True)

# 2) COLLINEARITY (numeric only)
df_fs = dp.reduce_collinearity(df_fs, target)
print("After reduce collinearity:", df_fs.shape, flush=True)

# 3) ENCODING (after filter)
df_fs_encoded = dp.encode_categorical(df_fs, target_variable=target)

# convert bool → int
bool_cols = df_fs_encoded.select_dtypes(include="bool").columns
df_fs_encoded[bool_cols] = df_fs_encoded[bool_cols].astype(int)

print("After encoding:", df_fs_encoded.shape, flush=True)

# 4) WRAPPER (final feature selection)
df_final = dp.wrapper_feature_selection(
    df_fs_encoded,
    target_col=target,
    Encoding_flag=False,
    n_features=5,
    direction="forward",
    model_type="classification",
)

print_list("Wrapper Selected Features", list(df_final.columns))
print("Final shape:", df_final.shape, flush=True)


# # =========================================================
# # PCA (FEATURE EXTRACTION)
# # =========================================================
# print_section("PCA")

# numeric_vars, _, _ = du.classify_variables(df_fs)
# numeric_vars = [col for col in numeric_vars if col in df_fs.columns]

# df_pca_input = df_fs[numeric_vars + [target]].copy()

# df_pca = dp.extract_features(
#     df_pca_input,
#     target_col=target,
#     scale_flag=False,
#     CV_cut=80,
# )

# cat_cols = [col for col in df_fs.columns if df_fs[col].dtype == 'object' and col != target]
# df_pca = pd.concat([df_pca, df_fs[cat_cols]], axis=1)

# print("\nPCA shape:", df_pca.shape)
# print("PCA columns:", list(df_pca.columns))

# print(df_final.columns)
# =========================================================
# TRAIN / BALANCE / MODELING
# =========================================================
print_section("MODELING IS STARTING .....")

print_section("MODEL INPUT")

df_model = df_final.copy()

chosen_name = "df_final (Wrapper Selected Features output)"
print(f"Modeling dataframe source: {chosen_name}")
print(f"Shape: {df_model.shape}")
print(f"Columns used: {list(df_model.columns)}")

print_section("TRAIN & BALANCE")

if target not in df_model.columns:
    raise ValueError("Target column missing after PCA")

X = df_model.drop(columns=[target]).copy()
y = df_model[target].copy()

train_df = pd.concat([X, y], axis=1)
print_distribution(train_df[target], "Training Set Before Balancing")

before_table = modeling.compare_knn_and_nb(
    X, y, X, y, 
    target_col=target
)

after_table = modeling.compare_tree_and_ensemble(
    X, y, X, y, 
    target_col=target
)

modeling.export_and_plot_roc(before_table, after_table, X, y)

# =========================================================
# CLOSE EDA PLOTS TO EXCLUDE THEM FROM WORD
# =========================================================
import matplotlib.pyplot as plt
plt.close('all')

# =========================================================
# START CAPTURING ALL MODELING PLOTS (Confusion Matrix, Individual ROC, etc.)
# =========================================================
modeling.init_word_plot_capture()

# =========================================================
# MODEL COMPARISON
# =========================================================

results_knn_nb_before = modeling.compare_knn_and_nb(X, y, X, y)
results_tree_before = modeling.compare_tree_and_ensemble(X, y, X, y)

def add_group(df, group_name):
    out = df.copy()
    if out.index.name is not None:
        out = out.reset_index()
    if out.columns[0] != "Model" and "Model" not in out.columns:
        out.rename(columns={out.columns[0]: "Model"}, inplace=True)
    if "Group" not in out.columns:
        out.insert(0, "Group", group_name)
    out.index = range(len(out))
    return out

before_table = pd.concat([
    add_group(results_knn_nb_before, "KNN + Naive Bayes"),
    add_group(results_tree_before, "Decision Tree + Ensembles"),
], axis=0, ignore_index=True)

after_table = before_table.copy()

before_table = before_table[
    ["Group", "Model", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1-score"]
].sort_values(by="F1-score", ascending=False)

after_table = after_table[
    ["Group", "Model", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1-score"]
].sort_values(by="F1-score", ascending=False)

print("\n==================== FINAL TABLE: BEFORE BALANCING ====================")
print(before_table.round(4).to_string(index=False))

print("\n==================== FINAL TABLE: AFTER BALANCING (NO CHANGES NEEDED) ====================")
print(after_table.round(4).to_string(index=False))

print("\nDone.")

# =========================================================
# Export Excel and Generate ROC Curve 
# =========================================================
modeling.export_and_plot_roc(before_table, after_table, X, y)

# =========================================================
# EXPORT RESULTS AND SAVE CAPTURED PLOTS
# =========================================================
plt.show()
modeling.save_captured_plots_to_word()
