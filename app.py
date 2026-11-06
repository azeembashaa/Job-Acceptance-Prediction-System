from pathlib import Path
import importlib
import pandas as pd  # pyright: ignore[reportMissingModuleSource]
import plotly.express as px  # pyright: ignore[reportMissingImports]
import streamlit as st  # pyright: ignore[reportMissingImports]


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Job Acceptance Prediction System",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "HR_Job_Placement_Cleaned.csv"
MODEL_PATH = BASE_DIR / "models" / "final_pipeline.pkl"


# ============================================================
# 2. COLUMN NAME SUPPORT
# ============================================================

COLUMN_ALIASES = {
    "status": [
        "status",
        "placement_status",
        "placed",
        "placement_outcome",
        "job_acceptance",
        "accepted",
        "outcome",
    ],
    "ssc_percentage": [
        "ssc_percentage",
        "ssc_percent",
        "ssc_score",
        "10th_percentage",
        "tenth_percentage",
    ],
    "hsc_percentage": [
        "hsc_percentage",
        "hsc_percent",
        "hsc_score",
        "12th_percentage",
        "twelfth_percentage",
    ],
    "degree_percentage": [
        "degree_percentage",
        "degree_percent",
        "degree_score",
        "graduation_percentage",
    ],
    "skills_match_score": [
        "skills_match_score",
        "skills_match_percentage",
    ],
    "interview_score": [
         "interview_score",
         "technical_score",
    ],
    "certification": [
         "certification",
         "certifications_count",
    ],
    "company_tier": [
        "company_tier",
        "company_level",
        "company_category",
        "tier",
    ],
    "experience_years": [
        "experience_years",
        "years_of_experience",
        "work_experience",
        "experience",
    ],
    "employability_test_score": [
        "employability_test_score",
        "employability_score",
        "aptitude_score",
        "test_score",
    ],
    "placement_probability": [
        "placement_probability",
        "acceptance_probability",
        "predicted_probability",
        "probability",
    ],
}


def normalize_name(value):
    """Normalize a column name for matching."""
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def standardize_columns(dataframe):
    """Rename recognized column variations to standard names."""
    dataframe = dataframe.copy()
    normalized_to_original = {
        normalize_name(column): column
        for column in dataframe.columns
    }

    rename_map = {}

    for standard_name, aliases in COLUMN_ALIASES.items():
        if standard_name in dataframe.columns:
            continue

        for alias in aliases:
            matched_column = normalized_to_original.get(normalize_name(alias))
            if matched_column is not None:
                rename_map[matched_column] = standard_name
                break

    return dataframe.rename(columns=rename_map)


# ============================================================
# 3. LOAD PROJECT FILES
# ============================================================

@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    joblib = importlib.import_module("joblib")
    return joblib.load(path)


if not DATA_PATH.exists():
    st.error(
        f"Dataset not found: `{DATA_PATH}`. "
        "Place the CSV file in the project's `data` folder."
    )
    st.stop()

try:
    df = standardize_columns(load_data(str(DATA_PATH)))
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

model = None
model_load_message = None

if MODEL_PATH.exists():
    try:
        model = load_model(str(MODEL_PATH))
    except Exception as error:
        model_load_message = f"Saved model could not be loaded: {error}"
else:
    model_load_message = (
        "No saved model found. Analyses using observed outcomes will still work."
    )


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def find_column(dataframe, standard_name):
    """Return a matching dataframe column, if available."""
    if standard_name in dataframe.columns:
        return standard_name

    accepted_names = {
        normalize_name(standard_name),
        *[normalize_name(name) for name in COLUMN_ALIASES.get(standard_name, [])],
    }

    for column in dataframe.columns:
        if normalize_name(column) in accepted_names:
            return column

    return None


def require_columns(dataframe, standard_names):
    """Show a helpful message if required columns are missing."""
    missing = [
        name for name in standard_names
        if find_column(dataframe, name) is None
    ]

    if missing:
        st.warning(
            "This analysis needs these dataset columns: "
            + ", ".join(f"`{name}`" for name in missing)
            + ". Update the CSV headers or add the column aliases near the top "
              "of `app.py`."
        )
        return False

    return True


def numeric_column(dataframe, column):
    """Convert a column to numeric values where possible."""
    return pd.to_numeric(dataframe[column], errors="coerce")


def accepted_mask(series):
    """Interpret common placement and acceptance labels as True/False."""
    positive_values = {
        "placed",
        "yes",
        "accepted",
        "selected",
        "1",
        "true",
        "offer accepted",
        "job accepted",
    }

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(positive_values)
    )


def calculate_model_probability(dataframe):
    """
    Return the model's positive-class probability when its inputs can be
    identified. Return None when the saved pipeline cannot be used safely.
    """
    if model is None or not hasattr(model, "predict_proba"):
        return None

    target_column = find_column(dataframe, "status")

    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
        if not all(name in dataframe.columns for name in feature_names):
            return None
        features = dataframe[feature_names].copy()
    else:
        excluded = {
            target_column,
            find_column(dataframe, "placement_probability"),
        }
        features = dataframe[
            [column for column in dataframe.columns if column not in excluded]
        ].copy()

    try:
        probabilities = model.predict_proba(features)
    except Exception:
        return None

    classes = list(getattr(model, "classes_", []))
    positive_index = None

    for index, label in enumerate(classes):
        if str(label).strip().lower() in {
            "placed",
            "yes",
            "accepted",
            "selected",
            "1",
            "true",
        }:
            positive_index = index
            break

    if positive_index is None:
        positive_index = 1 if probabilities.shape[1] > 1 else 0

    return probabilities[:, positive_index]


def add_status_labels(dataframe):
    """Add a readable placement label for charts."""
    status_column = find_column(dataframe, "status")
    if status_column is None:
        return dataframe

    result = dataframe.copy()
    result["Placement outcome"] = result[status_column].astype(str)
    return result


def show_numeric_scatter(dataframe, x_column, y_column, title, color=None):
    chart_data = dataframe[[x_column, y_column]].copy()
    chart_data[x_column] = numeric_column(chart_data, x_column)
    chart_data[y_column] = numeric_column(chart_data, y_column)
    chart_data = chart_data.dropna(subset=[x_column, y_column])

    if color and color in dataframe.columns:
        chart_data[color] = dataframe.loc[chart_data.index, color].astype(str)

    if chart_data.empty:
        st.info("There are no numeric rows available for this chart.")
        return

    fig = px.scatter(
        chart_data,
        x=x_column,
        y=y_column,
        color=color if color in chart_data.columns else None,
        title=title,
        opacity=0.7,
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 5. DASHBOARD HEADER AND SUMMARY
# ============================================================

st.title("📊 Job Acceptance Prediction System")
st.caption("Candidate performance, placement outcomes, and evaluation analytics")

if model_load_message:
    st.info(model_load_message)

status_column = find_column(df, "status")

if status_column:
    placement_rate = accepted_mask(df[status_column]).mean() * 100
else:
    placement_rate = None

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Candidates", f"{len(df):,}")
kpi2.metric("Dataset columns", f"{len(df.columns):,}")

if placement_rate is not None:
    kpi3.metric("Placement / acceptance rate", f"{placement_rate:.1f}%")
else:
    kpi3.metric("Placement / acceptance rate", "Status column unavailable")

with st.expander("Dataset overview and preview"):
    st.write(f"**Rows:** {len(df):,}  |  **Columns:** {len(df.columns):,}")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Missing values")
    missing_values = (
        df.isna()
        .sum()
        .rename("Missing values")
        .to_frame()
    )
    st.dataframe(missing_values, use_container_width=True)


# ============================================================
# 6. GRADIENT BOOSTING MODEL DEMO
# ============================================================

st.divider()
st.header("Gradient Boosting Model")

if model is None:
    st.warning(
        "The saved pipeline was not loaded. Confirm that "
        "`models/final_pipeline.pkl` exists and can be opened."
    )
else:
    # Re-read the original CSV so the model receives its training column names.
    model_data = load_data(str(DATA_PATH)).copy()

    # The notebook trained using this feature name. Support the app CSV's alias.
    if (
        "years_of_experience" not in model_data.columns
        and "experience_years" in model_data.columns
    ):
        model_data = model_data.rename(
            columns={"experience_years": "years_of_experience"}
        )

    required_source_columns = [
        "status",
        "years_of_experience",
        "ssc_percentage",
        "hsc_percentage",
        "degree_percentage",
        "skills_match_percentage",
        "technical_score",
        "aptitude_score",
        "communication_score",
    ]

    missing_source_columns = [
        column
        for column in required_source_columns
        if column not in model_data.columns
    ]

    if missing_source_columns:
        st.error(
            "The model needs these original notebook columns: "
            + ", ".join(f"`{column}`" for column in missing_source_columns)
        )
    else:
        try:
            # Match the notebook's data cleanup.
            model_data = model_data.drop_duplicates(keep="first").copy()

            categorical_columns = model_data.select_dtypes(
                include=["object", "category", "string"]
            ).columns

            for column in categorical_columns:
                model_data[column] = model_data[column].str.strip()

            if "gender" in model_data.columns:
                model_data["gender"] = model_data["gender"].str.title()

            if "internship_experience" in model_data.columns:
                model_data["internship_experience"] = (
                    model_data["internship_experience"].str.title()
                )

            # Recreate the features engineered in the notebook.
            model_data["experience_category"] = pd.cut(
                model_data["years_of_experience"],
                bins=[-1, 0, 2, 5],
                labels=["Fresher", "Junior", "Senior"],
            )

            model_data["academic_average"] = model_data[
                [
                    "ssc_percentage",
                    "hsc_percentage",
                    "degree_percentage",
                ]
            ].mean(axis=1)

            model_data["academic_performance_band"] = pd.cut(
                model_data["academic_average"],
                bins=[0, 60, 75, 100],
                labels=["Low", "Medium", "High"],
                include_lowest=True,
            )

            model_data["skills_match_level"] = pd.cut(
                model_data["skills_match_percentage"],
                bins=[0, 60, 80, 100],
                labels=["Low", "Medium", "High"],
                include_lowest=True,
            )

            model_data["interview_average"] = model_data[
                [
                    "technical_score",
                    "aptitude_score",
                    "communication_score",
                ]
            ].mean(axis=1)

            model_data["interview_performance_category"] = pd.cut(
                model_data["interview_average"],
                bins=[0, 60, 75, 100],
                labels=["Low", "Medium", "High"],
                include_lowest=True,
            )

            # Match the notebook's model inputs: remove the target and
            # the analytical-only score.
            model_features = model_data.drop(
                columns=["status", "placement_analytical_score"],
                errors="ignore",
            )

            # Use the exact feature names and order recorded by the pipeline.
            expected_features = getattr(model, "feature_names_in_", None)

            if expected_features is None and hasattr(model, "named_steps"):
                fitted_preprocessor = model.named_steps.get("preprocessor")
                expected_features = getattr(
                    fitted_preprocessor,
                    "feature_names_in_",
                    None,
                )

            if expected_features is None:
                st.error(
                    "The saved pipeline does not expose its expected feature "
                    "names, so the app cannot safely match inputs to the model."
                )
            else:
                expected_features = list(expected_features)
                missing_model_features = [
                    column
                    for column in expected_features
                    if column not in model_features.columns
                ]

                if missing_model_features:
                    st.error(
                        "The CSV is missing model features: "
                        + ", ".join(f"`{column}`" for column in missing_model_features)
                    )
                else:
                    model_features = model_features[expected_features]

                    metric_columns = st.columns(5)
                    metric_columns[0].metric("Accuracy", "90.58%")
                    metric_columns[1].metric("Precision", "85.30%")
                    metric_columns[2].metric("Recall", "83.21%")
                    metric_columns[3].metric("F1 score", "84.24%")
                    metric_columns[4].metric("ROC-AUC", "96.79%")

                    st.subheader("Try a candidate prediction")
                    candidate_number = st.number_input(
                        "Candidate row number",
                        min_value=1,
                        max_value=len(model_features),
                        value=1,
                        step=1,
                        key="gb_candidate_row",
                    )

                    candidate_features = model_features.iloc[
                        [int(candidate_number) - 1]
                    ]

                    predicted_class = model.predict(candidate_features)[0]
                    class_labels = list(model.classes_)

                    positive_class_index = next(
                        (
                            index
                            for index, label in enumerate(class_labels)
                            if str(label).strip().lower()
                            in {"1", "placed", "yes", "accepted", "true"}
                        ),
                        1 if len(class_labels) > 1 else 0,
                    )

                    predicted_probability = model.predict_proba(
                        candidate_features
                    )[0][positive_class_index]

                    placed_prediction = (
                        str(predicted_class).strip().lower()
                        in {"1", "placed", "yes", "accepted", "true"}
                    )

                    prediction_col, probability_col = st.columns(2)
                    prediction_col.metric(
                        "Predicted outcome",
                        "Placed" if placed_prediction else "Not Placed",
                    )
                    probability_col.metric(
                        "Model-estimated placement probability",
                        f"{predicted_probability:.1%}",
                    )

                    st.caption(
                        "The row number selects a record from the dataset for "
                        "demonstration. This prediction is not an offer-acceptance "
                        "or offer-dropout prediction."
                    )

                    # Show global feature importance from the Gradient Boosting
                    # classifier inside the saved pipeline.
                    st.subheader("What influences the model overall?")

                    if hasattr(model, "named_steps"):
                        fitted_preprocessor = model.named_steps.get("preprocessor")
                        classifier = model.named_steps.get("classifier")
                    else:
                        fitted_preprocessor = None
                        classifier = None

                    if (
                        fitted_preprocessor is not None
                        and classifier is not None
                        and hasattr(classifier, "feature_importances_")
                    ):
                        feature_names = (
                            fitted_preprocessor.get_feature_names_out()
                        )
                        importances = classifier.feature_importances_

                        if len(feature_names) == len(importances):
                            importance_data = pd.DataFrame(
                                {
                                    "Feature": feature_names,
                                    "Importance": importances,
                                }
                            ).nlargest(15, "Importance").sort_values(
                                "Importance",
                                ascending=True,
                            )

                            importance_chart = px.bar(
                                importance_data,
                                x="Importance",
                                y="Feature",
                                orientation="h",
                                title="Top 15 global feature importances",
                            )
                            st.plotly_chart(
                                importance_chart,
                                use_container_width=True,
                            )
                        else:
                            st.info(
                                "The pipeline's feature names do not align with "
                                "the classifier's importance values."
                            )
                    else:
                        st.info(
                            "This saved model does not expose feature importances."
                        )

        except Exception as error:
            st.error(f"Could not run the Gradient Boosting demo: {error}")


# ============================================================
# 7. ANALYST TASK SELECTOR
# ============================================================

st.divider()
st.header("📈 Analyst Tasks (EDA & ML Analytics)")

analysis_options = [
    "Select an analysis",
    "1. Academic scores vs placement outcome",
    "2. Skills match vs interview performance",
    "3. Certification impact on job acceptance",
    "4. Acceptance rate by company tier",
    "5. Experience vs placement success",
    "6. Interview score vs placement probability",
    "7. Employability test score analysis",
]

selected_analysis = st.selectbox(
    "Choose an analytical question:",
    analysis_options,
)


# ============================================================
# 8. ANALYSIS 1: ACADEMIC SCORES VS PLACEMENT OUTCOME
# ============================================================

if selected_analysis == analysis_options[1]:
    st.subheader("Academic scores vs placement outcome")

    score_columns = [
        column for column in [
            find_column(df, "ssc_percentage"),
            find_column(df, "hsc_percentage"),
            find_column(df, "degree_percentage"),
        ]
        if column is not None
    ]

    if not score_columns or not status_column:
        st.warning(
            "This analysis needs at least one academic score column "
            "(`ssc_percentage`, `hsc_percentage`, or `degree_percentage`) "
            "and a `status` column."
        )
    else:
        plot_data = df[[status_column] + score_columns].copy()

        for column in score_columns:
            plot_data[column] = numeric_column(plot_data, column)

        average_scores = (
            plot_data.groupby(status_column, dropna=False)[score_columns]
            .mean()
            .reset_index()
            .melt(
                id_vars=status_column,
                var_name="Academic score",
                value_name="Average score",
            )
        )

        fig = px.bar(
            average_scores,
            x="Academic score",
            y="Average score",
            color=status_column,
            barmode="group",
            title="Average academic scores by placement outcome",
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 9. ANALYSIS 2: SKILLS MATCH VS INTERVIEW PERFORMANCE
# ============================================================

elif selected_analysis == analysis_options[2]:
    st.subheader("Skills match vs interview performance")

    skill_column = find_column(df, "skills_match_score")
    interview_column = find_column(df, "interview_score")

    if not require_columns(
        df,
        ["skills_match_score", "interview_score"],
    ):
        st.stop()

    plot_data = df.copy()
    plot_data[skill_column] = numeric_column(plot_data, skill_column)
    plot_data[interview_column] = numeric_column(plot_data, interview_column)
    plot_data = plot_data.dropna(subset=[skill_column, interview_column])

    if plot_data.empty:
        st.info("No numeric skills and interview scores were found.")
    else:
        fig = px.scatter(
            add_status_labels(plot_data),
            x=skill_column,
            y=interview_column,
            color="Placement outcome" if status_column else None,
            title="Skills match score vs interview score",
            opacity=0.7,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Each point represents one candidate. Placement outcome is shown "
            "when the dataset contains a recognizable status column."
        )


# ============================================================
# 10. ANALYSIS 3: CERTIFICATION IMPACT ON JOB ACCEPTANCE
# ============================================================

elif selected_analysis == analysis_options[3]:
    st.subheader("Certification impact on job acceptance")

    if not require_columns(df, ["certification", "status"]):
        st.stop()

    certification_column = find_column(df, "certification")
    placement_data = df[[certification_column, status_column]].copy()
    placement_data["Accepted"] = accepted_mask(placement_data[status_column])

    certification_summary = (
        placement_data.groupby(certification_column, dropna=False)["Accepted"]
        .mean()
        .mul(100)
        .rename("Acceptance rate (%)")
        .reset_index()
    )

    fig = px.bar(
        certification_summary,
        x=certification_column,
        y="Acceptance rate (%)",
        title="Acceptance rate by certification group",
        text_auto=".1f",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(certification_summary, use_container_width=True)


# ============================================================
# 11. ANALYSIS 4: ACCEPTANCE RATE BY COMPANY TIER
# ============================================================

elif selected_analysis == analysis_options[4]:
    st.subheader("Acceptance rate by company tier")

    if not require_columns(df, ["company_tier", "status"]):
        st.stop()

    tier_column = find_column(df, "company_tier")
    placement_data = df[[tier_column, status_column]].copy()
    placement_data["Accepted"] = accepted_mask(placement_data[status_column])

    tier_summary = (
        placement_data.groupby(tier_column, dropna=False)["Accepted"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "Acceptance rate",
                "count": "Candidate count",
            }
        )
    )
    tier_summary["Acceptance rate (%)"] = tier_summary["Acceptance rate"] * 100

    fig = px.bar(
        tier_summary,
        x=tier_column,
        y="Acceptance rate (%)",
        text="Candidate count",
        title="Acceptance rate by company tier",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(tier_summary, use_container_width=True)


# ============================================================
# 12. ANALYSIS 5: EXPERIENCE VS PLACEMENT SUCCESS
# ============================================================

elif selected_analysis == analysis_options[5]:
    st.subheader("Experience vs placement success")

    if not require_columns(df, ["experience_years", "status"]):
        st.stop()

    experience_column = find_column(df, "experience_years")
    placement_data = df[[experience_column, status_column]].copy()
    placement_data[experience_column] = numeric_column(
        placement_data,
        experience_column,
    )
    placement_data["Accepted"] = accepted_mask(placement_data[status_column])
    placement_data = placement_data.dropna(subset=[experience_column])

    if placement_data.empty:
        st.info("No numeric experience values were found.")
    else:
        experience_summary = (
            placement_data.groupby(experience_column)["Accepted"]
            .mean()
            .mul(100)
            .rename("Acceptance rate (%)")
            .reset_index()
        )

        fig = px.line(
            experience_summary,
            x=experience_column,
            y="Acceptance rate (%)",
            markers=True,
            title="Acceptance rate by years of experience",
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 13. ANALYSIS 6: INTERVIEW SCORE VS PLACEMENT PROBABILITY
# ============================================================

elif selected_analysis == analysis_options[6]:
    st.subheader("Interview score vs placement probability")

    interview_column = find_column(df, "interview_score")

    if interview_column is None:
        st.warning("This analysis needs an `interview_score` column.")
    else:
        probability_column = find_column(df, "placement_probability")
        plot_data = df.copy()
        plot_data[interview_column] = numeric_column(
            plot_data,
            interview_column,
        )

        if probability_column:
            plot_data[probability_column] = numeric_column(
                plot_data,
                probability_column,
            )
            probability_source = probability_column
            probability_label = "Placement probability"
        else:
            model_probabilities = calculate_model_probability(df)

            if model_probabilities is not None:
                plot_data["placement_probability_calculated"] = model_probabilities
                probability_source = "placement_probability_calculated"
                probability_label = "Model placement probability"
            elif status_column:
                plot_data["Accepted"] = accepted_mask(plot_data[status_column])
                plot_data = plot_data.dropna(subset=[interview_column])

                if plot_data.empty:
                    st.info("No numeric interview scores were found.")
                    st.stop()

                plot_data["Interview score group"] = pd.qcut(
                    plot_data[interview_column],
                    q=min(10, plot_data[interview_column].nunique()),
                    duplicates="drop",
                )
                # Convert pandas Interval values to text so Plotly can serialize them
                plot_data["Interview score group"] = (
                     plot_data["Interview score group"].astype(str)
                )

                binned = (
                    plot_data.groupby("Interview score group", observed=True)[
                        "Accepted"
                    ]
                    .mean()
                    .mul(100)
                    .rename("Observed acceptance rate (%)")
                    .reset_index()
                )

                fig = px.line(
                    binned,
                    x="Interview score group",
                    y="Observed acceptance rate (%)",
                    markers=True,
                    title="Observed acceptance rate by interview score range",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "No usable model probability or probability column was "
                    "available, so this chart shows observed acceptance rates."
                )
                st.stop()
            else:
                st.warning(
                    "Provide a `placement_probability` column, a compatible "
                    "saved classification pipeline, or a `status` outcome "
                    "column to show this analysis."
                )
                st.stop()

        plot_data = plot_data.dropna(
            subset=[interview_column, probability_source]
        )

        if plot_data.empty:
            st.info("No numeric interview scores and probabilities were found.")
        else:
            fig = px.scatter(
                plot_data,
                x=interview_column,
                y=probability_source,
                title=f"{probability_label} by interview score",
                opacity=0.7,
            )
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 14. ANALYSIS 7: EMPLOYABILITY TEST SCORE ANALYSIS
# ============================================================

elif selected_analysis == analysis_options[7]:
    st.subheader("Employability test score analysis")

    employability_column = find_column(df, "employability_test_score")

    if employability_column is None:
        st.warning(
            "This analysis needs an `employability_test_score` column."
        )
    else:
        plot_data = df.copy()
        plot_data[employability_column] = numeric_column(
            plot_data,
            employability_column,
        )
        plot_data = plot_data.dropna(subset=[employability_column])

        if plot_data.empty:
            st.info("No numeric employability test scores were found.")
        else:
            if status_column:
                plot_data = add_status_labels(plot_data)
                color_column = "Placement outcome"
            else:
                color_column = None

            fig = px.histogram(
                plot_data,
                x=employability_column,
                color=color_column,
                marginal="box",
                title="Employability test score distribution",
                barmode="overlay",
                opacity=0.7,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.write(
                f"**Average score:** "
                f"{plot_data[employability_column].mean():.2f}  |  "
                f"**Median score:** "
                f"{plot_data[employability_column].median():.2f}"
            )

# ============================================================
# 15. KEY KPI SELECTOR
# ============================================================

st.divider()
st.header("🎯 Key KPIs")

kpi_options = [
    "Select a KPI",
    "Total Candidates",
    "Placement Rate (%)",
    "Job Acceptance Rate (%)",
    "Average Interview Score",
    "Average Skills Match (%)",
    "Non-placement Rate (%)",
    "Historical Non-placement Share (%)",
]

selected_kpi = st.selectbox(
    "Choose a KPI to view:",
    kpi_options,
    key="kpi_selector",
)

if selected_kpi == "Total Candidates":
    st.metric("Total Candidates", f"{len(df):,}")

elif selected_kpi == "Placement Rate (%)":
    if status_column:
        placement_rate_value = accepted_mask(df[status_column]).mean() * 100
        st.metric("Placement Rate", f"{placement_rate_value:.1f}%")
    else:
        st.info("A placement status column is needed for this KPI.")

elif selected_kpi == "Job Acceptance Rate (%)":
    if status_column:
        acceptance_proxy = accepted_mask(df[status_column]).mean() * 100
        st.metric("Job Acceptance Rate", f"{acceptance_proxy:.1f}%")
    else:
        st.info("A status column is needed for this proxy.")

elif selected_kpi == "Average Interview Score":
    interview_column = find_column(df, "interview_score")

    if interview_column:
        scores = numeric_column(df, interview_column).dropna()
        if not scores.empty:
            st.metric("Average Interview Score", f"{scores.mean():.2f}")
        else:
            st.info("The interview score column has no numeric values.")
    else:
        st.info("An interview score column is needed for this KPI.")

elif selected_kpi == "Average Skills Match (%)":
    skills_column = find_column(df, "skills_match_score")

    if skills_column:
        scores = numeric_column(df, skills_column).dropna()
        if not scores.empty:
            st.metric("Average Skills Match", f"{scores.mean():.2f}%")
        else:
            st.info("The skills match column has no numeric values.")
    else:
        st.info("A skills match column is needed for this KPI.")

elif selected_kpi == "Non-placement Rate (%)":
    if status_column:
        placement_rate_value = accepted_mask(df[status_column]).mean() * 100
        non_placement_proxy = 100 - placement_rate_value
        st.metric("Non-placement Rate", f"{non_placement_proxy:.1f}%")
    else:
        st.info("A placement status column is needed for this proxy.")

elif selected_kpi == "Historical Non-placement Share (%)":
    if status_column:
        placement_rate_value = accepted_mask(df[status_column]).mean() * 100
        historical_non_placement = 100 - placement_rate_value
        st.metric(
            "Historical Non-placement Share",
            f"{historical_non_placement:.1f}%",
        )
    else:
        st.info("A placement status column is needed for this proxy.")
