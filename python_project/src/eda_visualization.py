"""
Module 4: Exploratory Data Analysis (EDA)
-----------------------------------------
Generates Matplotlib, Seaborn, and Plotly visualizations for age distribution,
gender comparisons, gallstone associations, wall thickness, and correlations.
"""

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt


def plot_age_distribution(df: pd.DataFrame):
    """Interactive Plotly histogram comparing age distribution by cancer risk label."""
    fig = px.histogram(
        df,
        x="Age",
        color=df["Cancer_Risk"].map({0: "Low Risk (0)", 1: "Elevated Risk (1)"}),
        barmode="overlay",
        nbins=24,
        title="Age Distribution by Risk Classification",
        labels={"color": "Class", "Age": "Patient Age (Years)"},
        color_discrete_map={"Low Risk (0)": "#0D9488", "Elevated Risk (1)": "#DC2626"},
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_gallstones_and_gender(df: pd.DataFrame):
    """Grouped bar chart showing cancer risk prevalence by Gender and Gallstones."""
    grouped = (
        df.groupby(["Gender", "Gallstones"])["Cancer_Risk"]
        .mean()
        .mul(100)
        .reset_index(name="Risk_Prevalence_Pct")
    )
    grouped["Gallstones_Label"] = grouped["Gallstones"].map({0: "No Gallstones", 1: "Gallstones Present"})
    fig = px.bar(
        grouped,
        x="Gender",
        y="Risk_Prevalence_Pct",
        color="Gallstones_Label",
        barmode="group",
        title="Risk Prevalence (%) by Gender and Gallstone Status",
        labels={"Risk_Prevalence_Pct": "Elevated Risk Rate (%)"},
        color_discrete_sequence=["#64748B", "#0D9488"],
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_wall_thickness_box(df: pd.DataFrame):
    """Box plot comparing gallbladder wall thickness across target classes."""
    temp = df.copy()
    temp["Risk_Label"] = temp["Cancer_Risk"].map({0: "Low Risk (0)", 1: "Elevated Risk (1)"})
    fig = px.box(
        temp,
        x="Risk_Label",
        y="Wall_Thickness_mm",
        color="Risk_Label",
        title="Gallbladder Wall Thickness (mm) by Risk Class",
        color_discrete_map={"Low Risk (0)": "#0D9488", "Elevated Risk (1)": "#DC2626"},
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    """Compute Pearson correlation matrix on numeric columns and return a Plotly heatmap."""
    numeric_df = df.select_dtypes(include="number").copy()
    corr = numeric_df.corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Clinical Feature Correlation Matrix (Pearson r)",
    )
    fig.update_layout(template="plotly_white")
    return fig


def save_static_seaborn_report(df: pd.DataFrame, output_path: str = "eda_correlation.png"):
    """Optional Matplotlib/Seaborn static chart export for academic report figures."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Gallbladder Cancer Clinical Attributes Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
