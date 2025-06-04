# eval_geeval.py

import pandas as pd
import matplotlib.pyplot as plt

def _load_and_clean(path: str) -> pd.DataFrame:
    """CSV einlesen, Judgments lowercase & trim."""
    df = pd.read_csv(path)
    df['g_eval_judgment'] = (
        df['g_eval_judgment']
          .astype(str)
          .str.strip()
          .str.lower()
    )
    return df

def summarize_single(df: pd.DataFrame) -> dict:
    """Berechnet Basis-Statistiken und Judgment-Häufigkeiten."""
    scores = df['g_eval_score'].dropna()
    judg   = df['g_eval_judgment']
    summary = {
        'n_samples'    : len(df),
        'mean_score'   : scores.mean(),
        'median_score' : scores.median(),
        'std_score'    : scores.std(),
        'q25_score'    : scores.quantile(0.25),
        'q75_score'    : scores.quantile(0.75)
    }
    for label,cnt in judg.value_counts(dropna=False).items():
        summary[f"count_{label}"] = cnt
        summary[f"pct_{label}"]   = cnt / len(df)
    return summary

def plot_score_distribution(df: pd.DataFrame, title: str):
    """Histogramm der G-Eval Scores mit Raster."""
    plt.figure(figsize=(6,4))
    plt.hist(df['g_eval_score'].dropna(), bins=20, edgecolor='black')
    plt.title(f"Score-Verteilung", pad=15)
    plt.xlabel("G-Eval Score")
    plt.ylabel("Anzahl Beispiele")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_judgment_counts(df: pd.DataFrame, title: str):
    """Balkendiagramm der Judgments."""
    counts = df['g_eval_judgment'].value_counts()
    plt.figure(figsize=(6,4))
    bars = plt.bar(counts.index, counts.values, edgecolor='black')
    plt.title(f"Judgment-Verteilung", pad=15)
    plt.xlabel("Judgment")
    plt.ylabel("Anzahl Beispiele")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def top_bottom_examples(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Gibt die k schlechtesten (unten) und k besten (oben) Beispiele zurück."""
    df_sorted = df.sort_values('g_eval_score')
    bottom = df_sorted.head(k)
    top    = df_sorted.tail(k)
    result = pd.concat([
        bottom.assign(rank=[f"bottom_{i+1}" for i in range(len(bottom))]),
        top   .assign(rank=[f"top_{i+1}"    for i in range(len(top))])
    ])
    return result[['rank', 'g_eval_score', 'g_eval_judgment'] +
                  [c for c in df.columns if c not in ('g_eval_score','g_eval_judgment')]]

def evaluate_geeval(path: str,
                    name: str = None,
                    do_plot: bool = True,
                    show_examples: bool = False,
                    show_scores: bool = False) -> dict:

    name = name or path.split("/")[-1]
    df   = _load_and_clean(path)
    summ = summarize_single(df)

    # ---- Druck der Zusammenfassung ----
    print(f"G-Eval Auswertung")
    print(f"Anzahl Samples      : {summ['n_samples']}")
    print("=" * 30)
    print(f"Mittelwert G-Eval Score: {summ['mean_score']:.3f}")
    print("\n-- Judgment Verteilung --")
    for key in sorted(k for k in summ if k.startswith('pct_')):
        label = key.replace('pct_','')
        pct   = summ[key]*100
        cnt   = summ[f"count_{label}"]
        print(f"  {label:<12}: {pct:5.1f}% ({cnt}x)")
    print("="*40)

    # ---- Beispiele anzeigen ----
    if show_examples:
        ex = top_bottom_examples(df, k=5)
        print("\n-- Top/Bottom Beispiele --")
        print(ex.to_string(index=False))
        print("="*40)

    # ---- Plots ----
    if do_plot:
        plot_score_distribution(df, name)
        plot_judgment_counts(df, name)

def compare_models(paths: dict, do_plot: bool = True) -> pd.DataFrame:
    """
    Vergleicht mehrere G-Eval CSVs.
      paths   – Dict mit {Modellname: Pfad}
      do_plot – ob Vergleichsplots erzeugt werden

    Gibt DataFrame mit Kennzahlen pro Modell zurück.
    """
    all_summaries = {}
    for model_name, path in paths.items():
        df   = _load_and_clean(path)
        summ = summarize_single(df)
        all_summaries[model_name] = summ

    comp_df = pd.DataFrame(all_summaries).T
    comp_df = comp_df.sort_values('mean_score', ascending=False)

    # Druck der Score-Tabelle
    print("\n===== Modellvergleich G-Eval =====")
    display_cols = ['mean_score','median_score','std_score','q25_score','q75_score']
    print(comp_df[display_cols].round(3).to_string())

    # Plots
    if do_plot:
        # Mean Score Vergleich
        plt.figure(figsize=(6,4))
        comp_df['mean_score'].plot.bar(edgecolor='black')
        plt.title("Modellvergleich: Mittelwert der G-Eval Scores", pad=15)
        plt.ylabel("Mean G-Eval Score")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        # Correct vs Incorrect Prozent
        if 'pct_correct' in comp_df.columns and 'pct_incorrect' in comp_df.columns:
            plt.figure(figsize=(6,4))
            comp_df[['pct_correct','pct_incorrect']].plot.bar(edgecolor='black')
            plt.title("Modellvergleich: Correct vs Incorrect", pad=15)
            plt.ylabel("Anteil")
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()

    return comp_df
