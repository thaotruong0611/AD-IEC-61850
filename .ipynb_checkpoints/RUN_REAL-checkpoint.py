# ============================================================
# RUN_REAL.py
#
# Run notebooks with:
#   N_REAL_REF = 1, 2, 5, 10
#
# Save:
#   AUROC
#   Binary F1
#   MCC
#
# into ONE csv:
#   summary_results_N_REAL_REF.csv
#
# IMPORTANT:
# - NO executed_notebooks folder
# - notebook executed directly in BASE_DIR
# - CSV files must stay next to RUN_REAL.py
# ============================================================

import copy
import re
import subprocess
from pathlib import Path

import nbformat
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

NOTEBOOKS = [
    "Test_new data_LIST_REAL_REF.ipynb",
    "SSRDT_VERSION 2-real_data-LIST_REAL_REF.ipynb",
    "Deep_SAD-real_data-LIST_REAL_REF.ipynb",
]

N_REAL_REF_LIST = [1, 2, 5, 10]

OUTPUT_CSV = BASE_DIR / "summary_results_N_REAL_REF.csv"

LOG_DIR = BASE_DIR / "run_logs"
LOG_DIR.mkdir(exist_ok=True)


# ============================================================
# METRIC REGEX
# ============================================================

METRIC_PATTERNS = {
    "AUROC": [
        r"AUROC_test\s*[:=]\s*([-+]?\d*\.?\d+)",
        r"AUROC\s*[:=]\s*([-+]?\d*\.?\d+)",
    ],

    "Binary_F1": [
        r"F1_binary\(anomaly class\)_test\s*[:=]\s*([-+]?\d*\.?\d+)",
        r"Binary F1 anomaly=1\s*[:=]\s*([-+]?\d*\.?\d+)",
        r"Binary F1 \(anom=1\)\s*[:=]\s*([-+]?\d*\.?\d+)",
        r"Binary F1\s*[:=]\s*([-+]?\d*\.?\d+)",
    ],

    "MCC": [
        r"MCC_test\s*[:=]\s*([-+]?\d*\.?\d+)",
        r"MCC\s*[:=]\s*([-+]?\d*\.?\d+)",
    ],
}


# ============================================================
# REPLACE N_REAL_REF
# ============================================================

def replace_n_real_ref(nb, value):

    pattern = re.compile(
        r"^(\s*)N_REAL_REF\s*=\s*\d+.*$",
        re.MULTILINE,
    )

    replaced_count = 0

    for cell in nb.cells:

        if cell.cell_type != "code":
            continue

        src = cell.source

        if "N_REAL_REF" not in src:
            continue

        new_src, n = pattern.subn(
            rf"\1N_REAL_REF = {value}",
            src,
        )

        if n > 0:
            cell.source = new_src
            replaced_count += n

    return nb, replaced_count


# ============================================================
# EXECUTE NOTEBOOK
# ============================================================

def execute_notebook(notebook_path):

    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        notebook_path.name,
        "--inplace",
        "--ExecutePreprocessor.timeout=-1",
        "--ExecutePreprocessor.kernel_name=python3",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=BASE_DIR,
    )

    return result


# ============================================================
# COLLECT OUTPUT
# ============================================================

def collect_output_text(nb):

    all_text = []

    for cell in nb.cells:

        if cell.cell_type != "code":
            continue

        for out in cell.get("outputs", []):

            output_type = out.get("output_type", "")

            # stream
            if output_type == "stream":

                all_text.append(
                    out.get("text", "")
                )

            # display / execute_result
            elif output_type in [
                "execute_result",
                "display_data",
            ]:

                data = out.get("data", {})

                if "text/plain" in data:

                    txt = data["text/plain"]

                    if isinstance(txt, list):
                        all_text.append("\n".join(txt))
                    else:
                        all_text.append(str(txt))

            # traceback
            elif output_type == "error":

                tb = "\n".join(
                    out.get("traceback", [])
                )

                all_text.append(tb)

    return "\n".join(all_text)


# ============================================================
# EXTRACT METRICS
# ============================================================

def extract_one_metric(text, patterns):

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if matches:
            return float(matches[-1])

    return None


def extract_metrics(text):

    metrics = {}

    for name, patterns in METRIC_PATTERNS.items():

        metrics[name] = extract_one_metric(
            text,
            patterns,
        )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print("BASE_DIR:", BASE_DIR)

    results = []

    for notebook_name in NOTEBOOKS:

        notebook_path = BASE_DIR / notebook_name

        print("\n" + "=" * 80)
        print("RUNNING:", notebook_name)
        print("=" * 80)

        # ----------------------------------------------------
        # notebook exists?
        # ----------------------------------------------------
        if not notebook_path.exists():

            print("[ERROR] Notebook not found:", notebook_name)

            for n_ref in N_REAL_REF_LIST:

                results.append({
                    "file": notebook_name,
                    "N_REAL_REF": n_ref,
                    "AUROC": None,
                    "Binary_F1": None,
                    "MCC": None,
                    "status": "NOTEBOOK_NOT_FOUND",
                })

            pd.DataFrame(results).to_csv(
                OUTPUT_CSV,
                index=False,
            )

            continue

        # ----------------------------------------------------
        # backup original notebook
        # ----------------------------------------------------
        with open(notebook_path, "r", encoding="utf-8") as f:

            original_nb = nbformat.read(
                f,
                as_version=4,
            )

        # ====================================================
        # LOOP N_REAL_REF
        # ====================================================
        for n_ref in N_REAL_REF_LIST:

            print("\n" + "-" * 80)
            print(f"{notebook_name} | N_REAL_REF = {n_ref}")
            print("-" * 80)

            log_path = LOG_DIR / f"log_{notebook_path.stem}_ref{n_ref}.txt"

            row = {
                "file": notebook_name,
                "N_REAL_REF": n_ref,
                "AUROC": None,
                "Binary_F1": None,
                "MCC": None,
                "status": None,
            }

            try:

                # ------------------------------------------------
                # restore notebook
                # ------------------------------------------------
                nb = copy.deepcopy(original_nb)

                # ------------------------------------------------
                # replace N_REAL_REF
                # ------------------------------------------------
                nb, replaced_count = replace_n_real_ref(
                    nb,
                    n_ref,
                )

                if replaced_count == 0:

                    print("[ERROR] N_REAL_REF not found")

                    row["status"] = "N_REAL_REF_NOT_FOUND"

                    results.append(row)

                    pd.DataFrame(results).to_csv(
                        OUTPUT_CSV,
                        index=False,
                    )

                    continue

                # ------------------------------------------------
                # force working directory
                # ------------------------------------------------
                cwd_cell = nbformat.v4.new_code_cell(
                    f"""
import os
os.chdir(r"{BASE_DIR}")
print("WORKDIR =", os.getcwd())
"""
                )

                nb.cells.insert(0, cwd_cell)

                # ------------------------------------------------
                # overwrite notebook temporarily
                # ------------------------------------------------
                with open(notebook_path, "w", encoding="utf-8") as f:

                    nbformat.write(nb, f)

                # ------------------------------------------------
                # execute notebook
                # ------------------------------------------------
                result = execute_notebook(
                    notebook_path
                )

                # ------------------------------------------------
                # save log
                # ------------------------------------------------
                combined_log = (
                    "STDOUT:\n"
                    + result.stdout
                    + "\n\nSTDERR:\n"
                    + result.stderr
                )

                log_path.write_text(
                    combined_log,
                    encoding="utf-8",
                )

                # ------------------------------------------------
                # execution failed?
                # ------------------------------------------------
                if result.returncode != 0:

                    print("[ERROR] Notebook execution failed.")
                    print("See log:", log_path)

                    row["status"] = "EXECUTION_FAILED"

                    results.append(row)

                    pd.DataFrame(results).to_csv(
                        OUTPUT_CSV,
                        index=False,
                    )

                    continue

                # ------------------------------------------------
                # read executed notebook
                # ------------------------------------------------
                with open(notebook_path, "r", encoding="utf-8") as f:

                    executed_nb = nbformat.read(
                        f,
                        as_version=4,
                    )

                # ------------------------------------------------
                # collect outputs
                # ------------------------------------------------
                output_text = collect_output_text(
                    executed_nb
                )

                with open(log_path, "a", encoding="utf-8") as f:

                    f.write("\n\nNOTEBOOK_OUTPUT_TEXT:\n")
                    f.write(output_text)

                # ------------------------------------------------
                # extract metrics
                # ------------------------------------------------
                metrics = extract_metrics(
                    output_text
                )

                row["AUROC"] = metrics["AUROC"]
                row["Binary_F1"] = metrics["Binary_F1"]
                row["MCC"] = metrics["MCC"]

                # ------------------------------------------------
                # status
                # ------------------------------------------------
                if (
                    row["AUROC"] is None
                    or row["Binary_F1"] is None
                    or row["MCC"] is None
                ):

                    row["status"] = "SUCCESS_BUT_PARSE_FAILED"

                else:

                    row["status"] = "SUCCESS"

                # ------------------------------------------------
                # print
                # ------------------------------------------------
                print("AUROC     :", row["AUROC"])
                print("Binary F1 :", row["Binary_F1"])
                print("MCC       :", row["MCC"])
                print("Status    :", row["status"])

                results.append(row)

                # ------------------------------------------------
                # save ONE csv
                # ------------------------------------------------
                pd.DataFrame(results).to_csv(
                    OUTPUT_CSV,
                    index=False,
                )

            except Exception as e:

                print("[ERROR]", e)

                row["status"] = f"ERROR: {e}"

                results.append(row)

                pd.DataFrame(results).to_csv(
                    OUTPUT_CSV,
                    index=False,
                )

        # ----------------------------------------------------
        # restore original notebook
        # ----------------------------------------------------
        with open(notebook_path, "w", encoding="utf-8") as f:

            nbformat.write(
                original_nb,
                f,
            )

    # ========================================================
    # FINAL SAVE
    # ========================================================

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("\n" + "=" * 80)
    print("FINISHED")
    print("=" * 80)

    print("Saved:", OUTPUT_CSV)

    print(results_df)


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()