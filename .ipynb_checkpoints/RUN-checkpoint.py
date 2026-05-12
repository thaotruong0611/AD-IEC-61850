import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import re

FILE1 = "data split-lack 1 attcks.ipynb"
FILE2 = "new_tech-test lab data.ipynb"   # 🔥 đổi tên file 3 ở đây
FILE3 = "SSRDT_VERSION 2.ipynb"


# =========================
# 5 CASE
# =========================
cases = {
    1: 'small_df1 = small_df',
    2: 'small_df1 = small_df[small_df["Label"] != "Fault"]',

    3: '''small_df1 = small_df[small_df["Label"] != "Fault"]
small_df1 = small_df1[small_df1["Label"] != "Data Integrity"]''',

   4: '''small_df1 = small_df[small_df["Label"] != "Fault"]
small_df1 = small_df1[small_df1["Label"] != "Data Integrity"]
small_df1 = small_df1[small_df1["Label"] != "FDIA"]''',

    5: '''small_df1 = small_df[small_df["Label"] != "Fault"]
small_df1 = small_df1[small_df1["Label"] != "Data Integrity"]
small_df1 = small_df1[small_df1["Label"] != "FDIA"]
small_df1 = small_df1[small_df1["Label"] != "DoS"]''',


}

# =========================
# RUN NOTEBOOK
# =========================
def run_notebook(path):
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=9999, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': './'}})

    return nb

# =========================
# EXTRACT METRICS
# =========================
def extract_metrics(nb):
    text = ""

    for cell in nb.cells:
        if "outputs" in cell:
            for out in cell["outputs"]:
                if "text" in out:
                    text += "".join(out["text"])

    auroc = re.search(r"AUROC:\s*([\d\.]+)", text)
    f1 = re.search(r"Binary F1.*?:\s*([\d\.]+)", text)
    mcc = re.search(r"MCC:\s*([\d\.]+)", text)

    return (
        float(auroc.group(1)) if auroc else None,
        float(f1.group(1)) if f1 else None,
        float(mcc.group(1)) if mcc else None
    )

# =========================
# LOOP 5 CASE
# =========================
results = []

for i in range(1, 6):
    print(f"\n===== CASE {i} =====")

    with open(FILE1) as f:
        nb1 = nbformat.read(f, as_version=4)

    # ===== sửa file1 =====
    for cell in nb1.cells:
        if cell.cell_type == "code" and "small_df1 = small_df" in cell.source:
            
            lines = cell.source.split("\n")
            new_lines = []

            for line in lines:
                if "small_df1 = small_df" in line:
                    break
                new_lines.append(line)

            new_lines.append(cases[i])
            cell.source = "\n".join(new_lines)

    # save file1 temp
    temp_file1 = f"file1_case{i}.ipynb"
    with open(temp_file1, "w") as f:
        nbformat.write(nb1, f)

    # ===== chạy file1 =====
    print("Running file1...")
    run_notebook(temp_file1)

    # ===== chạy file2 =====
    print("Running file2...")
    nb2 = run_notebook(FILE2)
    auroc2, f12, mcc2 = extract_metrics(nb2)

    print("FILE2 → AUROC:", auroc2)
    print("FILE2 → F1:", f12)
    print("FILE2 → MCC:", mcc2)

    # ===== chạy file3 =====
    print("Running file3...")
    nb3 = run_notebook(FILE3)
    auroc3, f13, mcc3 = extract_metrics(nb3)

    print("FILE3 → AUROC:", auroc3)
    print("FILE3 → F1:", f13)
    print("FILE3 → MCC:", mcc3)

    # ===== lưu kết quả =====
    results.append({
        "case": i,

        # file2
        "AUROC_file2": auroc2,
        "F1_file2": f12,
        "MCC_file2": mcc2,

        # file3
        "AUROC_file3": auroc3,
        "F1_file3": f13,
        "MCC_file3": mcc3,
    })

# =========================
# FINAL
# =========================
print("\n===== FINAL =====")
for r in results:
    print(r)