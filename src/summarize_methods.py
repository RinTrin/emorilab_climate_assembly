import pandas as pd
import os

def summarize_top1_by_source(df, presenter_role_dict, save_dir="."):
    """
    表1：Top1_SourceFile別の出現割合を計算し、PresenterとRoleを付加してCSV出力。
    """
    source_counts = df['Top1_SourceFile'].value_counts().reset_index()
    source_counts.columns = ['SourceFile', 'Count']

    total = source_counts['Count'].sum()
    source_counts['Percentage'] = source_counts['Count'] / total * 100

    source_counts['Presenter'] = source_counts['SourceFile'].map(
        lambda x: presenter_role_dict.get(x, {}).get("Presenter", "Unknown"))
    source_counts['Role'] = source_counts['SourceFile'].map(
        lambda x: presenter_role_dict.get(x, {}).get("Role", "Unknown"))

    summary_df = source_counts[['SourceFile', 'Presenter', 'Role', 'Percentage']]

    # 保存
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "top1_sourcefile_summary.csv")
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return summary_df, csv_path


def summarize_top1_by_role(summary_df, save_dir="."):
    """
    表2：RoleごとのTop1出現割合と、対応するPresenter一覧を含む表を生成してCSV保存。
    """
    # Percentage集計
    role_summary = summary_df.groupby('Role')['Percentage'].sum().reset_index()
    role_summary.columns = ['Role', 'TotalPercentage']

    # PresenterListを作成
    role_to_presenters = summary_df.groupby('Role')['Presenter'].unique().reset_index()
    role_to_presenters.columns = ['Role', 'PresenterList']

    # PresenterListを文字列に変換（見やすくする）
    role_to_presenters['PresenterList'] = role_to_presenters['PresenterList'].apply(lambda x: ", ".join(sorted(set(x))))

    # 結合
    merged = pd.merge(role_summary, role_to_presenters, on='Role')

    # 保存
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "top1_role_summary.csv")
    merged.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return merged, csv_path




def summarize(csv_path, presenter_role_dict):
    
    df = pd.read_csv(csv_path)

    # 表1を生成・保存
    table1, path1 = summarize_top1_by_source(df, presenter_role_dict, save_dir="/Users/rintrin/codes/emorilab_climate_assembly/analysis_results/each_sentence_all_files")

    # 表2を生成・保存
    table2, path2 = summarize_top1_by_role(table1, save_dir="/Users/rintrin/codes/emorilab_climate_assembly/analysis_results/each_sentence_all_files")

    print("表1の保存先:", path1)
    print("表2の保存先:", path2)
    
    return path1, path2