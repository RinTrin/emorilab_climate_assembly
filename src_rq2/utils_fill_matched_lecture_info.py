from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

TARGET_COLUMNS = (
    "matched_lecture_key",
    "matched_presenter",
    "matched_role",
)


def _is_missing(value: Any) -> bool:
    """Return True for NaN, None, or a blank string."""
    return pd.isna(value) or (isinstance(value, str) and not value.strip())


def _extract_lecture_key(
    city_name: Any,
    matched_input_pkl: Any,
    inputmaterial_info: dict[str, dict[str, dict[str, Any]]],
) -> str | None:
    """Infer the lecture key from the PKL path and the YAML metadata.

    The text after the city directory is examined first. A filename-derived
    candidate is preferred; a longest-match fallback handles minor filename
    variations safely.
    """
    if _is_missing(city_name) or _is_missing(matched_input_pkl):
        return None

    city = str(city_name).strip()
    if city not in inputmaterial_info:
        return None

    normalized_path = str(matched_input_pkl).replace("\\", "/")
    city_marker = f"/{city}/"
    relevant_text = (
        normalized_path.split(city_marker, 1)[1]
        if city_marker in normalized_path
        else Path(normalized_path).name
    )

    city_lectures = inputmaterial_info[city]

    # Examples handled:
    # lecture2_3_youtube_txt_segmented.pkl
    # lecture4_2_2_youtube_txt.syusei_segmented.pkl_segmented.pkl
    candidate_match = re.search(
        r"(lecture[^/]*?)(?=_youtube|_segmented|\.pkl|$)",
        relevant_text,
        flags=re.IGNORECASE,
    )
    if candidate_match:
        candidate = candidate_match.group(1)
        if candidate in city_lectures:
            return candidate

    # Fallback for harmless deviations in filename suffixes. Longest first is
    # important for keys such as lecture2_1 and lecture2_1_3.
    matched_keys = [key for key in city_lectures if key in relevant_text]
    if not matched_keys:
        return None
    return max(matched_keys, key=len)


def fill_matched_lecture_info(
    csv_path: str | Path,
    yaml_path: str | Path,
    output_csv_path: str | Path | None = None,
    *,
    fail_on_unmatched: bool = True,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Fill blank matched lecture metadata columns in a CSV file.

    Existing non-blank values are never overwritten. The completed CSV is
    written to ``output_csv_path``. If it is omitted, the input CSV is updated
    in place.

    Parameters
    ----------
    csv_path:
        Path to the CSV that contains ``city_name`` and ``matched_input_pkl``.
    yaml_path:
        Path to ``inputmaterial_info.yaml``.
    output_csv_path:
        Optional output path. Omit it to overwrite ``csv_path``.
    fail_on_unmatched:
        If True, raise an error when a row that needs completion cannot be
        matched to a YAML lecture. If False, leave such cells blank.
    encoding:
        Encoding used to read and write the CSV.

    Returns
    -------
    pandas.DataFrame
        The completed dataframe.
    """
    csv_path = Path(csv_path)
    yaml_path = Path(yaml_path)
    output_csv_path = Path(output_csv_path) if output_csv_path else csv_path

    df = pd.read_csv(csv_path, encoding=encoding)
    with yaml_path.open("r", encoding="utf-8") as f:
        inputmaterial_info = yaml.safe_load(f)

    required_columns = {"city_name", "matched_input_pkl"}
    missing_required = required_columns - set(df.columns)
    if missing_required:
        raise ValueError(
            "CSVに必要な列がありません: " + ", ".join(sorted(missing_required))
        )

    if not isinstance(inputmaterial_info, dict):
        raise ValueError("YAMLの最上位は都市名をキーとする辞書である必要があります。")

    for column in TARGET_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
        # All three target columns store text. Explicit object dtype avoids
        # pandas warnings when a completely blank CSV column was inferred as float.
        df[column] = df[column].astype("object")

    unmatched_rows: list[int] = []

    for idx, row in df.iterrows():
        needs_completion = any(_is_missing(row[column]) for column in TARGET_COLUMNS)
        if not needs_completion:
            continue

        city = row["city_name"]
        city_key = str(city).strip() if not _is_missing(city) else ""

        lecture_key = (
            str(row["matched_lecture_key"]).strip()
            if not _is_missing(row["matched_lecture_key"])
            else _extract_lecture_key(city, row["matched_input_pkl"], inputmaterial_info)
        )

        lecture_info = inputmaterial_info.get(city_key, {}).get(lecture_key)
        if lecture_key is None or lecture_info is None:
            unmatched_rows.append(idx)
            continue

        if _is_missing(row["matched_lecture_key"]):
            df.at[idx, "matched_lecture_key"] = lecture_key
        if _is_missing(row["matched_presenter"]):
            df.at[idx, "matched_presenter"] = lecture_info.get("Presenter")
        if _is_missing(row["matched_role"]):
            df.at[idx, "matched_role"] = lecture_info.get("Role")

    if unmatched_rows and fail_on_unmatched:
        preview = df.loc[
            unmatched_rows[:10], ["city_name", "matched_input_pkl"]
        ].to_string(index=True)
        raise ValueError(
            f"YAMLと対応付けられない行が {len(unmatched_rows)} 行あります。"
            f"\n先頭10件:\n{preview}"
        )

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False, encoding=encoding)
    return df


if __name__ == "__main__":
    # Example: keep the original file and create a completed copy.
    completed = fill_matched_lecture_info(
        csv_path="gpt_check_comprehensive.csv",
        yaml_path="inputmaterial_info.yaml",
        output_csv_path="gpt_check_comprehensive_filled.csv",
    )
    print(completed[list(TARGET_COLUMNS)].isna().sum())
