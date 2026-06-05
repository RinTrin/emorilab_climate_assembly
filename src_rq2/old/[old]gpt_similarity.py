import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import pandas as pd
from pydantic import BaseModel, Field
from openai import OpenAI

from utils import pickle_load

ROOT = "/Users/rintrin/codes/emorilab_climate_assembly"


class PickBest(BaseModel):
    best_id: int = Field(...)


def extract_sentences(obj: Any) -> List[str]:
    if obj is None:
        return []
    if isinstance(obj, list):
        if not obj:
            return []
        if all(isinstance(x, str) for x in obj):
            return [x.strip() for x in obj if x and x.strip()]
        if all(isinstance(x, dict) for x in obj):
            keys = ("sentence", "text", "sent", "utterance", "content")
            out: List[str] = []
            for d in obj:
                for k in keys:
                    v = d.get(k)
                    if isinstance(v, str) and v.strip():
                        out.append(v.strip())
                        break
            return out
    if isinstance(obj, dict):
        for k in ("sentences", "sentence_list", "texts", "items", "data"):
            if k in obj:
                return extract_sentences(obj[k])
    if isinstance(obj, str) and obj.strip():
        return [obj.strip()]
    return []


def build_pklname_to_meta(presenter_role_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for lecture_key, info in presenter_role_dict.items():
        if not isinstance(info, dict):
            continue
        p = info.get("Pickle")
        if isinstance(p, str) and p:
            out[Path(p).name] = {
                "lecture_key": lecture_key,
                "presenter": info.get("Presenter"),
                "role": info.get("Role"),
            }
    return out


def build_input_candidates(
    input_sentences_pkl: List[str],
    pklname_to_meta: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    cid = 0
    for pkl_path in input_sentences_pkl:
        base = Path(pkl_path).name
        meta = pklname_to_meta.get(base, {})
        sents = extract_sentences(pickle_load(pkl_path))
        for idx, sent in enumerate(sents):
            if not sent:
                continue
            candidates.append(
                {
                    "id": cid,
                    "sentence": sent,  # ← clipしない
                    "input_pkl": str(pkl_path),
                    "sentence_idx": idx,
                    "lecture_key": meta.get("lecture_key"),
                    "presenter": meta.get("presenter"),
                    "role": meta.get("role"),
                }
            )
            cid += 1
    return candidates

import time
from openai import RateLimitError

def gpt_pick_best_id_with_retry(
    client,
    model,
    action_sentence,
    candidates_text,
    temperature,
    max_retries=3,
    base_sleep=1.0,
):
    last_err = None
    for attempt in range(max_retries):
        try:
            return gpt_pick_best_id(client, model, action_sentence, candidates_text, temperature)
        except RateLimitError as e:
            last_err = e
            # ざっくり指数バックオフ（エラーメッセージの秒数を厳密パースしない簡易版）
            time.sleep(base_sleep * (1.5 ** attempt))
        except Exception as e:
            # TPM以外ならそのまま投げる（原因究明のため）
            raise
    raise last_err

def build_candidates_payload(
    candidates: List[Dict[str, Any]],
    *,
    id_key: str = "id",
    sentence_key: str = "sentence",
) -> List[Dict[str, Any]]:
    """
    candidates を LLM に渡すための構造化ペイロードに変換する。
    返り値は [{"id": int, "sentence": str}, ...]
    """
    payload: List[Dict[str, Any]] = []
    for c in candidates:
        if id_key not in c or sentence_key not in c:
            raise KeyError(f"Candidate must have keys '{id_key}' and '{sentence_key}': {c.keys()}")

        # id は int 前提（ここで強制変換しておくと後段が安定）
        cid = int(c[id_key])
        sent = str(c[sentence_key])

        payload.append({"id": cid, "sentence": sent})
    

    target_ids = [141, 199, 1077, 1121, 329, 667]

    id2sent = {p["id"]: p["sentence"] for p in payload}

    for idx in target_ids:
        sent = id2sent.get(idx)
        print(idx, sent)
    ghjk
    
    return payload


def gpt_pick_best_id(
    client: OpenAI,
    model: str,
    action_sentence: str,
    candidates: List[Dict],
    temperature: float,
) -> PickBest:
    
    candidates_payload = build_candidates_payload(candidates)
    valid_ids: Set[int] = {c["id"] for c in candidates_payload}
    if not candidates_payload:
        raise ValueError("candidates is empty.")
    
    system_msg = (
        "あなたは日本語テキストの意味類似度判定アシスタントです。"
        "与えられたアクション文に最も意味が近い候補文を、候補一覧から必ず1つ選んでください。"
        "候補にない内容は作らず、候補のidだけを返してください。"
    )

    user_payload = {
        "action_sentence": action_sentence,
        "candidates": candidates_payload,
        "instruction": (
            "candidates の中から最も意味が近い1つを必ず選び、その id を best_id として返してください。"
            "候補の文面を出力しないでください。"
        ),
    }

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        text_format=PickBest,
        temperature=temperature,
    )

    # parse を使っている前提：パース済みモデルが取れる
    result: PickBest = resp.output_parsed  # ここはSDK差異があるなら適宜調整

    # 最重要：候補にない id を返したら即落とす（“創作ID”を完全に封じる）
    if result.best_id not in valid_ids:
        raise ValueError(
            f"Model returned invalid best_id={result.best_id}. "
            f"Valid ids include: {sorted(valid_ids)[:20]}{'...' if len(valid_ids) > 20 else ''}"
        )
    
    print(action_sentence)
    print(result)
    
    return result
    
    # PRICES_PER_1M = {
    #     "gpt-4o":   {"in": 2.50, "cin": 1.25,  "out": 10.00},  # Standard
    #     "gpt-5.2-2025-12-11": {"in": 1.75, "cin": 0.175, "out": 14.00},  # Standard（gpt-5.2と同枠）
    # }

    # def estimate_cost(model, input_tokens, output_tokens, cached_input_tokens=0, usd_jpy=157.0):
    #     """
    #     1回のAPI呼び出しコストを試算する。
    #     - input_tokens: 総入力トークン（cached分も含む）
    #     - cached_input_tokens: そのうちキャッシュ割引が効く入力トークン
    #     - output_tokens: 出力トークン
    #     - usd_jpy: 為替（円換算したいとき）
    #     """
    #     p = PRICES_PER_1M[model]
    #     cached_input_tokens = min(cached_input_tokens, input_tokens)

    #     usd = ((input_tokens - cached_input_tokens) * p["in"]
    #         + cached_input_tokens * p["cin"]
    #         + output_tokens * p["out"]) / 1_000_000

    #     return {"usd": usd, "jpy": usd * usd_jpy}
    
    # for mod in ["gpt-4o", "gpt-5.2-2025-12-11"]:
    #     usage = resp.usage
    #     print(usage)

    #     in_tok  = getattr(usage, "input_tokens", None)
    #     out_tok = getattr(usage, "output_tokens", None)

    #     # キャッシュ内訳があれば拾う（無ければ0扱い）
    #     cached_in_tok = getattr(usage, "cached_input_tokens", 0)

    #     if in_tok is None or out_tok is None:
    #         # ここに来たら usage のフィールド名が違う可能性が高いので、
    #         # print(usage) の結果を見て該当キーに合わせてください。
    #         raise ValueError("resp.usage に input_tokens / output_tokens が見つかりませんでした。")

    #     a = estimate_cost(mod, in_tok, out_tok, cached_in_tok)
    #     print(a)
    # ghjk
    
    # return resp.output_parsed


def select_similar_sentence(
    action_sentences_pkl: List[str],
    input_sentences_pkl: List[str],
    presenter_role_dict: Dict[str, Any],
    city_name: str,
    actionplan_excel_sheetname: str,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    out_csv_path: Optional[str] = None,
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in os.environ.")
    client = OpenAI(api_key=api_key)

    if len(action_sentences_pkl) != 1:
        raise ValueError("action_sentences_pkl must contain exactly one pickle path.")
    action_pkl = action_sentences_pkl[0]

    pklname_to_meta = build_pklname_to_meta(presenter_role_dict)
    candidates = build_input_candidates(input_sentences_pkl, pklname_to_meta)
    if not candidates:
        raise ValueError("No input candidates.")
    
    action_sents = [s for s in extract_sentences(pickle_load(action_pkl)) if s]
    print("action", len(action_sents))
    if not action_sents:
        raise ValueError("No action sentences.")

    rows = []
    for i, action_sentence in enumerate(action_sents):
        out = gpt_pick_best_id_with_retry(client, model, action_sentence, candidates, temperature)
        chosen = next((c for c in candidates if c["id"] == out.best_id), candidates[0])

        rows.append(
            {
                "city_name": city_name,
                "actionplan_excel_sheetname": actionplan_excel_sheetname,
                "action_idx": i,
                "action_sentence": action_sentence,
                "matched_input_sentence": chosen["sentence"],
                "matched_input_pkl": chosen["input_pkl"],
                "matched_input_sentence_idx": chosen["sentence_idx"],
                "matched_lecture_key": chosen.get("lecture_key"),
                "matched_presenter": chosen.get("presenter"),
                "matched_role": chosen.get("role"),
            }
        )

    df = pd.DataFrame(rows)

    if out_csv_path is None:
        out_csv_path = str(
            Path(ROOT) / "analysis_results" / "similar_sentence_gpt" / city_name / actionplan_excel_sheetname / "analyzed_similar_sentence.csv"
        )
    Path(Path(out_csv_path).parent).mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
    return out_csv_path
