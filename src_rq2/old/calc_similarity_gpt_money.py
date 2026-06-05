import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Set, Tuple
import json
import pandas as pd
from pydantic import BaseModel, Field
from openai import OpenAI

from utils import pickle_load

ROOT = "/Users/rintrin/codes/emorilab_climate_assembly"


class PickBest(BaseModel):
    best_id: int = Field(...)


class NAJudge(BaseModel):
    decision: Literal["OK", "N/A"]


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
                    "sentence": sent,
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
            print("RateLimitError")
            time.sleep(base_sleep * (1.5 ** attempt))
        except Exception:
            raise
    raise last_err


def build_candidates_payload(
    candidates: List[Dict[str, Any]],
    *,
    id_key: str = "id",
    sentence_key: str = "sentence",
) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for c in candidates:
        if id_key not in c or sentence_key not in c:
            raise KeyError(f"Candidate must have keys '{id_key}' and '{sentence_key}': {c.keys()}")

        cid = int(c[id_key])
        sent = str(c[sentence_key])
        payload.append({"id": cid, "sentence": sent})

    return payload


def get_response_token_usage(resp) -> Dict[str, int]:
    """
    OpenAI Responses API のレスポンスから token usage を安全に取り出す。
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens) or (input_tokens + output_tokens)

    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
    }


def estimate_gpt54_request_cost_usd(
    input_tokens: int,
    output_tokens: int,
    *,
    threshold_input_tokens: int = 272_000,
    base_input_price_per_1m: float = 2.50,
    base_output_price_per_1m: float = 15.00,
    over_threshold_input_multiplier: float = 2.0,
    over_threshold_output_multiplier: float = 1.5,
) -> float:
    """
    1リクエスト単位で GPT-5.4 の料金をUSD試算する。
    input_tokens が 272K を超えたら、そのリクエスト全体に対して
    input 2x, output 1.5x の単価を適用する。
    """
    input_price = base_input_price_per_1m
    output_price = base_output_price_per_1m

    if input_tokens > threshold_input_tokens:
        input_price *= over_threshold_input_multiplier
        output_price *= over_threshold_output_multiplier

    return (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price


def gpt_pick_best_id(
    client: OpenAI,
    model: str,
    action_sentence: str,
    candidates: List[Dict[str, Any]],
    temperature: float,
) -> Tuple[PickBest, Dict[str, int], float]:
    candidates_payload = build_candidates_payload(candidates)
    valid_ids: Set[int] = {c["id"] for c in candidates_payload}
    if not candidates_payload:
        raise ValueError("candidates is empty.")

    system_msg = (
        "action_sentenceに最も類似度の高い1文を、候補一覧から必ず1つ選んでください。"
        "候補にない内容は作らず、候補のidだけを返してください。"
    )

    user_payload = {
        "action_sentence": action_sentence,
        "candidates": candidates_payload,
        "instruction": (
            "action_sentenceに最も類似度の高い1文を、candidatesの中から必ず1つ選び、そのidをbest_idとして返してください。"
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
        # temperature=temperature,
        reasoning={"effort": "high"},
    )

    usage_dict = get_response_token_usage(resp)
    estimated_cost_usd = estimate_gpt54_request_cost_usd(
        input_tokens=usage_dict["input_tokens"],
        output_tokens=usage_dict["output_tokens"],
    )

    result: PickBest = resp.output_parsed

    if result.best_id not in valid_ids:
        raise ValueError(
            f"Model returned invalid best_id={result.best_id}. "
            f"Valid ids include: {sorted(valid_ids)[:20]}{'...' if len(valid_ids) > 20 else ''}"
        )

    return result, usage_dict, estimated_cost_usd


def gpt_delete_non_similar(
    client,
    model,
    action_sentence,
    cand_text,
) -> Tuple[bool, Dict[str, int], float]:
    resp = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "あなたは施策同一性の判定器です。action_sentenceとcandidateが同一施策ならOK、違えばN/Aを返してください。\n"
                    "判定は次の考え方に従うこと：\n"
                    "【最重要】何をどう変える施策かが一致しているか。\n"
                    "【重要】対象（何に対する施策か）と方向性・目的が整合しているか。\n"
                    "【補助】数値・期限・場所・主体は一致必須ではないが、明確な矛盾があればN/A。\n"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "action_sentence": action_sentence,
                        "candidate": cand_text,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        text_format=NAJudge,
        reasoning={"effort": "high"},
    )

    usage_dict = get_response_token_usage(resp)
    estimated_cost_usd = estimate_gpt54_request_cost_usd(
        input_tokens=usage_dict["input_tokens"],
        output_tokens=usage_dict["output_tokens"],
    )

    return resp.output_parsed.decision == "OK", usage_dict, estimated_cost_usd


def select_similar_sentence(
    action_sentences_pkl: List[str],
    input_sentences_pkl: List[str],
    presenter_role_dict: Dict[str, Any],
    city_name: str,
    actionplan_excel_sheetname: str,
    model: str = "gpt-5.4",
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

    total_input_tokens = 0
    total_output_tokens = 0
    total_estimated_cost_usd = 0.0
    request_count = 0

    rows = []
    for i, action_sentence in enumerate(action_sents):
        if i > 10:
            ghjk

        out, usage_pick, cost_pick = gpt_pick_best_id_with_retry(
            client, model, action_sentence, candidates, temperature
        )
        total_input_tokens += usage_pick["input_tokens"]
        total_output_tokens += usage_pick["output_tokens"]
        total_estimated_cost_usd += cost_pick
        request_count += 1

        chosen = next((c for c in candidates if c["id"] == out.best_id), candidates[0])

        na_ok, usage_na, cost_na = gpt_delete_non_similar(
            client, model, action_sentence, chosen["sentence"]
        )
        total_input_tokens += usage_na["input_tokens"]
        total_output_tokens += usage_na["output_tokens"]
        total_estimated_cost_usd += cost_na
        request_count += 1

        print(action_sentence)
        print(out)
        print(chosen)
        print(na_ok)
        print(
            f"[pick_best] input_tokens={usage_pick['input_tokens']}, "
            f"output_tokens={usage_pick['output_tokens']}, "
            f"cost_usd=${cost_pick:.6f}, "
            f"over_272k={usage_pick['input_tokens'] > 272000}"
        )
        print(
            f"[na_judge ] input_tokens={usage_na['input_tokens']}, "
            f"output_tokens={usage_na['output_tokens']}, "
            f"cost_usd=${cost_na:.6f}, "
            f"over_272k={usage_na['input_tokens'] > 272000}"
        )
        print("=====")

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

    print("=== GPT-5.4 cost estimate summary ===")
    print(f"model: {model}")
    print(f"request_count: {request_count}")
    print(f"total_input_tokens: {total_input_tokens}")
    print(f"total_output_tokens: {total_output_tokens}")
    print(f"total_tokens: {total_input_tokens + total_output_tokens}")
    print(f"estimated_total_cost_usd: ${total_estimated_cost_usd:.6f}")

    if out_csv_path is None:
        out_csv_path = str(
            Path(ROOT)
            / "analysis_results"
            / "similar_sentence_gpt"
            / city_name
            / actionplan_excel_sheetname
            / "analyzed_similar_sentence.csv"
        )
    Path(Path(out_csv_path).parent).mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
    return out_csv_path