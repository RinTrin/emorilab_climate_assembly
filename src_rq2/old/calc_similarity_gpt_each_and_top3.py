import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Set
from collections import defaultdict

import pandas as pd
from pydantic import BaseModel, Field
from openai import OpenAI, RateLimitError

from utils import pickle_load

ROOT = "/Users/rintrin/codes/emorilab_climate_assembly"


class PickBest(BaseModel):
    best_id: int = Field(...)


class NAJudge(BaseModel):
    decision: Literal["OK", "N/A"]


class RankTopK(BaseModel):
    ranked_ids: List[int] = Field(..., description="類似度順に並べたcandidate idの配列。最大3件。")


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


def group_candidates_by_expert(
    candidates: List[Dict[str, Any]],
    group_key: str = "input_pkl",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    各専門家ごとに候補文をまとめる。
    デフォルトでは input_pkl 単位でグルーピング。
    """
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        key = c.get(group_key)
        if key is None:
            continue
        grouped[str(key)].append(c)
    return dict(grouped)


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

    result: PickBest = resp.output_parsed

    if result.best_id not in valid_ids:
        raise ValueError(
            f"Model returned invalid best_id={result.best_id}. "
            f"Valid ids include: {sorted(valid_ids)[:20]}{'...' if len(valid_ids) > 20 else ''}"
        )

    return result


def gpt_pick_best_id_with_retry(
    client,
    model,
    action_sentence,
    candidates,
    temperature,
    max_retries=3,
    base_sleep=1.0,
):
    last_err = None
    for attempt in range(max_retries):
        try:
            return gpt_pick_best_id(client, model, action_sentence, candidates, temperature)
        except RateLimitError as e:
            last_err = e
            print("RateLimitError")
            time.sleep(base_sleep * (1.5 ** attempt))
        except Exception:
            raise
    raise last_err


def gpt_delete_non_similar(client, model, action_sentence, cand_text):
    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content":
                "あなたは施策同一性の判定器です。action_sentenceとcandidateが同一施策ならOK、違えばN/Aを返してください。\n"
                "判定は次の考え方に従うこと：\n"
                "【最重要】何をどう変える施策かが一致しているか。\n"
                "【重要】対象（何に対する施策か）と方向性・目的が整合しているか。\n"
                "【補助】数値・期限・場所・主体は一致必須ではないが、明確な矛盾があればN/A。\n"
            },
            {"role": "user", "content": json.dumps({
                "action_sentence": action_sentence,
                "candidate": cand_text
            }, ensure_ascii=False)},
        ],
        text_format=NAJudge,
        reasoning={"effort": "high"},
    )
    return resp.output_parsed.decision == "OK"


def gpt_rank_top_candidates(
    client: OpenAI,
    model: str,
    action_sentence: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 3,
) -> RankTopK:
    """
    NA判定でOKとなった「各専門家の代表文」を順位づけする。
    返り値は ranked_ids=[id1, id2, id3...] （最大top_k件）
    """
    if not candidates:
        return RankTopK(ranked_ids=[])

    payload = []
    valid_ids: Set[int] = set()

    for c in candidates:
        cid = int(c["id"])
        valid_ids.add(cid)
        payload.append(
            {
                "id": cid,
                "sentence": c["sentence"],
                "presenter": c.get("presenter"),
                "role": c.get("role"),
                "lecture_key": c.get("lecture_key"),
            }
        )

    system_msg = (
        "あなたは施策類似度の順位付け器です。"
        "action_sentenceに対して、candidatesの中から『より同一施策に近いもの』を上位に順位付けしてください。"
        "候補にない内容を作らず、idのみを ranked_ids として返してください。"
    )

    user_payload = {
        "action_sentence": action_sentence,
        "candidates": payload,
        "instruction": (
            f"action_sentenceに対して、candidatesを類似度の高い順に最大{top_k}件並べ、"
            "そのidを ranked_ids として返してください。"
            "順位付けでは、まず『何をどう変える施策か』の一致を最重視し、"
            "次に対象・方向性・目的の整合を重視してください。"
            "候補文の文面は出力せず、id配列だけ返してください。"
        ),
    }

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        text_format=RankTopK,
        reasoning={"effort": "high"},
    )

    result: RankTopK = resp.output_parsed

    if len(result.ranked_ids) > top_k:
        raise ValueError(f"Model returned too many ids: {result.ranked_ids}")

    if len(set(result.ranked_ids)) != len(result.ranked_ids):
        raise ValueError(f"Model returned duplicated ids: {result.ranked_ids}")

    invalid_ids = [x for x in result.ranked_ids if x not in valid_ids]
    if invalid_ids:
        raise ValueError(f"Model returned invalid ranked ids: {invalid_ids}")

    return result


def select_similar_sentence_top3_by_expert(
    action_sentences_pkl: List[str],
    input_sentences_pkl: List[str],
    presenter_role_dict: Dict[str, Any],
    city_name: str,
    actionplan_excel_sheetname: str,
    model_pick: str = "gpt-4o",
    model_na: str = "gpt-4o",
    model_rank: str = "gpt-4o",
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
    all_candidates = build_input_candidates(input_sentences_pkl, pklname_to_meta)
    if not all_candidates:
        raise ValueError("No input candidates.")

    # 専門家ごとに候補文をまとめる
    candidates_by_expert = group_candidates_by_expert(all_candidates, group_key="input_pkl")
    if not candidates_by_expert:
        raise ValueError("No grouped candidates.")

    action_sents = [s for s in extract_sentences(pickle_load(action_pkl)) if s]
    print("action", len(action_sents))
    if not action_sents:
        raise ValueError("No action sentences.")

    rows = []

    for i, action_sentence in enumerate(action_sents):
        # if i > 10:
        #     break

        print(f"===== action_idx={i} =====")
        print(action_sentence)

        # 1) 各専門家ごとに最類似1文を選ぶ
        per_expert_best_rows: List[Dict[str, Any]] = []
        for expert_key, expert_candidates in candidates_by_expert.items():
            out = gpt_pick_best_id_with_retry(
                client=client,
                model=model_pick,
                action_sentence=action_sentence,
                candidates=expert_candidates,
                temperature=temperature,
            )

            chosen = next(c for c in expert_candidates if c["id"] == out.best_id)
            is_ok = gpt_delete_non_similar(
                client=client,
                model=model_na,
                action_sentence=action_sentence,
                cand_text=chosen["sentence"],
            )

            one_row = {
                "id": chosen["id"],
                "sentence": chosen["sentence"],
                "input_pkl": chosen["input_pkl"],
                "sentence_idx": chosen["sentence_idx"],
                "lecture_key": chosen.get("lecture_key"),
                "presenter": chosen.get("presenter"),
                "role": chosen.get("role"),
                "na_decision": "OK" if is_ok else "N/A",
            }
            per_expert_best_rows.append(one_row)

            print("--- expert ---")
            print("expert_key:", expert_key)
            print("best_id:", out.best_id)
            print("presenter:", chosen.get("presenter"))
            print("sentence:", chosen["sentence"])
            print("NA:", "OK" if is_ok else "N/A")

        # 2) OKだけを集めて順位づけ
        ok_candidates = [x for x in per_expert_best_rows if x["na_decision"] == "OK"]

        if ok_candidates:
            rank_result = gpt_rank_top_candidates(
                client=client,
                model=model_rank,
                action_sentence=action_sentence,
                candidates=ok_candidates,
                top_k=3,
            )
            ranked_ids = rank_result.ranked_ids
        else:
            ranked_ids = []

        rank_map = {cid: rank + 1 for rank, cid in enumerate(ranked_ids)}

        # id -> row lookup
        id_to_row = {x["id"]: x for x in per_expert_best_rows}

        top1 = id_to_row[ranked_ids[0]] if len(ranked_ids) >= 1 else None
        top2 = id_to_row[ranked_ids[1]] if len(ranked_ids) >= 2 else None
        top3 = id_to_row[ranked_ids[2]] if len(ranked_ids) >= 3 else None

        # 3) action_sentenceごとの1行を作る
        row = {
            "city_name": city_name,
            "actionplan_excel_sheetname": actionplan_excel_sheetname,
            "action_idx": i,
            "action_sentence": action_sentence,

            # 参考: 各専門家のbestの全件JSON（確認用）
            "per_expert_best_json": json.dumps(per_expert_best_rows, ensure_ascii=False),

            # Top1
            "rank1_id": top1["id"] if top1 else None,
            "rank1_sentence": top1["sentence"] if top1 else None,
            "rank1_input_pkl": top1["input_pkl"] if top1 else None,
            "rank1_sentence_idx": top1["sentence_idx"] if top1 else None,
            "rank1_lecture_key": top1["lecture_key"] if top1 else None,
            "rank1_presenter": top1["presenter"] if top1 else None,
            "rank1_role": top1["role"] if top1 else None,

            # Top2
            "rank2_id": top2["id"] if top2 else None,
            "rank2_sentence": top2["sentence"] if top2 else None,
            "rank2_input_pkl": top2["input_pkl"] if top2 else None,
            "rank2_sentence_idx": top2["sentence_idx"] if top2 else None,
            "rank2_lecture_key": top2["lecture_key"] if top2 else None,
            "rank2_presenter": top2["presenter"] if top2 else None,
            "rank2_role": top2["role"] if top2 else None,

            # Top3
            "rank3_id": top3["id"] if top3 else None,
            "rank3_sentence": top3["sentence"] if top3 else None,
            "rank3_input_pkl": top3["input_pkl"] if top3 else None,
            "rank3_sentence_idx": top3["sentence_idx"] if top3 else None,
            "rank3_lecture_key": top3["lecture_key"] if top3 else None,
            "rank3_presenter": top3["presenter"] if top3 else None,
            "rank3_role": top3["role"] if top3 else None,
        }

        rows.append(row)
        print(row)

    df = pd.DataFrame(rows)

    if out_csv_path is None:
        out_csv_path = str(
            Path(ROOT)
            / "analysis_results"
            / "similar_sentence_gpt"
            / city_name
            / actionplan_excel_sheetname
            / "analyzed_similar_sentence_top3_by_expert.csv"
        )

    Path(Path(out_csv_path).parent).mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
    return out_csv_path