# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, json, time
from typing import Dict, List, Any

# 載入 .env 檔案中的環境變數
from dotenv import load_dotenv
load_dotenv()

# ====== LLM（生成端，使用新的 Google Genai SDK） ======
from google import genai
genai_client = genai.Client(api_key="AIzaSyBpcEbaDBdr1EQlD1Ee9F3NrihKbzG2aj8")

# ====== Cohere：Judge + Rerank ======
import cohere
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
CO = cohere.Client(api_key=COHERE_API_KEY) if COHERE_API_KEY else None
COHERE_JUDGE_MODEL = "command-r-plus"
COHERE_RERANK_MODEL = "rerank-english-v3.0"  # 多語可用 v3.0-multilingual

# ====== Braintrust：實驗初始化／上傳 ======
import braintrust as bt

# ====== RAGAS：使用非 LLM 指標進行評估 ======
# 根據官方文件，使用 BleuScore、ROUGE 等非 LLM 指標

def ragas_non_llm_scores(sample: Dict[str, Any], model_text: str) -> Dict[str, float]:
    """使用 RAGAS 的非 LLM 指標進行評估，避免需要 OpenAI API Key"""
    try:
        from ragas import SingleTurnSample
        from ragas.metrics import BleuScore, RougeScore, StringPresence, ExactMatch
        
        # 準備評估資料
        contexts = [sample["question"], sample["explanation"]] + list(sample["options"].values())
        context_text = " ".join(contexts)
        
        # 建立測試資料
        test_data = {
            "user_input": f"根據題目與解釋生成教師回饋\n{context_text}",
            "response": model_text,
            "reference": context_text  # 使用上下文作為參考
        }
        
        test_sample = SingleTurnSample(**test_data)
        
        # 使用多種非 LLM 指標
        bleu_metric = BleuScore()
        rouge_metric = RougeScore()
        presence_metric = StringPresence()
        exact_metric = ExactMatch()
        
        # 計算各項分數
        bleu_score = bleu_metric.single_turn_score(test_sample)
        rouge_score = rouge_metric.single_turn_score(test_sample)
        presence_score = presence_metric.single_turn_score(test_sample)
        exact_score = exact_metric.single_turn_score(test_sample)
        
        # 將分數映射到 RAGAS 指標
        return {
            "faithfulness": min(1.0, (bleu_score + rouge_score) / 2),  # 文本相似度作為忠實度
            "context_precision": presence_score,  # 關鍵詞出現作為精確度
            "context_recall": presence_score,     # 簡化為與精確度相同
            "answer_relevancy": exact_score       # 精確匹配作為相關性
        }
        
    except Exception as e:
        print(f"[WARN] RAGAS 非 LLM 評估失敗: {e}")
        print("[WARN] 使用預設分數")
        return {
            "faithfulness": 0.5,
            "context_precision": 0.5,
            "context_recall": 0.5,
            "answer_relevancy": 0.5
        }


# ---------------------------
# 1) 測資（放你的題庫；下方會自動造 正/誤 兩種作答）
# ---------------------------
ITEMS: List[Dict[str, Any]] = [
    {
        "id": "M7A-0001",
        "grade": "7A",
        "subject": "數學",
        "chapter": "1-1正數與負數",
        "topic": "正數與負數",
        "knowledge_point": ["正負數的定義", "數線表示"],
        "question": "下列關於正數與負數的敘述，何者正確？",
        "options": {
            "A": "0 是正數。",
            "B": "0 是負數。",
            "C": "0 既不是正數也不是負數。",
            "D": "0 是最小的正數。"
        },
        "answer": "C",
        "explanation": "0 既不是正數也不是負數，它是正負數的分界點。"
    },
    {
        "id": "M7A-0002",
        "grade": "7A",
        "subject": "數學",
        "chapter": "1-1正數與負數",
        "topic": "正數與負數",
        "knowledge_point": ["絕對值", "定義"],
        "question": "下列哪一個數的絕對值最大？",
        "options": {"A": "-5","B": "3","C": "-8","D": "6"},
        "answer": "C",
        "explanation": "絕對值是指數線上該點到原點的距離。|-5|=5, |3|=3, |-8|=8, |6|=6。"
    }
]

def make_samples(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """為每題產生 正確/錯誤 兩份學生作答"""
    samples = []
    for it in items:
        samples.append({**it, "student_answer": it["answer"], "sid": it["id"] + "-correct"})
        wrong = next(k for k in it["options"].keys() if k != it["answer"])
        samples.append({**it, "student_answer": wrong, "sid": it["id"] + "-wrong"})
    return samples

SAMPLES = make_samples(ITEMS)

# ---------------------------
# 2) 生成 Prompt（固定 JSON→再文字）
# ---------------------------
PROMPT_TMPL = """你是一位國中數學老師。只根據下列資料回饋，禁止使用外部知識。
請先輸出 JSON（見格式），再輸出一段給學生看的短評（先肯定再建議）。

[題目]
{question}

[選項]
{opts}

[正確答案]
{correct}

[學生作答]
{student}

[解釋]
{expl}

【必填 JSON】
{{
  "correctness": true|false,
  "reason": "需引用題幹或解釋關鍵語",
  "references": {{
    "student_option": "A|B|C|D",
    "correct_option": "A|B|C|D",
    "quotes": ["從解釋或正確選項複述1句"]
  }},
  "next_steps": ["動詞開頭步驟1", "步驟2"],
  "confidence": 0.0
}}
"""

def call_llm(sample: Dict[str, Any], temperature: float = 0.2) -> str:
    opts = "\n".join([f"{k}. {v}" for k, v in sample["options"].items()])
    prompt = PROMPT_TMPL.format(
        question=sample["question"],
        opts=opts,
        correct=sample["answer"],
        student=sample["student_answer"],
        expl=sample["explanation"],
    )
    # 若無 GEMINI_API_KEY，或產生失敗，回傳可解析的本地結果
    def _fallback_text() -> str:
        is_correct = sample["student_answer"] == sample["answer"]
        refs = {
            "student_option": sample["student_answer"],
            "correct_option": sample["answer"],
            "quotes": [sample["explanation"][:60]]
        }
        out = {
            "correctness": is_correct,
            "reason": "依據解釋判斷作答正確性，並引用解釋要點。",
            "references": refs,
            "next_steps": [
                "複習關鍵定義",
                "在數線上標示例題以加深概念"
            ],
            "confidence": 0.2
        }
        short = ("做得不錯！" if is_correct else "再接再厲！") + " 根據題意與解釋調整思路，逐步驗算即可。"
        return json.dumps(out, ensure_ascii=False) + "\n\n" + short

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[WARN] 未偵測到 GEMINI_API_KEY，使用本地 fallback 模式")
        return _fallback_text()
    try:
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"[WARN] Gemini API 呼叫失敗: {e}")
        print("[WARN] 使用本地 fallback 模式")
        return _fallback_text()

JSON_PAT = re.compile(r"\{[\s\S]*?\}")
def parse_output(text: str) -> Dict[str, Any]:
    # 先嘗試從第一個 '{' 起用 raw_decode 解析最外層 JSON 物件
    first_brace = text.find("{")
    if first_brace != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[first_brace:])
            return obj
        except Exception:
            pass
    # 後備：使用較寬鬆的正則抓到第一個物件
    m = JSON_PAT.search(text)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    raise RuntimeError("無法在模型輸出中找到可解析的 JSON。")

# ---------------------------
# 3) Cohere：Judge（0~1）與 Rerank 引用對位
# ---------------------------
def cohere_judge_score(system_prompt: str, user_payload: Dict[str, Any]) -> float:
    """要求 Cohere 以 JSON 回傳 {"score": 0~1, "explanation": "..."}"""
    if CO is None:
        return 0.0
    
    # Cohere Trial 版本限制：每分鐘 10 次，加入延遲避免觸發限制
    time.sleep(6)  # 6 秒延遲，確保每分鐘不超過 10 次
    
    try:
        resp = CO.chat(model=COHERE_JUDGE_MODEL,
                       message=system_prompt + "\n\n" + json.dumps(user_payload, ensure_ascii=False),
                       temperature=0)
        text = resp.text
    except TypeError:
        # SDK 版本差異 fallback
        resp = CO.chat(model=COHERE_JUDGE_MODEL,
                       messages=[{"role":"user","content":system_prompt+"\n\n"+json.dumps(user_payload, ensure_ascii=False)}],
                       temperature=0)
        text = resp.message.content[0].text
    # 解析分數
    try:
        j = json.loads(re.search(r"\{[\s\S]*\}", text).group(0))
        score = float(j.get("score", 0.0))
    except Exception:
        m = re.search(r"0?\.\d+|1(?:\.0+)?", text)
        score = float(m.group(0)) if m else 0.0
    return max(0.0, min(1.0, score))

def judge_reason_grounded(sample: Dict[str, Any], out_json: Dict[str, Any]) -> float:
    system = ("You are a strict grading judge (0~1). "
              "Score higher only if REASON/QUOTES are supported by sources. "
              'Return JSON: {"score": <0..1>, "explanation": "..."}')
    payload = {
        "question": sample["question"],
        "explanation": sample["explanation"],
        "options": sample["options"],
        "student_answer": sample["student_answer"],
        "model_output": out_json
    }
    return cohere_judge_score(system, payload)

def judge_actionability(out_json: Dict[str, Any]) -> float:
    system = ("Pedagogy judge (0~1). Are next_steps concrete/actionable/verifiable? "
              'Return JSON: {"score": <0..1>, "explanation": "..."}')
    return cohere_judge_score(system, {"next_steps": out_json.get("next_steps", [])})

def judge_tone(out_json: Dict[str, Any]) -> float:
    system = ("Tone judge (0~1). Supportive & constructive gets higher score. "
              'Return JSON: {"score": <0..1>, "explanation": "..."}')
    return cohere_judge_score(system, {"text": out_json})

def citation_precision_with_rerank(sample: Dict[str, Any], out_json: Dict[str, Any], threshold: float = 0.5) -> float:
    if CO is None:
        return 0.0
    quotes: List[str] = out_json.get("references", {}).get("quotes", [])
    if not quotes:
        return 0.0
    contexts = [sample["question"], sample["explanation"]] + list(sample["options"].values())
    hit = 0
    for q in quotes:
        rr = CO.rerank(model=COHERE_RERANK_MODEL, query=q, documents=contexts, top_n=1)
        top = rr.results[0]
        # 命中條件：指向解釋/正確選項 或 分數達門檻
        if top.index == 1 or contexts[top.index] == sample["options"][sample["answer"]] or top.relevance_score >= threshold:
            hit += 1
    return hit / len(quotes)

# ---------------------------
# 4) RAGAS：把題幹/解釋/選項當 contexts 量「有無據」
# ---------------------------
def ragas_scores(sample: Dict[str, Any], model_text: str) -> Dict[str, float]:
    contexts = [sample["question"], sample["explanation"]] + list(sample["options"].values())
    dataset = [{
        "question": "根據題目與解釋生成的教師回饋",
        "answer": model_text,
        "contexts": contexts
    }]
    res = ragas_eval(dataset, metrics=[faithfulness, context_precision, context_recall, answer_relevancy])
    return {m.name: float(res[m.name]) for m in res}

# ---------------------------
# 5) Braintrust：實驗初始化（無 Key → 離線模式）
# ---------------------------
def init_braintrust_experiment(project: str, experiment: str):
    api_key = os.getenv("BRAINTRUST_API_KEY", "")
    if not api_key:
        print("[WARN] 未偵測到 BRAINTRUST_API_KEY，將以離線模式執行（不會上傳到 Braintrust）。")
        return None
    exp = bt.init(
        project=project,
        experiment=experiment,
        api_key=api_key,
        metadata={"owner": "edu-feedback", "stack": "cohere+ragas"},
    )
    return exp

# ---------------------------
# 6) 主流程：逐筆評估 +（可選）上傳 Braintrust
# ---------------------------
def evaluate_one(sample: Dict[str, Any]) -> Dict[str, Any]:
    raw_text = call_llm(sample, temperature=0.2)
    out_json = parse_output(raw_text)

    # RAGAS 非 LLM 評估
    rag = ragas_non_llm_scores(sample, raw_text)

    # Cohere LLM-as-Judge & Rerank
    reason_llm = judge_reason_grounded(sample, out_json)
    action_llm = judge_actionability(out_json)
    tone_llm = judge_tone(out_json)
    cite_prec = citation_precision_with_rerank(sample, out_json, threshold=0.5)

    # 最小規則：格式合規、對錯一致
    def format_ok(j: Dict[str, Any]) -> float:
        need = {"correctness","reason","references","next_steps","confidence"}
        if not need.issubset(j.keys()): return 0.0
        refs = j["references"]
        if not {"student_option","correct_option","quotes"}.issubset(refs.keys()): return 0.0
        if not isinstance(j["next_steps"], list) or len(j["next_steps"]) < 2: return 0.0
        return 1.0
    def correctness_ok(s: Dict[str, Any], j: Dict[str, Any]) -> float:
        gt = s["student_answer"] == s["answer"]
        return 1.0 if bool(j.get("correctness")) == gt else 0.0

    f_ok = format_ok(out_json)
    c_ok = correctness_ok(sample, out_json)

    # 綜合分（可調權重）
    final = (
        0.25 * rag["faithfulness"] +
        0.10 * rag["context_precision"] +
        0.10 * rag["context_recall"] +
        0.10 * rag["answer_relevancy"] +
        0.15 * reason_llm +
        0.05 * cite_prec +
        0.10 * f_ok +
        0.10 * c_ok +
        0.03 * action_llm +
        0.02 * tone_llm
    )

    return {
        "sid": sample["sid"],
        "model_text": raw_text,
        "model_json": out_json,
        "scores": {
            "faithfulness": rag["faithfulness"],
            "ctx_precision": rag["context_precision"],
            "ctx_recall": rag["context_recall"],
            "relevancy": rag["answer_relevancy"],
            "reason_llm": reason_llm,
            "citation_precision": cite_prec,
            "format_ok": f_ok,
            "correctness_ok": c_ok,
            "actionability_llm": action_llm,
            "tone_llm": tone_llm,
            "final_score": final
        }
    }

def main():
    project = "Sunny-Edu-Feedback"     # 你在 Braintrust 的專案名
    experiment = f"cohere-ragas-v1-{int(time.time())}"
    exp = init_braintrust_experiment(project, experiment)

    print(f"Total samples: {len(SAMPLES)}")
    for s in SAMPLES:
        r = evaluate_one(s)
        # === 上傳到 Braintrust（若有金鑰） ===
        if exp is not None:
            bt.log(
                input={
                    "sid": r["sid"],
                    "question": s["question"],
                    "options": s["options"],
                    "correct_answer": s["answer"],
                    "student_answer": s["student_answer"],
                    "explanation": s["explanation"],
                },
                output={
                    "model_json": r["model_json"],
                    "model_text": r["model_text"],
                },
                scores=r["scores"],
                metadata={"grade": s["grade"], "topic": s["topic"]}
            )
        # === 終端列印（離線或輔助觀察） ===
        sc = r["scores"]
        print(f"\n=== {r['sid']} 詳細評分 ===")
        print(f"最終分數: {sc['final_score']:.3f}")
        print(f"RAGAS - 忠實度: {sc['faithfulness']:.3f}")
        print(f"RAGAS - 上下文精確度: {sc['ctx_precision']:.3f}")
        print(f"RAGAS - 上下文召回率: {sc['ctx_recall']:.3f}")
        print(f"RAGAS - 答案相關性: {sc['relevancy']:.3f}")
        print(f"Cohere - 理由根據性: {sc['reason_llm']:.3f}")
        print(f"Cohere - 引用精確度: {sc['citation_precision']:.3f}")
        print(f"格式檢查: {sc['format_ok']:.1f}")
        print(f"正確性判斷: {sc['correctness_ok']:.1f}")
        print(f"Cohere - 可執行性: {sc['actionability_llm']:.3f}")
        print(f"Cohere - 語氣評分: {sc['tone_llm']:.3f}")
        print(f"AI 生成內容: {r['model_text'][:200]}...")
        print("=" * 50)

    if exp is not None:
        bt.flush()  # 送出批次事件
        print(f"[OK] 已上傳到 Braintrust 專案：{project} / 實驗：{experiment}")

if __name__ == "__main__":
    main()
