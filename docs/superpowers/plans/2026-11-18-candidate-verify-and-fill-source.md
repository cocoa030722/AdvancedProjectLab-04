# 추천 문장 검증과 채운 값의 출처 표시 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** AI가 추천 문장의 빈칸을 맥락으로 채우게 풀어 주되, 채운 값의 출처를 서버가 검증해 진행자에게 색으로 보여주고, 베끼거나 새 정보가 없거나 서로 겹치는 추천 문장은 1회 재생성 후 폐기한다.

**Architecture:** 같은 시기에 다른 작업자가 `decision_compiler.py`의 호출 흐름(추출·판정·생성 단계 분리)을 고치므로, **새 코드는 전부 새 파일에 넣고 `decision_compiler.py`는 끼워 넣는 줄만 고친다.** 검사·검증·표시용 순수 함수는 `decision_checks.py`(Django·LLM 모름, 필요한 정규식은 자체 구현 — 기존 코드를 옮기지 않는다. 중복은 머지 뒤 정리), LLM을 한 번 더 부르는 재생성은 `decision_refine.py`. 새 프롬프트 문구는 상수·함수로 빼서 기존 프롬프트 본문과 섞이지 않게 한다.

**Tech Stack:** Django 6.1, 표준 라이브러리만(`re`, `json`, `logging`), Django 템플릿 + base.html 기존 색 토큰

**Spec:** `docs/superpowers/specs/2026-11-18-candidate-verify-and-fill-source.md` (+ 2026-11-18 계획 승인 시 받은 수정 7건)

## Global Constraints

- **겹침 최소화가 최우선.** `decision_compiler.py`에서 허용되는 변경은 ① 합의한 두 문장 삭제 ② 새 상수·함수를 끼워 넣는 줄 ③ `parse_items`의 한 줄 호출 ④ `compile_specs`의 한 줄 호출 ⑤ 새 집계 칸. 기존 함수의 본문·순서·이름은 건드리지 않는다. 기존 정규식(`_DATE_EXPR` 등)과 `_bigrams`는 **그대로 둔다**.
- 진행자·팀원이 입력하거나 설정할 것을 늘리지 않는다.
- `MAX_INTERPRETATIONS = 1`, 진행자가 고르는 흐름(AI 추천 문장 | 직접 쓰기 | 다음 회의로 넘김 | 이 결정 빼기)은 그대로 유지한다.
- 제한 해제·fills·검증은 **애매(ambiguous: true) 항목의 추천 문장(candidate)에만** 적용한다. 명확 항목의 `final`은 원문 기준(`parse_items`의 근거 대조 + `safety_warnings`)을 그대로 둔다.
- `decision_checks.py`는 `django`·`services`·`decision_compiler`를 import하지 않는다.
- 테스트는 실제 LLM/API를 부르지 않는다 — 기존 방식대로 `patch.object(services, "_generate", ...)` + `patch.dict(os.environ, {"OLLAMA_MODEL": "m1"})`.
- 기존 테스트 522개는 전부 통과해야 한다. 어떤 기존 테스트라도 빨개지면 **고치지 말고 멈추고 보고한다**.
- 새 CSS 프레임워크·CDN 금지. `base.html`의 `--accent` / `--accent-soft` 토큰만 쓴다.
- 마이그레이션은 `0029_interpretation_fills` 하나(AddField)뿐이고 **단독 커밋**으로 올린다. 기존 칸은 바꾸거나 지우지 않는다.
- `services.py`는 고치지 않는다(재사용만). 커밋은 태스크마다 나눈다.
- 작업 중 `upstream/main`이 바뀌면 rebase해서 최신을 유지한다.

## Review Focus

1. **fills가 없거나 형식이 틀린 응답** — `fills` 누락, 문자열, `null`, 원소가 dict가 아닌 배열. 항목 하나의 형식 오류로 회의 전체를 실패시키지 않는 기존 원칙대로, 조용히 빈 목록이 되고 추천 문장은 그대로 살아야 한다. (Task 2)
2. **fills의 value가 추천 문장에 없는 경우** — AI가 문장과 다른 표기로 value를 보내면 강조할 자리가 없다. 강조 없이 문장 전체가 한 조각으로 그려지고, 경고 줄은 뜨지 않아야 한다. (Task 6)
3. **재생성 응답을 못 읽은 경우** — 연결 실패(`OSError`)나 파싱 실패. **원래 추천 문장을 유지**하고 로그·결과 dict에 "재생성 실패로 원본 유지"로 남긴다. 폐기는 "재생성된 문장을 다시 검사해서 또 걸렸을 때"만. (Task 4, 5)
4. **value에 정규식 특수문자가 들어간 경우** — "(주)" "A/B" "3개 이상?" 같은 값. 강조 분할이 `re.escape` 없이 깨지면 500이 난다. (Task 6)
5. **fills의 value·quote에 HTML이 섞인 경우** — 진행자 화면에 그대로 그려지므로 Django 자동 이스케이프 경로(`{{ }}`)만 쓰고 `|safe`·`mark_safe`를 쓰지 않는다. (Task 6)

---

### Task 1: 순수 검사 모듈 `decision_checks.py` (신규 파일만)

**Files:**
- Create: `APL_04/meetalign/decision_checks.py`
- Test: `APL_04/meetalign/tests.py` — 새 클래스 `DecisionCandidateCheckTests`
- `decision_compiler.py`는 **이 태스크에서 전혀 고치지 않는다.**

**Interfaces:**
- Consumes: 없음
- Produces (모두 이 파일 안에서 자체 구현):
  - `squash(text) -> str` (공백 제거), `bigrams(text) -> set`
  - `DATE_EXPR|TIME_EXPR|QUANTITY_EXPR|NUMBER_EXPR` 과 `date_tokens|time_tokens|quantity_tokens|number_tokens(text) -> set[str]`
  - `COPY_RATIO = 0.8`, `DUPLICATE_RATIO = 0.8`
  - `check_candidate(candidate, evidence, others=(), participant_names=()) -> (bool, list[str])`

**검사 셋과 사유 문구(화면에 안 나가고 로그·재생성 프롬프트에만 쓴다):**

| 검사 | 걸리는 조건 | 사유 |
|---|---|---|
| 베낌 | `len(C∩E)/len(C) >= COPY_RATIO` (C=candidate bigram, E=evidence bigram) | `"추천 문장이 원문 근거와 거의 같아요"` |
| 새 정보 없음 | candidate에는 있고 evidence에는 없는 날짜·시각·수량·숫자·참석자 이름 토큰이 0개 | `"원문에 없던 담당자·기한·수치가 하나도 없어요"` |
| 결정끼리 중복 | 다른 결정 문장 하나와 `jaccard >= DUPLICATE_RATIO` | `"다른 결정의 문장과 거의 같아요: <그 문장>"` |

- [ ] **Step 1: 실패하는 테스트를 쓴다** (`DecisionCandidateCheckTests`)

```python
def test_a_candidate_that_copies_the_evidence_is_flagged(self):
    evidence = "로그는 일단 한 달 정도면 충분할 것 같은데요"
    ok, reasons = decision_checks.check_candidate(evidence, evidence)
    self.assertFalse(ok)
    self.assertIn("추천 문장이 원문 근거와 거의 같아요", reasons)

def test_a_candidate_with_no_new_information_is_flagged(self):
    ok, reasons = decision_checks.check_candidate("로그 보관 기간을 정한다", "로그 보관 기간을 정해야 해요")
    self.assertFalse(ok)
    self.assertIn("원문에 없던 담당자·기한·수치가 하나도 없어요", reasons)

def test_a_filled_candidate_passes(self):
    ok, reasons = decision_checks.check_candidate(
        "민수가 12월 5일까지 로그 보관 기간을 30일로 바꾼다",
        "로그는 일단 한 달 정도면 충분할 것 같은데요",
        participant_names=["민수"])
    self.assertEqual((ok, reasons), (True, []))

def test_a_participant_name_counts_as_new_information(self):
    ok, _ = decision_checks.check_candidate(
        "민수가 발표 자료를 맡는다", "발표 자료는 누가 할까요", participant_names=["민수"])
    self.assertTrue(ok)

def test_a_number_with_a_particle_counts_as_new_information(self):
    """'한 달'을 '30일로'로 바꾼 것은 채운 값이다 — 기존 날짜 정규식은 조사가 붙은 기간을 일부러 안 잡는다."""
    ok, _ = decision_checks.check_candidate("민수가 30일로 바꾼다", "로그는 한 달 정도면 충분할 것 같은데요")
    self.assertTrue(ok)

def test_a_candidate_almost_identical_to_another_decision_is_flagged(self):
    other = "민수가 12월 5일까지 로그 보관 기간을 30일로 바꾼다"
    ok, reasons = decision_checks.check_candidate(
        other + ".", "로그 얘기였어요", others=[other], participant_names=["민수"])
    self.assertFalse(ok)
    self.assertIn("다른 결정의 문장과 거의 같아요: %s" % other, reasons)

def test_an_empty_candidate_has_nothing_to_verify(self):
    self.assertEqual(decision_checks.check_candidate("", "근거"), (True, []))

def test_the_module_stays_free_of_django_and_llm_imports(self):
    source = Path(decision_checks.__file__).read_text(encoding="utf-8")
    for forbidden in ("django", "services", "decision_compiler"):
        self.assertNotIn(forbidden, source)
```

- [ ] **Step 2: 실패를 확인한다** — `python manage.py test meetalign.tests.DecisionCandidateCheckTests`, 기대: `ModuleNotFoundError`
- [ ] **Step 3: `decision_checks.py`를 쓴다.** 정규식은 기존 것과 같은 내용을 이 파일에 새로 적는다(옮기지 않는다 — 머지 충돌을 피하려는 의도이고 중복은 나중에 정리한다). `NUMBER_EXPR = re.compile(r"\d+")`는 새 검사용이다.
- [ ] **Step 4: 테스트를 돌린다** — 새 클래스 + `DecisionWarningTests` + `DecisionEvidenceTests`, 기대: 전부 PASS
- [ ] **Step 5: 커밋** — `추천 문장 검사 순수 모듈`

---

### Task 2-A: `Interpretation.fills` 마이그레이션 (단독 커밋)

**Files:**
- Modify: `APL_04/meetalign/models.py` — `Interpretation.fills = models.JSONField(default=list, blank=True)`
- Create: `APL_04/meetalign/migrations/0029_interpretation_fills.py`

- [ ] **Step 1: 칸을 추가하고 마이그레이션을 만든다** — `python manage.py makemigrations meetalign --name interpretation_fills`
- [ ] **Step 2: AddField 하나뿐인지 확인한다** — 파일을 읽어 `AddField` 외 연산이 없는지 보고, `python manage.py migrate` 로 적용
- [ ] **Step 3: 커밋** — `Interpretation에 채운 값 칸 추가` (이 커밋에는 models.py와 마이그레이션만 담는다)

---

### Task 2-B: `fills` 파싱과 출처 검증

**Files:**
- Modify: `APL_04/meetalign/decision_checks.py` — `verify_fill_sources`
- Modify: `APL_04/meetalign/decision_compiler.py` — `parse_fills` 추가(새 함수), `parse_items`에서 **한 줄 호출**, `_save_items`에서 `fills=` **한 줄**
- Test: `tests.py` — 새 클래스 `DecisionFillSourceTests`

**Interfaces:**
- Consumes: Task 1의 `squash`
- Produces:
  - `decision_checks.SOURCE_QUOTE = "원문"`, `SOURCE_INFO = "회의정보"`, `SOURCE_GUESS = "AI추정"`
  - `decision_checks.verify_fill_sources(fills, source, info_text="") -> list[dict]` — `gap`/`value`/`source`/`quote` 네 칸만 남기고, 검증에 실패한 `source`는 `SOURCE_GUESS`로 내린다. dict가 아닌 원소와 `value`가 빈 원소는 버린다.
  - `decision_compiler.parse_fills(row, source=None, meeting_info=None) -> list[dict]` — 응답 한 줄에서 `fills`를 읽어 검증까지 끝낸다. `source`가 없으면(가짜 모드) 검증하지 않고 형식만 맞춘다.
  - `parse_items(raw, source_text=None, meeting_info=None)` — 기존 두 인자 호출은 그대로 동작한다.
  - `Interpretation.fills` 저장(후보가 남은 애매 항목만)

**검증 규칙:** `source="원문"`은 `squash(quote)`가 `squash(transcript)`에 있어야 한다. `source="회의정보"`는 `squash(value)`가 `squash(info_text)`에 있어야 한다. 어긋나면 `"AI추정"`. 아는 값이 아닌 `source`도 `"AI추정"`.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
SOURCE = "로그는 일단 한 달 정도면 충분할 것 같은데요. 민수가 발표를 맡을게요."

def test_a_quote_that_is_not_in_the_transcript_becomes_a_guess(self):
    fills = [{"gap": "기한", "value": "12월 5일", "source": "원문", "quote": "12월 5일까지 하죠"}]
    self.assertEqual(decision_checks.verify_fill_sources(fills, self.SOURCE)[0]["source"], "AI추정")

def test_a_quote_that_is_in_the_transcript_stays(self):
    fills = [{"gap": "담당", "value": "민수", "source": "원문", "quote": "민수가 발표를 맡을게요"}]
    self.assertEqual(decision_checks.verify_fill_sources(fills, self.SOURCE)[0]["source"], "원문")

def test_meeting_info_is_checked_against_the_real_meeting_info(self):
    fills = [{"gap": "담당", "value": "민수", "source": "회의정보"},
             {"gap": "기한", "value": "12월 9일", "source": "회의정보"}]
    verified = decision_checks.verify_fill_sources(fills, self.SOURCE, info_text="참석자: 민수, 지은")
    self.assertEqual([f["source"] for f in verified], ["회의정보", "AI추정"])

def test_broken_fills_are_dropped_without_touching_the_candidate(self):
    for broken in (None, "기한", [{"gap": "기한"}], ["문자열"], [{"value": ""}]):
        self.assertEqual(decision_checks.verify_fill_sources(broken, self.SOURCE), [])

def test_an_unknown_source_label_becomes_a_guess(self):
    fills = [{"gap": "범위", "value": "세 장", "source": "추측"}]
    self.assertEqual(decision_checks.verify_fill_sources(fills, self.SOURCE)[0]["source"], "AI추정")

def test_parse_items_keeps_verified_fills_for_ambiguous_items(self): ...
def test_a_clear_item_carries_no_fills(self): ...
def test_compile_stores_fills_on_the_saved_candidate(self): ...   # 다음 요청에서 색을 칠할 수 있어야 한다
```

- [ ] **Step 2: 실패를 확인한다** — `python manage.py test meetalign.tests.DecisionFillSourceTests`
- [ ] **Step 3: `verify_fill_sources`를 구현한다**(Task 1 모듈에 추가, 순수 유지)
- [ ] **Step 4: `decision_compiler.parse_fills`를 추가하고, `parse_items`의 item dict에 `"fills": parse_fills(row, source_text, meeting_info) if ambiguous else []` 한 줄, `_save_items`의 `Interpretation.objects.create(...)`에 `fills=` 한 줄을 넣는다**
- [ ] **Step 5: 테스트를 돌린다** — 새 클래스 + `DecisionParseTests` + `DecisionCompileTests`, 기대: PASS
- [ ] **Step 6: 커밋** — `채운 값의 출처 검증`

---

### Task 3: 프롬프트 — 제한 해제와 새 블록(상수·함수로 분리)

**Files:**
- Modify: `APL_04/meetalign/decision_compiler.py` — 새 상수·함수 추가, `_compile_prompt`에서 **삭제 2줄 + 끼워 넣는 2줄**
- Test: `tests.py` — 새 클래스 `DecisionFillPromptTests`

**Interfaces:**
- Consumes: 기존 `_participant_names(meeting)`, `carryover_decisions(meeting)`
- Produces:
  - `FILL_RULES` — candidate의 빈칸 채우기 규칙 문구(상수)
  - `FILLS_SCHEMA_EXAMPLE` — `fills` JSON 예시 한 줄(상수)
  - `collect_meeting_info(meeting) -> {"date": "2026년 11월 18일 (화)", "participants": [...], "deadline": "2026-11-20"|"", "carryover": ["안건 — 사유", ...]}`
  - `_meeting_info_block(info) -> str` — `"[회의 정보]"`로 시작하는 조각. 인자는 `collect_meeting_info()`의 dict다(재생성 프롬프트가 같은 블록을 쓰므로 모델이 아니라 dict를 받는다). `info`가 비면 빈 문자열.
  - `_info_text(info) -> str` — `verify_fill_sources`의 `info_text`로 넘길 한 덩어리
  - `_compile_prompt(source, meeting=None)` — 인자 하나로 부르는 기존 호출(테스트 포함)이 그대로 동작한다
  - `compile_specs(source, meeting=None)`

**프롬프트 변경:**
- 삭제 ①: `"원문에 없는 날짜·이름을 지어내지 말고, 원문에 나온 날짜·요일·시간·장소 안에서만 고른다. "`
- 삭제 ②: `"채울 근거가 원문에 전혀 없으면 \"~은 다음 회의 전까지 단톡방에서 정한다\"처럼 정하는 방법을 확정하는 문장을 쓴다.\n"`
- 그 자리에 `FILL_RULES` 한 줄, JSON 형식 안내 뒤에 `FILLS_SCHEMA_EXAMPLE` 한 줄, 힌트 블록 옆에 `_meeting_info_block(info)` 한 줄을 끼운다. **그 외 기존 문장은 건드리지 않는다.**

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
def test_the_prompt_no_longer_forbids_filling_in_missing_values(self):
    prompt = decision_compiler._compile_prompt("로그는 한 달 정도면 충분할 것 같은데요.")
    self.assertNotIn("원문에 없는 날짜·이름을 지어내지 말고", prompt)
    self.assertNotIn("단톡방에서 정한다", prompt)
    self.assertIn("가장 그럴듯한 값", prompt)

def test_the_prompt_asks_for_fills_with_a_source(self):
    prompt = decision_compiler._compile_prompt("x")
    for token in ('"fills"', '"gap"', '"source"', "원문|회의정보|AI추정"):
        self.assertIn(token, prompt)

def test_the_meeting_info_block_carries_date_participants_deadline_and_carryover(self):
    block = decision_compiler._meeting_info_block(decision_compiler.collect_meeting_info(self.meeting))
    self.assertIn("[회의 정보]", block)
    self.assertIn("민수", block)
    self.assertIn(str(self.meeting.deadline), block)
    self.assertIn("지난 회의 미결정", block)
    self.assertIn("화자 A", block)

def test_compile_sends_the_meeting_info_block_in_the_prompt(self): ...
def test_the_prompt_works_without_a_meeting(self):
    self.assertNotIn("[회의 정보]", decision_compiler._compile_prompt("x"))
```

- [ ] **Step 2: 실패를 확인한다** — `python manage.py test meetalign.tests.DecisionFillPromptTests`
- [ ] **Step 3: 상수와 `collect_meeting_info`·`_meeting_info_block`·`_info_text`를 구현한다.** 날짜는 `timezone.localtime(meeting.created_at)` + 요일 한 글자. 이월 안건은 `carryover_decisions(meeting)`의 `what`·`undecided_reason`만 쓴다(문장 전체를 넣으면 프롬프트가 회의록만큼 길어진다).
- [ ] **Step 4: `_compile_prompt`에 세 줄을 끼우고 두 문장을 지운다. `compile_specs`/`compile_meeting`이 `meeting`을 전달하게 한다**
- [ ] **Step 5: 테스트를 돌린다** — 새 클래스 + `DecisionAmbiguityHintTests` + `DecisionCompileTests`, 기대: PASS
- [ ] **Step 6: 커밋** — `프롬프트: 채우기 허용과 회의 정보 블록`

---

### Task 4: `decision_refine.py` — 검증 → 1회 재생성 → 폐기

**Files:**
- Create: `APL_04/meetalign/decision_refine.py`
- Modify: `APL_04/meetalign/decision_compiler.py` — `compile_specs`에 **저장 직전 한 줄 호출**(순환 import를 피해 함수 안에서 import)
- Test: `tests.py` — 새 클래스 `DecisionRefineTests`

**Interfaces:**
- Consumes: `decision_checks.check_candidate`·`verify_fill_sources`, `decision_compiler.context_around`·`_draft_hint_text`·`_meeting_info_block`·`_info_text`·`_generate_parsed`
- Produces:
  - `refine_candidates(items, source, meeting_info=None) -> items` (같은 리스트를 고쳐 돌려준다)
  - 항목마다 `item["check"] = {"reasons": [...], "regenerated": bool, "discarded": bool, "kept": bool}` — 검사에 걸린 애매 항목에만 붙는다. `kept=True`는 "재생성 실패로 원본 유지".
  - 재생성 응답 형식: `[{"n": 1, "candidate": "...", "fills": [...]}]` (topic이 겹칠 수 있어 번호로 맞춘다)

**동작:**
1. 애매 항목 중 후보가 있는 것만 검사한다. 다른 결정 문장 목록 = 같은 회의의 다른 항목의 후보 문장과 `final`.
2. 걸린 항목이 없으면 **LLM을 부르지 않고** 그대로 돌려준다.
3. 걸린 항목만 모아 한 번 호출한다. 항목별 번호·안건·걸린 이유·근거·`context_around(source, evidence)`·다른 결정 문장 목록 + `[회의 정보]`를 주고 "아래 문장들과 겹치지 않게, 걸린 이유를 고쳐서 다시 쓰라"고 한다.
4. 응답을 못 읽으면(연결 실패·파싱 실패) 원래 문장을 유지하고 `kept=True`.
5. 새 문장이 검사를 통과하면 후보 문장과 `fills`를 갈아끼우고 `regenerated=True`.
6. 또 걸리면 후보를 비우고(`item["candidates"] = []`) `discarded=True`. `item["draft_hint"]`가 비어 있으면 폐기한 문장으로 채운다(`_draft_hint_text`).

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
def test_a_passing_candidate_does_not_call_the_llm(self): ...
def test_a_flagged_candidate_is_regenerated_once_and_kept(self): ...
def test_the_refine_prompt_carries_the_reasons_context_and_other_sentences(self): ...
def test_a_candidate_that_fails_twice_is_discarded_into_the_custom_flow(self): ...
def test_the_llm_is_called_at_most_once_for_regeneration(self): ...
def test_a_failed_regeneration_call_keeps_the_original_candidate(self): ...   # kept is True
def test_refine_is_skipped_in_fake_mode(self): ...
```

- [ ] **Step 2: 실패를 확인한다** — `python manage.py test meetalign.tests.DecisionRefineTests`
- [ ] **Step 3: `decision_refine.py`를 쓴다**
- [ ] **Step 4: `compile_specs`에 한 줄을 넣는다** — 파싱 성공 직후, 돌려주기 전. 가짜 응답 경로에는 넣지 않는다(LLM이 없어 재생성을 할 수 없고, 가짜 모드는 화면 시연용이다).
- [ ] **Step 5: 테스트를 돌린다** — 새 클래스 + `DecisionCompileTests` + `DecisionParseTests`. `test_the_llm_is_called_exactly_once`가 빨개지면 멈추고 보고한다.
- [ ] **Step 6: 커밋** — `추천 문장 검증과 1회 재생성`

---

### Task 5: 검증 결과 기록

**Files:**
- Modify: `APL_04/meetalign/decision_refine.py` — `logging`
- Modify: `APL_04/meetalign/decision_compiler.py` — `compile_meeting` 결과 dict에 집계 칸
- Test: `tests.py` — `DecisionRefineTests`에 2개 추가

**Interfaces:**
- Consumes: Task 4의 `item["check"]`
- Produces: `compile_meeting` 결과에 `"checked"`, `"regenerated"`, `"discarded"`, `"regenerate_failed"` 추가(기존 세 칸은 그대로)

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
def test_compile_meeting_reports_the_verification_counts(self):
    self.assertEqual((summary["checked"], summary["regenerated"], summary["discarded"]), (3, 1, 1))

def test_the_verification_result_is_logged(self):
    with self.assertLogs("meetalign.decision_refine", level="INFO") as logs:
        ...
    self.assertIn("추천 문장 검증", "\n".join(logs.output))

def test_a_failed_regeneration_is_recorded_as_keeping_the_original(self):
    self.assertEqual(summary["regenerate_failed"], 1)
    self.assertIn("재생성 실패로 원본 유지", "\n".join(logs.output))
```

- [ ] **Step 2: 실패를 확인한다**
- [ ] **Step 3: `logger`를 두고 걸린 항목마다 한 줄씩(안건·걸린 이유·재생성 결과) 남기고, `compile_meeting`이 네 칸을 집계한다**
- [ ] **Step 4: 테스트를 돌린다** — 기대: PASS
- [ ] **Step 5: 커밋** — `검증 결과 기록`

---

### Task 6: 진행자 화면 — AI추정 값 강조와 경고 한 줄

**Files:**
- Modify: `APL_04/meetalign/decision_checks.py` — `fill_segments`
- Modify: `APL_04/meetalign/views.py:706-716` — `decision_review`에서 interpretation에 임시 속성 2개
- Modify: `APL_04/meetalign/templates/meetalign/_decision_choices.html:14-21`
- Modify: `APL_04/meetalign/templates/meetalign/base.html` — `.ai-fill`
- Test: `tests.py` — 새 클래스 `DecisionFillDisplayTests`

**Interfaces:**
- Consumes: `Interpretation.fills`
- Produces:
  - `decision_checks.fill_segments(text, values) -> [{"text": str, "estimated": bool}]` — `re.escape`로 감싸고 긴 값부터, 겹치지 않는 자리만
  - `interpretation.segments`, `interpretation.estimate_gaps` (저장되지 않는 임시 속성, 기존 `choice_value`·`chosen`과 같은 자리)
- 강조 대상은 `source == "AI추정"`이고 `value`가 문장에 실제로 있는 fills만.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

```python
def test_fill_segments_marks_only_the_estimated_values(self):
    self.assertEqual(decision_checks.fill_segments("민수가 12월 5일까지 바꾼다", ["12월 5일"]),
                     [{"text": "민수가 ", "estimated": False},
                      {"text": "12월 5일", "estimated": True},
                      {"text": "까지 바꾼다", "estimated": False}])

def test_fill_segments_handles_regex_characters_in_a_value(self): ...   # "A/B(주)"
def test_fill_segments_leaves_the_sentence_whole_when_the_value_is_absent(self): ...
def test_the_host_screen_highlights_guessed_values_and_warns_once(self):
    self.assertContains(r, 'class="ai-fill"')
    self.assertContains(r, "AI가 추정한 값이 있어요: 기한")
def test_quoted_and_meeting_info_values_are_not_marked(self): ...
def test_a_value_with_html_is_escaped(self): ...
def test_the_team_screen_never_shows_the_fill_markup(self): ...
```

- [ ] **Step 2: 실패를 확인한다** — `python manage.py test meetalign.tests.DecisionFillDisplayTests`
- [ ] **Step 3: `fill_segments`를 구현한다**(순수 유지)
- [ ] **Step 4: `decision_review`에서 `segments`·`estimate_gaps`를 붙인다**
- [ ] **Step 5: `_decision_choices.html`의 `{{ interpretation.text }}` 한 줄을 세그먼트 반복으로 바꾸고, `estimate_gaps`가 있으면 `class="meta"` 경고 한 줄을 넣는다.** `|safe` 금지. `part != "rest"` 블록만 고친다(진행자 전용 경로).
- [ ] **Step 6: `base.html`에 `.ai-fill`을 추가한다** — `background:var(--accent-soft); color:var(--accent); border-radius:4px; padding:0 2px;`
- [ ] **Step 7: 테스트를 돌린다** — 새 클래스 + `DecisionReviewTests` + `DecisionAckTests` + `DecisionCardTests`, 기대: PASS
- [ ] **Step 8: 커밋** — `AI가 추정한 값 강조`

---

### Task 7: 전체 회귀와 문서

**Files:**
- Modify: `진행사항.txt`, `README.md`

- [ ] **Step 1: 최신 main 확인** — `git fetch upstream && git rebase upstream/main` (바뀐 게 있으면)
- [ ] **Step 2: 전체 테스트** — `cd APL_04 && ../.venv/Scripts/python.exe manage.py test`, 기대: 기존 522개 + 신규 전부 OK
- [ ] **Step 3: `python manage.py makemigrations --check --dry-run`** — 0029 외에 생길 마이그레이션이 없어야 한다
- [ ] **Step 4: 가짜 모드 수동 확인** — 서버를 띄워 결정 추출 → 진행자 화면이 예전과 똑같이 동작하는지
- [ ] **Step 5: `진행사항.txt`·`README.md`에 바뀐 동작을 적는다**(CRLF 유지)
- [ ] **Step 6: 커밋** — `진행사항 정리`
- [ ] **Step 7: 보고** — 테스트 결과, 바뀐 프롬프트 전문, 화면 변경 설명, PR 설명 초안. push·PR 생성은 하지 않는다.
