#!/usr/bin/env python3
"""Inspect a dataset proposed for ingestion, and write a report about it.

The ARGUS agent, run from a shell instead of a notebook cell. An ordinary
deep agent with an ordinary filesystem: it plans, writes scripts, runs them,
reads what it finds and writes up the result. Nothing here restricts its tools
or splits the work into stages -- an earlier version of this app did both, and
the scaffolding was the thing that kept breaking.

The terminal output is `scripts/sage_magic.py`'s, ported: the same one-line tool summaries, the same narration printed before each step, so
a run reads the way the notebook does. The notebook-only parts are gone --
HTML and Markdown rendering, inline images and maps, the magics, the cross-cell
memory, the review middleware, and the stray-output reclaiming that exists
only because a notebook's working directory is not where its code runs.

    dataset_insight /path/to/delivery.zip
    dataset_insight /path/to/a/folder --show-output
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

# All-caps so it reads as "this is the one" in a folder of forty working
# files, and named for what it is: a review that ends in a decision, not a
# report. It also has to survive being copied somewhere else -- `report.md`
# in a downloads folder says nothing about which delivery it judged.
REPORT_NAME = "DATASET_REVIEW.md"

# Any OpenAI-compatible endpoint works, named the way sage's `%model` names
# one: a model, a URL, and the NAME of the variable holding the key. Writing
# `provider:model` fills the other two in from here, so a first run needs one
# line of .env and a key.
PROVIDERS = {
    "nrp":       ("https://ellm.nrp-nautilus.io/v1", "NRP_API_KEY"),
    "openai":    ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "zai":       ("https://api.z.ai/api/coding/paas/v4", "ZAI_API_KEY"),
    "together":  ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "groq":      ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama":    ("http://localhost:11434/v1", "OLLAMA_API_KEY"),
}
DEFAULT_MODEL = "nrp:deepseek-v4-flash"

TODO_MARKS = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}

START = time.monotonic()


def clock():
    s = time.monotonic() - START
    return f"{int(s // 60):>3d}:{int(s % 60):02d}"


def resolve_model(spec, url=None, key_var=None):
    """Work out which model to call, where, and whose key to use.

    Named after sage's `%model --name provider:model --url ... --api-key-env
    ...`, because the three things that vary are the same three. A provider
    prefix fills in the URL and the key variable for an endpoint this app
    already knows; an explicit `url`/`key_var` always wins, so an endpoint it
    has never heard of needs no code change.

    Returns (model, url, key_var), with url None when there is nothing to go
    on -- the caller says what is missing rather than guessing.
    """
    provider, _, rest = spec.partition(":")
    if rest and provider in PROVIDERS:
        model, known_url, known_key = rest, *PROVIDERS[provider]
    elif rest and "//" not in spec:
        # provider:model where the provider is not one we know -- the model is
        # still the part after the colon, and the rest has to be supplied.
        model, known_url, known_key = rest, None, None
    else:
        # A bare model name falls back to the default provider, so `--model
        # qwen3` keeps working. Anyone pointing somewhere else sets MODEL_URL
        # and MODEL_API_KEY_ENV once in .env, and those win over this.
        model = spec
        known_url, known_key = PROVIDERS[DEFAULT_MODEL.split(":")[0]]
    return model, (url or known_url), (key_var or known_key)


def load_dotenv():
    """Read .env from the working directory, then from beside this file.

    An exported variable wins: a key in the environment is the one the user
    just set, and a file that silently overrode it would be the harder bug.
    """
    for folder in (Path.cwd(), Path(__file__).resolve().parent):
        env = folder / ".env"
        if not env.is_file():
            continue
        for line in env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
        return env
    return None


# --------------------------------------------------------------- rendering --
def script_hint(command):
    """What a heredoc script is about, taken from its body.

    The agent runs most of its work as `<interpreter> - <<EOF ... EOF`. The
    interpreter is sixty identical characters and the body is the only part
    that differs, so a line showing the command and hiding the body makes
    every step look like every other step. This digs out a label instead: a
    comment the agent wrote for itself, else the file it opens, else its
    first real statement.
    """
    m = re.search(r"<<\s*'?(\w+)'?\s*\n(.*?)\n\s*\1", command, re.S)
    if not m:
        return None
    body = m.group(2)
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("#") and len(line) > 3:
            return line.lstrip("#").strip()
    files = re.findall(
        r"""['"]([^'"]*\.(?:xlsx|xls|csv|dat|txt|pdf|docx|tif|tiff|png|json|md))['"]""",
        body)
    if files:
        seen = list(dict.fromkeys(os.path.basename(f) for f in files))
        return seen[0] + (f"  (+{len(seen) - 1} more)" if len(seen) > 1 else "")
    for line in body.splitlines():
        line = line.strip()
        if line and not line.startswith(("import ", "from ", "#")):
            return line
    return None


def summarise(tool, args, root=None, dataset=None):
    """One line saying what this step does."""
    def shorten(text, limit=96):
        text = " ".join(str(text).split())
        for prefix in (root, dataset):
            if prefix:
                # Every command starts by changing into the working folder,
                # and every path names one of the two roots. That prefix is
                # the same sixty characters on every line and says nothing.
                text = re.sub(r"^cd\s+\S*" + re.escape(str(prefix)) + r"\S*\s*&&\s*",
                              "", text)
                text = text.replace(str(prefix) + "/", "")
                text = text.replace(str(prefix), ".")
        text = re.sub(r"\S*/bin/python3?(\.\d+)?", "python", text)
        # A heredoc puts a whole program on the line; that a program is being
        # written and run is the point, not its body.
        text = re.sub(r"<<\s*'?(\w+)'?.*", r"<<\1 ...", text)
        if len(text) <= limit:
            return text
        # Cut from the LEFT: the end of a path is the filename, which is the
        # part that identifies the step. Trimming the right loses exactly it.
        return "..." + text[-(limit - 3):]

    if tool == "execute":
        command = str(args.get("command", ""))
        hint = script_hint(command)
        return ("Running python: " + shorten(hint, 78)) if hint \
            else ("Executing: " + shorten(command))
    if tool in ("read_file", "write_file", "edit_file", "delete"):
        verb = {"read_file": "Reading", "write_file": "Writing",
                "edit_file": "Editing", "delete": "Deleting"}[tool]
        return f"{verb}: " + shorten(args.get("file_path", "?"))
    if tool in ("ls", "glob"):
        return "Listing: " + shorten(args.get("path") or args.get("pattern", "."))
    if tool == "grep":
        return "Searching: " + shorten(args.get("pattern", "?"))
    if tool == "write_todos":
        return f"{len(args.get('todos') or [])} todo(s)"
    if tool == "task":
        return "Subagent: " + shorten(args.get("description", "?"))
    return shorten(json.dumps(args, ensure_ascii=False), 100)


class Printer:
    """Everything the run says, and the only place that writes to stdout.

    Spacing is owned here rather than echoed. The model ends a paragraph with
    whatever trailing newlines it feels like, and a step that added its own on
    top left two and three blank lines between items. So every write goes
    through `_w`, which collapses runs of newlines and tracks how many are
    pending, and each kind of line then asks for exactly the separation it
    wants: a blank line after narration, none between a step and its result.

    Narration streams as it arrives rather than being buffered into blocks:
    the notebook can wait for a complete markdown cell, a terminal reads
    better when it fills as the model writes, and a long silence is the thing
    that makes a run look hung.
    """

    def __init__(self, root, dataset=None, show_output=False):
        self.root = root
        self.dataset = dataset
        self.show_output = show_output
        self._pending = 2          # trailing newlines written; 2 == blank line
        self._last = None          # narration | tool | result
        self._echo = None          # the step line last printed
        self.repeats = 0           # how many times it has repeated since

    def _w(self, text):
        """Write, with at most one blank line anywhere and none at the top."""
        if not text:
            return
        text = re.sub(r"\n{3,}", "\n\n", text)
        lead = len(text) - len(text.lstrip("\n"))
        if lead:                   # we may already have some of those newlines
            text = text[min(lead, self._pending):]
            if not text:
                return
        sys.stdout.write(text)
        tail = len(text) - len(text.rstrip("\n"))
        self._pending = tail
        sys.stdout.flush()

    def _space(self, blank):
        """Reach the separation this block wants -- written directly, because
        `_w` strips leading newlines against what is already pending and would
        swallow the very separator being asked for."""
        want = 2 if blank else 1
        if self._pending < want:
            sys.stdout.write("\n" * (want - self._pending))
            self._pending = want

    def close_streak(self):
        """Report a repeated step once it stops repeating.

        An agent that asks the same question sixteen times running is worth
        seeing, but not worth sixteen identical lines. It escaped on its own
        here, so this says what happened rather than intervening.
        """
        if self.repeats:
            self._space(blank=False)
            self._w(f"             (the same step, {self.repeats} more "
                    f"time(s))\n")
            self.repeats = 0
            self._last = "result"

    def narration(self, text):
        """A piece of the model's own words. Called once per streamed token,
        so the separation is asked for only when a block STARTS -- demanding
        it per chunk puts a blank line inside every sentence."""
        if not text:
            return
        if self._last != "narration":
            self.close_streak()
            self._space(blank=True)
        self._w(text)
        self._last = "narration"

    def tool_call(self, tool, args):
        line = (f"{tool:<12} "
                f"{summarise(tool, args, self.root, self.dataset)}")
        if line == self._echo:
            # Same step again. Count it and print nothing; `close_streak`
            # says how many once the agent moves on.
            self.repeats += 1
            self._last = "tool"
            return
        self.close_streak()
        self._echo = line
        # A blank line separates a step from the prose that explains it;
        # consecutive steps sit together, which is how they read.
        self._space(blank=self._last == "narration")
        self._w(f"{clock()}  {line}\n")
        if tool == "write_todos":
            # The plan is printed in full: it is the one step a reader
            # follows rather than skims.
            for todo in args.get("todos") or []:
                mark = TODO_MARKS.get(str(todo.get("status", "")), "[ ]")
                self._w(f"             {mark} {todo.get('content', '')}\n")
        self._last = "tool"

    def tool_result(self, tool, content):
        if self.repeats:
            return             # the step is repeating; its answer is too
        self._space(blank=False)
        lines = str(content).splitlines()
        if not self.show_output:
            self._w(f"             -> {tool}: {len(lines)} line(s)\n")
        else:
            self._w(f"             -> {tool}: {len(lines)} line(s):\n")
            for line in lines[:100]:
                self._w(f"             |  {line[:200]}\n")
            if len(lines) > 100:
                self._w(f"             |  ... ({len(lines) - 100} more)\n")
        self._last = "result"

    def note(self, text):
        """A block the run itself is saying, set off by a blank line."""
        self.close_streak()        # a run ending mid-streak still reports it
        self._space(blank=True)
        self._w(text.strip() + "\n")
        self._last = "result"

    def line(self, text=""):
        """One more line of the block already being printed."""
        self._space(blank=False)
        self._w(text.rstrip() + "\n")
        self._last = "result"


# ------------------------------------------------------------------ prompts --
def system_prompt(interpreter, workdir, dataset):
    """The operational rules, and nothing about how to do the inspection.

    These are the parts of sage's system prompt that are not about notebooks:
    which interpreter to use, where files go, plan first, keep what you wrote.
    What to look for in a delivery is left to the agent -- the cases that
    matter in a dataset nobody has seen cannot be enumerated in advance, and a
    prompt that tries ends up steering it away from whatever it would have
    noticed on its own.
    """
    return (
        f"You inspect datasets and write reports about them.\n\n"
        f"The Python interpreter is at: {interpreter} — always use this exact "
        f"path when running Python. It has openpyxl and xlrd (spreadsheets), "
        f"pymupdf (PDF -- import pymupdf, not fitz), python-docx, Pillow, "
        f"pandas and numpy already installed. Do not use "
        f"'python' or 'python3', and never try to install anything.\n\n"
        f"THE DATASET is at {dataset}. Read it; never modify, move or delete "
        f"anything in it.\n\n"
        f"YOUR WORKING DIRECTORY is {workdir}. Write every file you produce "
        f"there — scripts, intermediate tables, figures and your report. Your "
        f"shell starts there. Write nowhere else.\n\n"
        f"FILE ACCESS RULE — read and search in those two places only: the "
        f"dataset, and your working directory. Never read, list or browse "
        f"anywhere else on this machine. What sits beside the dataset is not "
        f"part of it, and a neighbouring folder reported as part of the "
        f"delivery is a wrong finding.\n\n"
        f"PLANNING RULE — begin by calling write_todos with the steps you "
        f"intend to take, before running any other tool. Mark each step "
        f"in_progress when you start it and completed when it finishes. If "
        f"what you find changes what needs doing, call write_todos again to "
        f"update the plan rather than departing from it silently: the plan is "
        f"a record the reader inspects, so it must track what you actually "
        f"did.\n\n"
        f"IMAGE FILES — this model reads text, not pictures. Inspect an image "
        f"with code: Pillow gives you its size, mode, bit depth and pixel "
        f"statistics, and numpy will compare two images and tell you how many "
        f"pixels differ and by how much. Report what those numbers show. "
        f"Opening an image with read_file puts its bytes in the conversation "
        f"where they cannot be interpreted, and a few large ones make the "
        f"request itself fail — a run died at minute seventeen with 2.9 MB of "
        f"rendered images in its context. Where a file's content can only be "
        f"seen and not read, say so and ask the provider for it: a file you "
        f"could not read is undetermined, not empty.\n\n"
        f"FILE DELETION RULE — never delete a file you have written, even one "
        f"you judge no longer needed. The reader needs every script intact to "
        f"check how a number was obtained.\n\n"
        f"SAY WHAT YOU ARE DOING — a sentence before each step, saying what "
        f"you are about to do and why. Someone is watching this run and has "
        f"only your words to follow it by."
    )


TASK = (
    "Inspect the dataset at {dataset} and write a comprehensive report about "
    "it.\n\n"
    "Work out what it contains, what each file is for, how the files relate "
    "to one another, and what is wrong with it or missing from it. Treat it "
    "as a standalone delivery: it is not connected to any other system, and "
    "no skill applies to it.\n\n"
    "Write the review to {workdir}/{report_name} as you go, and put anything "
    "with a shape — the issues you find, the quantities it measures, any "
    "index the delivery needs and does not have — into CSV files beside it. "
    "Prose is what the next step in an ingestion pipeline cannot read.\n\n"
    "The report is for someone who has never seen this delivery and must "
    "decide whether to accept it, or send it back to the provider with "
    "questions. Make the questions ones that can be sent as written."
)


# ---------------------------------------------------------------- the agent --
def run(agent, task, printer, config, attempts=4):
    """Stream the agent to completion, resuming if the endpoint drops it.

    The model is configured to retry, and the SDK does retry a 500 -- but a
    request that is too large fails identically every time, so three retries
    buy nothing. A run died at minute seventeen having just read 2.9 MB of
    rendered images into the conversation, which every later request resends.

    So the recovery is at this level instead: the graph is checkpointed, and
    an attempt that dies resumes from the last completed step rather than
    from the beginning. The printing state is kept across attempts, so a step
    that was already shown is not shown twice when the graph replays it.
    """
    state = {"shown": set(), "buffers": {}, "cur_text_id": None,
             "had_tool": True, "skip_id": None, "final": []}
    payload = {"messages": [{"role": "user", "content": task}]}
    for attempt in range(1, attempts + 1):
        try:
            _stream(agent, payload, printer, config, state)
            break
        except KeyboardInterrupt:
            raise
        except Exception as exc:                             # noqa: BLE001
            if attempt == attempts:
                printer.note(f"\n  {type(exc).__name__}: {exc}")
                printer.note(f"  gave up after {attempts} attempts")
                break
            wait = 5 * attempt
            printer.note(f"\n  {type(exc).__name__}: {str(exc)[:120]}")
            printer.note(f"  resuming from the last completed step in {wait}s "
                         f"(attempt {attempt + 1} of {attempts})")
            time.sleep(wait)
            payload = None          # None resumes from the checkpoint
    return "".join(state["final"]).strip()


def _stream(agent, payload, printer, config, state):
    """One pass of the stream. Ported from sage's loop.

    The dedup is the part worth keeping: with token streaming the same
    message arrives more than once, and text that follows text with no tool
    call between them is a repeat rather than a continuation.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    shown, buffers, final = state["shown"], state["buffers"], state["final"]
    cur_text_id = [state["cur_text_id"]]
    had_tool_after_text = [state["had_tool"]]
    skip_id = [state["skip_id"]]

    for chunk in agent.stream(payload, stream_mode=["messages"], config=config):
        # With several stream modes langgraph yields (mode, payload); with one
        # it yields the payload alone. Detected, not assumed.
        if isinstance(chunk, tuple) and len(chunk) >= 2 and \
                isinstance(chunk[0], str) and chunk[0] == "messages":
            payload = chunk[1]
        else:
            payload = chunk
        if not isinstance(payload, tuple) or len(payload) < 2:
            continue
        message, metadata = payload[0], payload[1]

        if isinstance(message, ToolMessage):
            printer.tool_result(getattr(message, "name", "tool"),
                                getattr(message, "content", "") or "")
            had_tool_after_text[0] = True
            continue
        if not isinstance(message, AIMessage):
            continue
        if metadata and metadata.get("lc_source") == "summarization":
            continue

        raw = getattr(message, "content", "")
        if isinstance(raw, str):
            content = raw
        elif isinstance(raw, list):
            content = "".join(p.get("text", "") for p in raw
                              if isinstance(p, dict) and p.get("type") == "text"
                              and isinstance(p.get("text"), str))
        else:
            content = ""
        if content:
            mid = getattr(message, "id", None)
            if mid is not None:
                if mid == skip_id[0]:
                    content = ""
                elif mid != cur_text_id[0]:
                    if not had_tool_after_text[0]:
                        skip_id[0] = mid          # text after text: a repeat
                        content = ""
                    else:
                        cur_text_id[0] = mid
                        had_tool_after_text[0] = False
            if content:
                final.append(content)
                printer.narration(content)

        calls = list(getattr(message, "tool_call_chunks", None) or [])
        if not calls:
            calls = [{"id": tc.get("id"), "name": tc.get("name"),
                      "args": tc.get("args", {}), "index": i}
                     for i, tc in enumerate(getattr(message, "tool_calls", None) or [])]

        for call in calls:
            key = call.get("index") if call.get("index") is not None else call.get("id")
            if key is None:
                key = f"unknown-{len(buffers)}"
            buf = buffers.setdefault(key, {"name": None, "id": None,
                                           "args": None, "parts": []})
            if call.get("name"):
                buf["name"] = call["name"]
            if call.get("id"):
                buf["id"] = call["id"]
            args = call.get("args")
            if isinstance(args, dict):
                buf["args"], buf["parts"] = args, []
            elif isinstance(args, str) and args:
                buf["parts"].append(args)
                buf["args"] = "".join(buf["parts"])
            if not buf["name"]:
                continue

            parsed = buf["args"]
            if isinstance(parsed, str):
                if not parsed:
                    continue
                try:
                    parsed = json.loads(parsed)
                except json.JSONDecodeError:
                    continue          # arguments still arriving
            elif parsed is None:
                continue
            if not isinstance(parsed, dict):
                parsed = {"value": parsed}

            seen_key = buf["id"] or key
            buffers.pop(key, None)
            if seen_key in shown:
                continue
            shown.add(seen_key)
            printer.tool_call(buf["name"], parsed)
            had_tool_after_text[0] = True

        state["cur_text_id"] = cur_text_id[0]
        state["had_tool"] = had_tool_after_text[0]
        state["skip_id"] = skip_id[0]


def main():
    # BEFORE the parser is built. argparse evaluates `default=` at build
    # time, so a default read from the environment is fixed before .env has
    # been loaded -- MODEL=nrp:qwen3 in .env was silently ignored and the run
    # used the built-in default instead.
    env_file = load_dotenv()

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="the delivery: a folder or an archive")
    ap.add_argument("-o", "--out", help="working folder "
                                        "(default: a folder in your temp dir)")
    ap.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL),
                    help="provider:model, or just a model name "
                         "(default: %(default)s). Also MODEL in .env")
    ap.add_argument("--url", default=os.environ.get("MODEL_URL"),
                    help="the OpenAI-compatible endpoint, if the provider is "
                         "not one this app knows. Also MODEL_URL in .env")
    ap.add_argument("--api-key-env", default=os.environ.get("MODEL_API_KEY_ENV"),
                    help="NAME of the variable holding the API key -- not the "
                         "key. Also MODEL_API_KEY_ENV in .env")
    ap.add_argument("--show-output", action="store_true",
                    help="print what each tool returned, not just its size")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--fresh", action="store_true",
                    help="clear the working folder first, even if you named "
                         "it with -o")
    ap.add_argument("--keep", action="store_true",
                    help="leave the working folder as it is; the agent will "
                         "read whatever a previous run left there")
    a = ap.parse_args()

    dataset = Path(a.dataset).expanduser().resolve()
    if not dataset.exists():
        print(f"\n  no such dataset: {dataset}\n")
        return 2

    import tempfile
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", dataset.stem) or "dataset"
    work = Path(a.out).expanduser().resolve() if a.out else \
        Path(tempfile.gettempdir()) / f"dataset-insight-{name}"

    # Settle the model BEFORE touching the working folder: a typo in a model
    # name should not destroy the previous run's output on its way to an
    # error message.
    model_name, url, key_var = resolve_model(a.model, a.url, a.api_key_env)
    if url is None:
        print(f"\n  don't know where to reach {a.model!r}.\n"
              f"    Give the endpoint and the name of its key variable:\n"
              f"        MODEL={a.model}\n"
              f"        MODEL_URL=https://your-endpoint/v1\n"
              f"        MODEL_API_KEY_ENV=YOUR_API_KEY\n"
              f"    or write it as provider:model, where provider is one of: "
              f"{', '.join(sorted(PROVIDERS))}\n")
        return 2
    key = os.environ.get(key_var)
    if not key:
        print(f"\n  no {key_var} found\n"
              f"    Put it in a .env file next to this app, or export it:\n"
              f"        echo '{key_var}=your-key' >> .env\n"
              + (f"    (.env was read from {env_file}, but has no "
                 f"{key_var})\n" if env_file else ""))
        return 2

    # A run must not start in another run's leftovers. The agent reads its
    # working folder, and it cannot tell a file it wrote from one left behind:
    # a run read a previous run's decoder script and announced its conclusion
    # as "an important discovery", then listed that run's CSV among its own
    # deliverables. It had also read an `analysis.json` written a day earlier
    # by an implementation that no longer existed. Every one of those was
    # true, and none of them was established by the run reporting it.
    existing = sorted(work.iterdir()) if work.is_dir() else []
    if existing and not a.keep:
        auto_named = a.out is None
        if not (auto_named or a.fresh):
            print(f"\n  {work} already holds {len(existing)} item(s).\n"
                  f"    The agent reads this folder and cannot tell its own "
                  f"files from a previous run's.\n"
                  f"    Pass --fresh to clear it, --keep to inspect it as it "
                  f"is, or -o for another folder.\n")
            return 2
        shutil.rmtree(work, ignore_errors=True)
        print(f"\n  cleared {len(existing)} item(s) from a previous run")
    work.mkdir(parents=True, exist_ok=True)

    print()
    print("  dataset_insight")
    print(f"  dataset   {dataset}")
    print(f"  working   {work}")
    print(f"  model     {model_name}  at {url}")
    print()

    from deepagents import create_deep_agent
    from deepagents.backends import CompositeBackend, LocalShellBackend
    from langchain.agents.middleware import TodoListMiddleware
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=model_name, api_key=key, base_url=url,
        temperature=0, timeout=600, max_retries=3, max_tokens=32000)

    # The shell does not inherit this process's environment by default, which
    # leaves `python3` as the system interpreter without any of the libraries
    # the inspection needs. A run spent its first ten steps on pip, apt-get
    # and sudo trying to install what was already in our own venv.
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    shell = LocalShellBackend(root_dir=str(work), virtual_mode=False,
                              timeout=600, inherit_env=True, env=env)
    # When the conversation grows, deepagents summarises it and offloads the
    # older messages to `{artifacts_root}/conversation_history/`. That root is
    # read off a CompositeBackend and is otherwise "/" -- the real filesystem
    # root here, which nothing may write to. A run offloaded 200 messages,
    # failed on every one, and reported "older messages will not be
    # recoverable": it was discarding its own history as it went. Routing the
    # artifacts root at the working folder puts them somewhere writable, and
    # leaves them on disk afterwards for anyone checking what the agent saw.
    backend = CompositeBackend(default=shell, routes={},
                               artifacts_root=str(work))

    from langgraph.checkpoint.memory import InMemorySaver

    agent = create_deep_agent(
        model=model, backend=backend, middleware=[TodoListMiddleware()],
        system_prompt=system_prompt(sys.executable, work, dataset),
        checkpointer=InMemorySaver())

    printer = Printer(work, dataset=dataset, show_output=a.show_output)
    report = work / REPORT_NAME
    try:
        final = run(agent, TASK.format(dataset=dataset, workdir=work,
                                       report_name=REPORT_NAME), printer,
                    {"recursion_limit": a.max_steps * 2,
                     "configurable": {"thread_id": "inspection"}})
    except KeyboardInterrupt:
        printer.note(f"\n  interrupted; what was written is in {work}\n")
        return 130
    except Exception as exc:                                 # noqa: BLE001
        printer.note(f"\n  stopped: {type(exc).__name__}: {exc}")
        final = ""

    # The agent may deliver its report through write_file, in its closing
    # message, or both. A run that investigated for an hour does not end with
    # nothing on disk because of which channel its answer arrived on.
    if not (report.exists() and report.stat().st_size) and len(final) > 1000:
        report.write_text(final + "\n")
        printer.note(f"\n  report taken from the closing message "
                     f"({len(final):,} characters)")

    made = sorted(p for p in work.iterdir() if p.is_file())
    if made:
        # The review first and marked, whatever it is called: a listing of
        # forty working files buries the one thing the run was for.
        made.sort(key=lambda f: (f != report, f.suffix != ".csv", f.name))
        printer.note(f"  {len(made)} file(s) written:")
        for path in made:
            mark = "<--" if path == report else "   "
            printer.line(f"  {path.stat().st_size:>10,}  {path.name} {mark}")
    printer.note(f"  everything is in {work}")
    printer.line()
    return 0 if (report.exists() and report.stat().st_size) else 1


if __name__ == "__main__":
    sys.exit(main())
