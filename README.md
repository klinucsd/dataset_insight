# dataset_insight

Point it at a folder or an archive. It reads what is in there and writes a
review that supports one decision: take this dataset in, or send it back to
whoever produced it with a list of questions.

```
dataset_insight /path/to/a/delivery.zip
dataset_insight /path/to/a/folder
```

It is an agent with an ordinary filesystem and shell. It plans, writes
scripts, runs them, reads what they print, and writes up what it found —
opening spreadsheets, PDFs, Word documents, images and binary data files as
it goes. Nothing about it is specific to one field or one kind of data.

## What it produces

A working folder, named after the dataset, in your temp directory:

```
DATASET_REVIEW.md   the review: what the dataset is, how the files relate,
                    what is wrong with it, and the questions to send back
issues.csv          one row per issue, with the question to send
quantities.csv      one row per measured quantity, with its unit and source
<catalogs>.csv      whatever index the dataset needs and does not have --
                    an identifier crosswalk, a per-subject table
<scripts>.py        every script it wrote, kept so each number can be rechecked
```

The CSVs matter as much as the review. Prose is what the next step in an
ingestion pipeline cannot read.

## Setup

Python **3.11 or newer**.

```
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env     # then put your key in it
```

Or with any Python 3.11+ you already have: `python3.12 -m venv .venv`,
`source .venv/bin/activate`, `pip install -r requirements.txt`.

## Choosing a model

Any OpenAI-compatible endpoint. Set it in `.env`:

```
MODEL=openai:gpt-5.5
OPENAI_API_KEY=sk-...
```

The provider prefix fills in the endpoint and the key variable —
`openai`, `zai`, `together`, `groq`, `openrouter`, `ollama`, `nrp`. For
anything else, name all three:

```
MODEL=my-model
MODEL_URL=https://your-endpoint/v1
MODEL_API_KEY_ENV=MY_API_KEY
MY_API_KEY=...
```

`--model`, `--url` and `--api-key-env` override `.env` for one run.

The model needs tool calling and room to work. A text-only model cannot read
a scanned page or a drawing whose text is outlined — it will say so rather
than guess, which is the right answer, but those files stay undetermined.

## Options

```
-o, --out DIR       working folder (default: one in your temp directory)
--model, --url, --api-key-env    override .env for this run
--show-output       print what each tool returned, not just its size
--fresh             clear the working folder first
--keep              leave the working folder as it is
--max-steps N       how many tool calls the run may make (default 400)
```

A run starts in a clean folder. The agent reads its working folder and cannot
tell a file it wrote from one a previous run left behind, so a stale artifact
gets read as this run's own finding — which is why the default is to clear.

## What it runs on your machine

The inspection executes model-written shell commands and Python **on the
machine you run it on, as you, with your permissions and no sandbox.** That
is what lets it write a parser for a format nobody anticipated, and it is a
real risk: run it on a dataset you are willing to let code touch, on a
machine where that is acceptable.

The agent is told where the dataset is, to read nothing else, and to write
only into the working folder. Nothing enforces that.

## Notes

The dataset itself is never modified. An archive is extracted into the working
folder; a directory is read where it sits.

A dropped connection resumes from the last completed step rather than losing
the run. If the model delivers its review in its closing message instead of
writing the file, that message is saved as the review.
