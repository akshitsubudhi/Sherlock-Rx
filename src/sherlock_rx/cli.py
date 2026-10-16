"""`sherlock` command line: generate data, train NLU, chat, evaluate, benchmark, serve."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from sherlock_rx.dialogue.manager import Sherlock

app = typer.Typer(add_completion=False, help="Sherlock Rx: root-cause analysis chatbot for pharma brands.")
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("SHERLOCK_DATA", ROOT / "data" / "warehouse"))
MODELS = Path(os.environ.get("SHERLOCK_MODELS", ROOT / "models"))


def build_bot(data: Path = DATA, models: Path = MODELS, llm: bool = False,
              threads: int | None = None) -> Sherlock:
    from sherlock_rx.dialogue.manager import Sherlock
    from sherlock_rx.nlp.pipeline import NLU
    from sherlock_rx.warehouse.engine import Warehouse

    wh = Warehouse(data, threads=threads)
    geo = wh.query("SELECT territory, state FROM dim_geo")
    nlu = NLU.load(models, territory_state=dict(zip(geo.territory, geo.state, strict=True)))
    fallback = None
    if llm:
        from sherlock_rx.nlp.llm_fallback import ClaudeFallback

        fallback = ClaudeFallback(nlu.extractor.gaz)
    return Sherlock(wh, nlu, fallback=fallback)


@app.command()
def generate(scale: float = typer.Option(1.0, help="1.0 ≈ 113 territories / 76k sales rows"),
             seed: int = 7, out: Path = DATA) -> None:
    """Generate the synthetic warehouse with planted root causes."""
    from sherlock_rx.data.generator import generate as gen

    s = gen(out, scale=scale, seed=seed)
    typer.echo(f"{s.territories} territories, {s.rows['fact_sales']:,} sales rows, "
               f"{s.scenarios} planted scenarios, {s.controls} controls in {s.seconds}s -> {out}")


@app.command()
def train(out: Path = MODELS, seed: int = 13) -> None:
    """Train the NLU stack (tagger + intent classifier) and write reports/nlu_report.md."""
    from sherlock_rx.nlp.train import train as run

    run(out, seed=seed)


@app.command()
def chat(llm: bool = typer.Option(False, help="Use Claude as NLU fallback (needs ANTHROPIC_API_KEY)"),
         debug: bool = typer.Option(False, help="Print the NLU trace for every turn")) -> None:
    """Interactive chat in the terminal."""
    bot = build_bot(llm=llm)
    state = bot.new_state()
    typer.echo("Sherlock Rx: ask me why a brand is under-performing. Type 'quit' to exit.\n")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        r = bot.handle(text, state)
        if debug:
            n = r.nlu
            typer.secho(f"  [intent={r.intent} conf={r.confidence:.2f} policy={r.policy} "
                        f"masked='{n['masked']}' slots={n['slots']} {r.timings_ms}]", fg="cyan")
        typer.echo(f"\nsherlock> {r.text}\n")
        if r.intent == "goodbye":
            break


@app.command()
def ask(question: list[str]) -> None:
    """One-shot question, e.g. `sherlock ask why is Cardivex down in North`."""
    bot = build_bot()
    r = bot.handle(" ".join(question), bot.new_state())
    typer.echo(r.text)


@app.command(name="eval")
def evaluate(what: str = typer.Argument("all", help="rca | nlu | dialogue | all")) -> None:
    """Run evaluations and write reports/."""
    from sherlock_rx.eval.report import run_all

    run_all(what, DATA, MODELS, ROOT / "reports")


@app.command()
def bench(scales: str = "1,10,50", out: Path = ROOT / "reports") -> None:
    """Scale benchmark: generation, load, RCA and chat latency at several scale factors."""
    from sherlock_rx.eval.bench import run as run_bench

    res = run_bench([float(s) for s in scales.split(",")], MODELS, out)
    typer.echo(json.dumps(res, indent=2))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, workers: int = 1) -> None:
    """Run the FastAPI service."""
    import uvicorn

    uvicorn.run("sherlock_rx.api.main:app", host=host, port=port, workers=workers)


if __name__ == "__main__":
    app()
