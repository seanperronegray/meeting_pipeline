"""Command-line interface for the meeting research pipeline.

Every stage is independently rerunnable:

    pipeline import meetingbank --path meetings.jsonl
    pipeline import localview   --path localview.json
    pipeline standardize
    pipeline dedup [--use-llm] [--date-tolerance 1]
    pipeline stats

This mirrors a data-lake workflow: raw is immutable, everything downstream is
derived and can be rebuilt without re-importing.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .analysis import MeetingAnalyzer
from .database import REVIEW_STATES, Database
from .dedup import DedupConfig, Deduplicator
from .localview import LocalViewImporter, LocalViewStandardizer
from .meetingbank import MeetingBankImporter, MeetingBankStandardizer

# Registries — extend these to add a new source.
IMPORTERS = {
    "meetingbank": MeetingBankImporter,
    "localview": LocalViewImporter,
}
STANDARDIZERS = {
    "meetingbank": MeetingBankStandardizer,
    "localview": LocalViewStandardizer,
}


def _cmd_import(args: argparse.Namespace) -> int:
    importer_cls = IMPORTERS[args.source]
    kwarg = "jsonl_path" if args.source == "meetingbank" else "json_path"
    importer = importer_cls(**{kwarg: args.path})
    with Database(args.db) as db:
        result = importer.run(db)
        print(result)
        print(f"raw {args.source} rows: {db.count_raw(args.source)}")
    return 0


def _cmd_standardize(args: argparse.Namespace) -> int:
    sources = [args.source] if args.source else list(STANDARDIZERS)
    with Database(args.db) as db:
        total = 0
        for source in sources:
            n = STANDARDIZERS[source]().run(db)
            print(f"standardized {source}: {n}")
            total += n
        print(f"meeting table now holds {db.count_meetings()} rows")
    return 0


def _cmd_dedup(args: argparse.Namespace) -> int:
    config = DedupConfig(
        date_tolerance_days=args.date_tolerance,
        use_llm=args.use_llm,
    )
    with Database(args.db) as db:
        if args.rebuild:
            db.clear_dedup()
        deduper = Deduplicator(config=config)
        summaries = deduper.run(db)
        matched = [s for s in summaries if s["decision"] == "matched"]
        review = [s for s in summaries if s["decision"] == "needs_review"]
        unique = [s for s in summaries if s["decision"] == "unique"]
        print(f"canonical entities: {len(summaries)}")
        print(f"  matched:      {len(matched)}")
        print(f"  needs_review: {len(review)}")
        print(f"  unique:       {len(unique)}")
        for s in review:
            print(f"  [review] canonical={s['canonical_id']} "
                  f"members={s['members']} conf={s.get('confidence')}")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    with Database(args.db) as db:
        for source in IMPORTERS:
            print(f"raw {source}: {db.count_raw(source)}")
        print(f"meeting: {db.count_meetings()}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    with Database(args.db) as db:
        summary = MeetingAnalyzer().run(
            db,
            prompt_name=args.name,
            prompt_text=args.prompt,
            description=args.description,
        )
        print(
            f"run {summary.run_id} for prompt {summary.prompt_name!r} "
            f"(model={summary.model}, exemplars={summary.exemplar_count})"
        )
        print(
            f"  scanned {summary.meetings_scanned} meetings, "
            f"{summary.matches} matches"
        )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    with Database(args.db) as db:
        findings = MeetingAnalyzer().report(
            db, prompt_name=args.name, run_id=args.run,
        )
        if not findings:
            print("no findings")
            return 0
        print(
            f"{len(findings)} finding(s) for prompt "
            f"{args.name or 'run=' + str(args.run)}"
        )
        for f in findings:
            loc = f.get("municipality") or "?"
            state = f.get("state") or ""
            fips = f.get("fips_code") or "-"
            date = f.get("meeting_date") or "?"
            conf = f.get("confidence") or 0.0
            status = f.get("review_status") or "unreviewed"
            print(
                f"  [{f['finding_id']}] {loc}, {state} ({fips}) "
                f"{date}  conf={conf:.3f}  review={status}"
            )
            if f.get("summary"):
                print(f"      summary: {f['summary']}")
            for q in f.get("quotes") or []:
                print(f"      quote:   {q}")
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    with Database(args.db) as db:
        MeetingAnalyzer().review(
            db, args.finding_id, args.status, reviewer=args.reviewer,
        )
        print(f"finding {args.finding_id} -> {args.status}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline", description=__doc__)
    parser.add_argument("--db", default="pipeline.db", help="SQLite DB path")
    sub = parser.add_subparsers(dest="command", required=True)

    p_import = sub.add_parser("import", help="import a source into the raw layer")
    p_import.add_argument("source", choices=list(IMPORTERS))
    p_import.add_argument("--path", required=True, help="input file path")
    p_import.set_defaults(func=_cmd_import)

    p_std = sub.add_parser("standardize", help="rebuild the standardized layer")
    p_std.add_argument("source", nargs="?", choices=list(STANDARDIZERS))
    p_std.set_defaults(func=_cmd_standardize)

    p_dedup = sub.add_parser("dedup", help="run entity resolution")
    p_dedup.add_argument("--date-tolerance", type=int, default=1)
    p_dedup.add_argument("--use-llm", action="store_true")
    p_dedup.add_argument("--rebuild", action="store_true",
                         help="clear prior dedup results first")
    p_dedup.set_defaults(func=_cmd_dedup)

    p_stats = sub.add_parser("stats", help="row counts per layer")
    p_stats.set_defaults(func=_cmd_stats)

    p_analyze = sub.add_parser(
        "analyze",
        help="run a named prompt against every standardized meeting",
    )
    p_analyze.add_argument("--name", required=True,
                           help="stable prompt identifier (created on first use)")
    p_analyze.add_argument("--prompt", help="prompt text (required on first use)")
    p_analyze.add_argument("--description", help="optional prompt description")
    p_analyze.set_defaults(func=_cmd_analyze)

    p_report = sub.add_parser("report", help="print findings for a prompt / run")
    p_report.add_argument("--name", help="prompt name (uses latest run)")
    p_report.add_argument("--run", type=int, help="specific analysis_run id")
    p_report.set_defaults(func=_cmd_report)

    p_review = sub.add_parser("review", help="set the human review field")
    p_review.add_argument("finding_id", type=int)
    p_review.add_argument("--status", required=True, choices=list(REVIEW_STATES))
    p_review.add_argument("--reviewer")
    p_review.set_defaults(func=_cmd_review)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
