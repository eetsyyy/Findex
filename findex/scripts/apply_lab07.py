"""Safely integrate the Lab 07 CLI entrypoint and project version."""
from pathlib import Path

root = Path(__file__).resolve().parent.parent
project = root / "pyproject.toml"
cli = root / "src" / "findex" / "cli.py"

config = project.read_text(encoding="utf-8")
if 'version = "0.6.0"' in config:
    project.write_text(config.replace('version = "0.6.0"', 'version = "0.7.0"', 1), encoding="utf-8")
elif 'version = "0.7.0"' not in config:
    raise SystemExit("Невідома версія у pyproject.toml: перевір вручну.")

text = cli.read_text(encoding="utf-8")
if '@app.command("serve")' not in text:
    marker = "def main() -> None:"
    if marker not in text:
        raise SystemExit("Не знайшов функцію main() у CLI. Файл не змінено.")
    addition = '''@app.command("serve")
def serve(
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8000,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
) -> None:
    """Запустити вебсервер Findex."""
    import uvicorn

    uvicorn.run("findex.web:app", host="0.0.0.0", port=port, workers=workers)


'''
    cli.write_text(text.replace(marker, addition + marker, 1), encoding="utf-8")
print("Lab 07: CLI serve і версія 0.7.0 встановлені.")
