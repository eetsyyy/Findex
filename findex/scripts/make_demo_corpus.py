"""Generate deterministic, original Ukrainian sample texts (no external downloads)."""

import argparse
from pathlib import Path

PARAGRAPHS = [
    "Python підтримує генератори та ітератори. Пошукова система аналізує документи. ",
    "Індекс допомагає знаходити слова у великому корпусі текстів. Київ — місто України. ",
    "Нормалізація Unicode об'єднує однакові слова. Токенізація розбиває текст на терміни. ",
    "Продуктивність програми вимірюють часом і використанням пам'яті. ",
    "Ліниве обчислення читає файли по одному, а жадібне накопичує списки. ",
    "Документ містить назву, номер, приклади та пояснення алгоритмів. ",
    "Словник термінів дає статистику, а лічильник визначає частоту. ",
    "Українська мова містить апостроф: п'ять, м'ята та зв'язок. ",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", type=int, default=120)
    parser.add_argument("--repeat", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.documents < 1 or args.repeat < 1:
        parser.error("--documents and --repeat must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    for i in range(args.documents):
        paragraph = PARAGRAPHS[i % len(PARAGRAPHS)]
        text = f"Стаття {i + 1}. Тема: {i % len(PARAGRAPHS) + 1}.\n" + paragraph * args.repeat
        (args.output / f"article_{i:04}.txt").write_text(text, encoding="utf-8")
    print(f"Generated {args.documents} documents in {args.output}")


if __name__ == "__main__":
    main()
