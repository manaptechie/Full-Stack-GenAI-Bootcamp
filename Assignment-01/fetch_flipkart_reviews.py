from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.flipkart.com/apple-iphone-15-black-128-gb/product-reviews/itm6ac6485515ae4"
COMMON_PARAMS = {
    "pid": "MOBGTAGPTB3VS24W",
    "lid": "LSTMOBGTAGPTB3VS24WVZQKQO",
    "marketplace": "FLIPKART",
}
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
REQUEST_TIMEOUT = 30


@dataclass
class ReviewRecord:
    review_title: str
    review_text: str
    sentiment: str
    sort_order: str
    page_number: int
    product_variant: str
    review_meta: str
    source_url: str


def build_url(sort_order: str, page_number: int) -> str:
    params = dict(COMMON_PARAMS)
    if sort_order != "MOST_HELPFUL":
        params["sortOrder"] = sort_order
    if page_number > 1:
        params["page"] = page_number
    return f"{BASE_URL}?{urlencode(params)}"


def normalize_text(value: str) -> str:
    return " ".join(value.split())


def extract_reviews(html: str, sort_order: str, page_number: int, source_url: str) -> list[ReviewRecord]:
    soup = BeautifulSoup(html, "lxml")
    review_records: list[ReviewRecord] = []

    for body_node in soup.select("span.css-1jxf684"):
        review_text = normalize_text(body_node.get_text(" ", strip=True))
        if not review_text or review_text.lower() == "more":
            continue

        card = body_node.find_parent("div", class_="r-w7s2jr")
        if card is None:
            continue

        children = card.find_all(recursive=False)
        title = normalize_text(children[0].get_text(" ", strip=True)) if len(children) > 0 else ""
        product_variant = normalize_text(children[1].get_text(" ", strip=True)) if len(children) > 1 else ""
        review_meta = normalize_text(children[-1].get_text(" ", strip=True)) if len(children) > 0 else ""
        sentiment = "positive" if sort_order == "POSITIVE_FIRST" else "negative"

        review_records.append(
            ReviewRecord(
                review_title=title,
                review_text=review_text,
                sentiment=sentiment,
                sort_order=sort_order,
                page_number=page_number,
                product_variant=product_variant,
                review_meta=review_meta,
                source_url=source_url,
            )
        )

    return review_records


def collect_reviews(target_count: int, max_pages_per_sort: int) -> list[ReviewRecord]:
    session = requests.Session()
    session.headers.update(HEADERS)
    collected: list[ReviewRecord] = []
    seen_texts: set[str] = set()

    sort_orders = ("POSITIVE_FIRST", "NEGATIVE_FIRST")
    target_per_class = max(target_count // len(sort_orders), 1)

    for sort_order in sort_orders:
        class_count = 0
        for page_number in range(1, max_pages_per_sort + 1):
            url = build_url(sort_order=sort_order, page_number=page_number)
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            for review in extract_reviews(response.text, sort_order, page_number, url):
                normalized_key = review.review_text.casefold()
                if normalized_key in seen_texts:
                    continue

                seen_texts.add(normalized_key)
                collected.append(review)
                class_count += 1

                if class_count >= target_per_class:
                    break

            if class_count >= target_per_class:
                break

    return collected[:target_count]


def to_dataframe(records: Iterable[ReviewRecord]) -> pd.DataFrame:
    dataframe = pd.DataFrame(asdict(record) for record in records)
    if dataframe.empty:
        return dataframe
    return dataframe.sort_values(["sentiment", "page_number", "review_title", "review_text"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Flipkart reviews for Assignment 01.")
    parser.add_argument(
        "--output",
        default=Path(__file__).with_name("flipkart_reviews.csv"),
        type=Path,
        help="Path to the CSV file that will be created.",
    )
    parser.add_argument(
        "--target-count",
        default=110,
        type=int,
        help="Total number of reviews to collect across positive and negative sorts.",
    )
    parser.add_argument(
        "--max-pages-per-sort",
        default=8,
        type=int,
        help="Maximum pages to scrape for each review sort order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = collect_reviews(target_count=args.target_count, max_pages_per_sort=args.max_pages_per_sort)
    dataframe = to_dataframe(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(args.output, index=False, encoding="utf-8")

    print(f"Saved {len(dataframe)} reviews to {args.output}")
    if not dataframe.empty:
        print(dataframe["sentiment"].value_counts().to_string())


if __name__ == "__main__":
    main()