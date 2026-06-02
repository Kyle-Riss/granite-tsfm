# Copyright contributors to the TSFM project
#
"""KPX CSV 원격 다운로드 시도 (공공데이터포털 fileData).

이 데이터셋(15133498)은 OpenAPI가 아니라 **파일 직접 다운로드** 형식입니다.
포털은 로그인·세션·Referer 검사 때문에 스크립트만으로는 자주 403 됩니다.

권장 순서:
  1) 브라우저에서 CSV 받기 → Data/kpx/kpx_hourly.csv
  2) 또는 로그인 후 «다운로드» 링크 URL 복사 → 아래 --url
  3) 당장은: uv run python -m pipelines.seoul.bootstrap_kpx_demo

  uv run python -m pipelines.seoul.download_kpx --url 'https://...'
  KPX_DOWNLOAD_URL='https://...' uv run python -m pipelines.seoul.download_kpx
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pipelines.seoul.config import DATA_DIR

KPX_DIR = DATA_DIR / "kpx"
DEFAULT_OUT = KPX_DIR / "kpx_hourly.csv"

# 공공데이터포털 fileData 페이지 (브라우저에서 열기)
PORTAL_PAGE = "https://www.data.go.kr/data/15133498/fileData.do"


def download_file(url: str, output: Path, timeout: int = 120) -> None:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; granite-tsfm-pipeline/1.0)",
        "Referer": PORTAL_PAGE,
    }
    req = Request(url, headers=headers)
    output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if len(data) < 500:
        raise ValueError(f"Download too small ({len(data)} bytes); likely HTML error page, not CSV.")
    output.write_bytes(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download KPX CSV if you have a direct URL.")
    parser.add_argument(
        "--url",
        type=str,
        default=os.environ.get("KPX_DOWNLOAD_URL"),
        help="Direct download URL (copy from browser after login on data.go.kr)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.url:
        print(
            "원격 자동 다운로드가 막혀 있는 경우가 많습니다.\n\n"
            "이유:\n"
            "  • 데이터셋 15133498 = fileData (CSV 일괄), OpenAPI 아님\n"
            "  • data.go.kr 은 비로그인·봇 요청에 403\n"
            "  • 다운로드 링크는 로그인 세션·일회성 URL인 경우가 많음\n\n"
            f"수동: 브라우저에서 열기 → {PORTAL_PAGE}\n"
            f"       저장: {args.output}\n\n"
            "또는 로그인 후 «다운로드» 우클릭 → 링크 주소 복사:\n"
            "  uv run python -m pipelines.seoul.download_kpx --url '붙여넣기'\n\n"
            "파이프라인만 확인:\n"
            "  uv run python -m pipelines.seoul.bootstrap_kpx_demo\n"
        )
        return 1

    if args.dry_run:
        print(f"Would download:\n  {args.url}\n  → {args.output}")
        return 0

    try:
        download_file(args.url, args.output)
    except (HTTPError, URLError, ValueError) as e:
        print(f"Download failed: {e}\n")
        print("→ 브라우저로 CSV를 받아 두거나 bootstrap_kpx_demo 를 사용하세요.")
        return 1

    print(f"Saved {args.output} ({args.output.stat().st_size:,} bytes)")
    print("Next: uv run python -m pipelines.seoul.ingest_kpx --input", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
