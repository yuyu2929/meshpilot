"""签发 MeshPilot 企业许可证。

    python scripts/issue_license.py --org "某游戏工作室" --seats 50 --days 365 \
        --features batch_pipeline,ci_report,sso

产出：keys/demo-license.txt（发给客户，客户设 MESHPILOT_LICENSE=<内容>）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meshpilot import licensing  # noqa: E402

PRIV_PATH = Path("keys/meshpilot-license-private.pem")


def main() -> None:
    parser = argparse.ArgumentParser(description="签发 MeshPilot 企业许可证")
    parser.add_argument("--org", required=True)
    parser.add_argument("--seats", type=int, default=50)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--features", default="batch_pipeline,ci_report,sso")
    parser.add_argument("--out", default="keys/demo-license.txt")
    args = parser.parse_args()

    if not PRIV_PATH.exists():
        sys.exit(f"先生成密钥对：python scripts/gen_keys.py（期待 {PRIV_PATH}）")

    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    pk = load_pem_private_key(PRIV_PATH.read_bytes(), password=None)
    priv_raw = pk.private_bytes_raw()

    now = time.time()
    payload = {
        "tier": licensing.TIER_ENTERPRISE,
        "org": args.org,
        "seats": args.seats,
        "features": [f.strip() for f in args.features.split(",") if f.strip()],
        "issued_at": now,
        "expires_at": now + args.days * 86400,
    }
    lic = licensing.sign_license(payload, priv_raw)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(lic)
    print(f"许可证已签发 → {out}")
    print(f"  org={args.org} seats={args.seats} days={args.days} "
          f"features={payload['features']}")
    print(f"客户配置：export MESHPILOT_LICENSE={lic[:40]}...")


if __name__ == "__main__":
    main()
