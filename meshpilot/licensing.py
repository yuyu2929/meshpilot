"""Ed25519 许可证验证（MeshPilot 开源侧）。

与 aria-agent / FPTES 同一套设计（格式 MP1.<payload>.<sig>）：
- 非对称签名而非 HMAC：HMAC 密钥随源码分发 = 谁都能伪造
- 公钥随仓库分发（license_public.pem），私钥只在 keys/（gitignored，600）
- 任何篡改（seats/org/有效期）→ 验签失败 → 静默降级社区版

核心算法/可视化/基准**永远不经过这里**——企业授权只解锁
batch_pipeline / ci_report / sso 三类流水线周边能力。
"""
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

PREFIX = "MP1"
TIER_COMMUNITY = "community"
TIER_ENTERPRISE = "enterprise"

_PUBKEY_PATH = Path(__file__).parent / "license_public.pem"
_ENV_VAR = "MESHPILOT_LICENSE"
_LICENSE_FILE = Path("keys/license.txt")


@dataclass
class LicenseInfo:
    tier: str
    org: str
    seats: int
    features: tuple[str, ...]
    issued_at: float
    expires_at: float

    @property
    def expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def days_left(self) -> int:
        return max(0, int((self.expires_at - time.time()) / 86400))

    def has(self, feature: str) -> bool:
        return feature in self.features


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def sign_license(payload: dict, private_key_raw: bytes) -> str:
    """签发许可证（只有持私钥的人能调）。"""
    body = _b64e(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())
    sig = _b64e(Ed25519PrivateKey.from_private_bytes(private_key_raw).sign(body.encode()))
    return f"{PREFIX}.{body}.{sig}"


def _load_public_key() -> Ed25519PublicKey | None:
    if not _PUBKEY_PATH.exists():
        return None
    try:
        from cryptography.hazmat.primitives.serialization import load_pem_public_key

        return load_pem_public_key(_PUBKEY_PATH.read_bytes())
    except Exception:
        return None


_cache: dict[tuple, "LicenseInfo | None"] = {}


def clear_cache() -> None:
    """测试或热更换许可证时清缓存。"""
    _cache.clear()
    _file_cache["mtime"] = None
    _file_cache["raw"] = ""


def parse(license_str: str, public_key_raw: bytes | None = None) -> LicenseInfo | None:
    """解析并验证许可证。任何失败返回 None（调用方按社区版处理）。

    结果按 (许可证字符串, 公钥) 缓存：Ed25519 验签不便宜，而
    enterprise.is_enabled() 会被每次请求/每次调用触发。
    """
    key = (license_str or "", public_key_raw)
    if key in _cache:
        return _cache[key]
    info = _parse_uncached(license_str, public_key_raw)
    _cache[key] = info
    return info


def _parse_uncached(license_str: str,
                    public_key_raw: bytes | None = None) -> LicenseInfo | None:
    if not license_str or not license_str.startswith(PREFIX + "."):
        return None
    parts = license_str.split(".")
    if len(parts) != 3:
        return None
    _, body, sig = parts
    if public_key_raw is not None:
        pub = Ed25519PublicKey.from_public_bytes(public_key_raw)
    else:
        pub = _load_public_key()
    if pub is None:
        return None
    try:
        pub.verify(_b64d(sig), body.encode())
    except Exception:
        return None  # 伪造/篡改 → 静默降级（含 InvalidSignature 与其他解析异常）
    try:
        payload = json.loads(_b64d(body))
        return LicenseInfo(
            tier=payload.get("tier", TIER_ENTERPRISE),
            org=payload.get("org", ""),
            seats=int(payload.get("seats", 1)),
            features=tuple(payload.get("features", ())),
            issued_at=float(payload.get("issued_at", 0)),
            expires_at=float(payload.get("expires_at", 0)),
        )
    except Exception:
        return None


_file_cache: dict = {"mtime": None, "raw": ""}


def _read_file_license() -> str:
    """按 mtime 缓存许可证文件，避免每次请求都读盘。"""
    try:
        st = _LICENSE_FILE.stat()
    except OSError:
        return ""
    if _file_cache["mtime"] != st.st_mtime:
        try:
            _file_cache["raw"] = _LICENSE_FILE.read_text().strip()
            _file_cache["mtime"] = st.st_mtime
        except OSError:
            return ""
    return _file_cache["raw"]


def current() -> LicenseInfo | None:
    """当前生效的许可证：环境变量优先，其次 keys/license.txt。"""
    raw = os.environ.get(_ENV_VAR, "").strip() or _read_file_license()
    return parse(raw)
