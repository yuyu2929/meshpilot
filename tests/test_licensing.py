"""许可证测试：签发/验证/篡改/过期/伪造降级。"""
from __future__ import annotations

import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from meshpilot import licensing


def _make_pair():
    pk = Ed25519PrivateKey.generate()
    return pk.private_bytes_raw(), pk.public_key().public_bytes_raw()


def test_sign_and_verify_roundtrip():
    priv_raw, pub_raw = _make_pair()
    lic = licensing.sign_license({
        "tier": "enterprise", "org": "测试工作室", "seats": 10,
        "features": ["batch_pipeline"], "issued_at": time.time(),
        "expires_at": time.time() + 86400,
    }, priv_raw)
    info = licensing.parse(lic, public_key_raw=pub_raw)
    assert info is not None
    assert info.org == "测试工作室"
    assert info.seats == 10
    assert info.has("batch_pipeline")
    assert not info.expired


def test_tampered_seats_rejected():
    priv_raw, pub_raw = _make_pair()
    lic = licensing.sign_license({
        "tier": "enterprise", "org": "A", "seats": 1,
        "features": ["sso"], "issued_at": time.time(),
        "expires_at": time.time() + 86400,
    }, priv_raw)
    # 篡改 seats 1 → 999（重签 body 但无私钥 → 签名对不上）
    prefix, body, sig = lic.split(".")
    import base64, json
    payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    payload["seats"] = 999
    forged_body = base64.urlsafe_b64encode(
        json.dumps(payload, sort_keys=True).encode()).decode().rstrip("=")
    forged = f"{prefix}.{forged_body}.{sig}"
    assert licensing.parse(forged, public_key_raw=pub_raw) is None


def test_expired_license():
    priv_raw, pub_raw = _make_pair()
    lic = licensing.sign_license({
        "tier": "enterprise", "org": "A", "seats": 1,
        "features": ["sso"], "issued_at": time.time() - 172800,
        "expires_at": time.time() - 86400,
    }, priv_raw)
    info = licensing.parse(lic, public_key_raw=pub_raw)
    assert info is not None
    assert info.expired
    assert info.days_left == 0


def test_garbage_rejected():
    priv_raw, pub_raw = _make_pair()
    assert licensing.parse("", public_key_raw=pub_raw) is None
    assert licensing.parse("MP1.garbage", public_key_raw=pub_raw) is None
    assert licensing.parse("not-a-license", public_key_raw=pub_raw) is None


def test_wrong_key_rejected():
    priv_raw, _ = _make_pair()
    _, other_pub = _make_pair()
    lic = licensing.sign_license({
        "tier": "enterprise", "org": "A", "seats": 1,
        "features": ["sso"], "issued_at": time.time(),
        "expires_at": time.time() + 86400,
    }, priv_raw)
    assert licensing.parse(lic, public_key_raw=other_pub) is None


# ---------------- 缓存（v0.1.1） ----------------

def test_parse_is_cached():
    """同一串许可证只验签一次：第二次返回同一个对象。"""
    priv_raw, pub_raw = _make_pair()
    lic = licensing.sign_license({
        "tier": "enterprise", "org": "缓存测试", "seats": 1,
        "features": ["ci_report"], "issued_at": time.time(),
        "expires_at": time.time() + 3600,
    }, priv_raw)
    licensing.clear_cache()
    a = licensing.parse(lic, public_key_raw=pub_raw)
    b = licensing.parse(lic, public_key_raw=pub_raw)
    assert a is not None and a is b


def test_clear_cache_forces_revalidation():
    priv_raw, pub_raw = _make_pair()
    lic = licensing.sign_license({"org": "x", "features": [], "seats": 1,
                                  "issued_at": 0, "expires_at": 1e12}, priv_raw)
    licensing.clear_cache()
    a = licensing.parse(lic, public_key_raw=pub_raw)
    licensing.clear_cache()
    b = licensing.parse(lic, public_key_raw=pub_raw)
    assert a is not None and b is not None and a is not b