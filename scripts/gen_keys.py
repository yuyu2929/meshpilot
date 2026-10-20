"""生成 MeshPilot 许可证密钥对。

    python scripts/gen_keys.py

产出：
- meshpilot/license_public.pem   （随仓库分发）
- keys/meshpilot-license-private.pem（gitignored，权限 600）
"""
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    keys_dir = root / "keys"
    keys_dir.mkdir(exist_ok=True)

    pk = Ed25519PrivateKey.generate()
    priv_path = keys_dir / "meshpilot-license-private.pem"
    pub_path = root / "meshpilot" / "license_public.pem"

    priv_path.write_bytes(
        pk.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    import os
    os.chmod(priv_path, 0o600)
    pub_path.write_bytes(
        pk.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ))
    print(f"私钥 → {priv_path}（600，gitignored，勿外传）")
    print(f"公钥 → {pub_path}（随仓库分发）")


if __name__ == "__main__":
    main()
