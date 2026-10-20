"""企业版功能门禁（开源侧，MIT）。

开源仓库只放"集成点 + 许可校验"；真正的企业实现放在私有包
`meshpilot_enterprise`（pip 安装后由本模块动态加载）。

边界：算法、可视化、基准、CLI——**永远免费，不经过本模块**。
企业付费的是流水线周边能力：
- batch_pipeline：目录批量优化 + 汇总报告（工作室资产流水线）
- ci_report：CI 门控报告（JSON/JUnit + 阈值失败退出码）
- sso：团队席位管理（SAML/OIDC）
"""
from __future__ import annotations

from functools import wraps

from .. import licensing

FEATURES = {
    "batch_pipeline": "批量流水线（目录递归优化 + 汇总报告）",
    "ci_report": "CI 集成（JUnit/JSON 报告 + 预算门控退出码）",
    "sso": "团队席位管理（SAML/OIDC 单点登录）",
}

_PLUGIN_NAME = "meshpilot_enterprise"


class EnterpriseRequired(Exception):
    """功能属于企业版：未安装扩展包或许可证无效。"""


def _upgrade_message(feature: str) -> str:
    return (
        f"「{FEATURES.get(feature, feature)}」属于 MeshPilot 企业版功能。\n"
        f"  1) 获取许可证（组织名/席位数/有效期）\n"
        f"  2) 安装私有扩展包：pip install {_PLUGIN_NAME}\n"
        f"  3) 设置环境变量 MESHPILOT_LICENSE=<你的许可证>\n"
        f"核心算法/可视化/CLI（MIT）永久免费——企业付费的是流水线能力。"
    )


def plugin():
    try:
        import importlib

        return importlib.import_module(_PLUGIN_NAME)
    except ImportError:
        return None


def is_enabled(feature: str) -> bool:
    info = licensing.current()
    if info is None or info.expired:
        return False
    if not info.has(feature):
        return False
    return plugin() is not None


def require(feature: str):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not is_enabled(feature):
                raise EnterpriseRequired(_upgrade_message(feature))
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def status() -> dict:
    info = licensing.current()
    installed = plugin() is not None
    return {
        "tier": licensing.TIER_ENTERPRISE if (info and installed) else licensing.TIER_COMMUNITY,
        "license_valid": bool(info and not info.expired),
        "plugin_installed": installed,
        "org": info.org if info else "",
        "seats": info.seats if info else 0,
        "days_left": info.days_left if info else 0,
        "features": {name: {"description": d, "enabled": is_enabled(name)}
                     for name, d in FEATURES.items()},
    }
