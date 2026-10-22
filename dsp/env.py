"""第三方库引导。

仓库自带的 python（3.12.14）只有 numpy / pandas / python-docx / python-pptx / Pillow，
**没有 scipy、matplotlib、scikit-learn**。这三个装在归档的 `task1_w2/_env/pylibs`
（scipy 1.16.3 / matplotlib 3.10.6 / sklearn 1.7.2，已实测可加载）。

用法：脚本里第一件事调用 ``add_pylibs()``，之后再 import scipy / matplotlib / sklearn。

    from dsp.env import add_pylibs
    add_pylibs()
    import scipy, matplotlib, sklearn

以后如果把 `_env` 挪到仓库根目录，只改 _CANDIDATES 即可，其它代码不用动。
"""

import sys

from .config import REPO_ROOT

# 查找顺序：仓库根优先，其次归档里的旧位置
_CANDIDATES = (
    REPO_ROOT / "_env" / "pylibs",
    REPO_ROOT / "task1_w2" / "_env" / "pylibs",
)


def pylibs_dir():
    """返回实际存在的 pylibs 目录，找不到返回 None。"""
    for cand in _CANDIDATES:
        if cand.is_dir():
            return cand
    return None


def add_pylibs():
    """把 pylibs 插到 sys.path 最前面，返回其路径；找不到返回 None。"""
    found = pylibs_dir()
    if found is None:
        return None
    path = str(found)
    if path not in sys.path:
        sys.path.insert(0, path)
    return found
