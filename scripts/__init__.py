"""实验脚本包：`python -m scripts.xxx` 的方式运行。

导入本包时先把 `_env/pylibs`（scipy / matplotlib / scikit-learn）挂到 sys.path 上，
所以每个脚本都不用再自己调 add_pylibs()。只用 numpy 的脚本不受影响。
"""

from dsp.env import add_pylibs

add_pylibs()
