# Adam 长时 p-tVMC 诊断文稿

主文件：`adam_long_time_ptvmc_diagnostics.tex`。正文使用中文，含公式、参数设置、五张已有图片、两张数据表和实现限制说明。只包含已完成的 Adam 数据。

在此目录执行以下命令，使用 XeLaTeX 连续编译两次以解析交叉引用：

```powershell
xelatex -interaction=nonstopmode -halt-on-error adam_long_time_ptvmc_diagnostics.tex
xelatex -interaction=nonstopmode -halt-on-error adam_long_time_ptvmc_diagnostics.tex
```

依赖标准 TeX Live/MiKTeX 的 ctex、Fandol 中文字体、amsmath、graphicx、booktabs、tabularx、geometry、caption、float、placeins、hyperref、xurl 等包。本机本次未发现可调用的 TeX 编译器，因此交付 LaTeX 源文稿，不声称已通过排版编译。

也可以把主文件、`tables/`、`figures/` 一起上传到 Overleaf，选择 XeLaTeX 编译器，设置主文件后编译。数据表与图片已放在文稿目录内，不依赖原始 D 盘绝对路径。

`prepare_adam_report_assets.py` 从原始结果读取数值表，复制既有图片，生成 SHA-256 输入指纹 `data_manifest.json`。如果需要刷新本报告的数据副本，从本目录运行：

```powershell
python prepare_adam_report_assets.py
```

该脚本不修改实验结果。刷新会更新本报告的表格、图片和指纹；正文数值及解释需要相应复核。当前固定数据为231个 A 拟合任务、21个 B/C 点和41个轨迹记录。

报告明确记录两点实现限制：轨迹 `projection_loss` 是最后一次内部更新前的量；A 的 `best_loss` 与保存参数存在一次更新的错位。本报告不修复计算代码，也不重跑或覆盖既有实验。
