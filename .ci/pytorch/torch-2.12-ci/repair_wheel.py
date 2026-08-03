import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from wheel.wheelfile import WheelFile

WHEEL_DIR = Path("/dist/raw")

# ===== 查找原始 wheel =====
wheel_files = list(WHEEL_DIR.glob("*.whl"))

if len(wheel_files) == 0:
    raise RuntimeError(f"{WHEEL_DIR} 中没有找到 wheel")

if len(wheel_files) > 1:
    raise RuntimeError(
        "发现多个 wheel，无法确定要处理哪个：\n"
        + "\n".join(str(path) for path in wheel_files)
    )

wheel_path = wheel_files[0]

new_version = os.environ.get("PYTORCH_BUILD_VERSION", "").strip()

if not new_version:
    raise RuntimeError("环境变量 PYTORCH_BUILD_VERSION 为空")

print(f"old wheel: {wheel_path.name}")
print(f"new version: {new_version}")

# ===== 生成新 wheel 文件名 =====
name_parts = wheel_path.name.split("-")

if len(name_parts) < 5:
    raise RuntimeError(f"无法解析 wheel 文件名：{wheel_path.name}")

name_parts[1] = new_version
new_wheel_name = "-".join(name_parts)
new_wheel_path = wheel_path.parent / new_wheel_name

# 防止之前失败留下同名文件
if new_wheel_path.exists() and new_wheel_path != wheel_path:
    new_wheel_path.unlink()
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)

        # ===== 解压原始 wheel =====
        with zipfile.ZipFile(wheel_path, "r") as zip_file:
            zip_file.extractall(temp_path)

        # ===== 检查原始 dist-info =====
        dist_info_dirs = [
            path
            for path in temp_path.iterdir()
            if path.is_dir() and path.name.endswith(".dist-info")
        ]

        if len(dist_info_dirs) != 1:
            raise RuntimeError(
                "原始 wheel 中 dist-info 数量不正确：\n"
                + "\n".join(path.name for path in dist_info_dirs)
            )

        old_dist_info = dist_info_dirs[0]
        with WheelFile(new_wheel_path, "w") as wheel_file:
            new_dist_info = temp_path / wheel_file.dist_info_path

            print(f"old dist-info: {old_dist_info.name}")
            print(f"new dist-info: {new_dist_info.name}")

            if old_dist_info != new_dist_info:
                if new_dist_info.exists():
                    shutil.rmtree(new_dist_info)

                old_dist_info.rename(new_dist_info)
            # ===== 修改 METADATA 中的版本 =====
            metadata_path = new_dist_info / "METADATA"

            if not metadata_path.exists():
                raise RuntimeError(
                    f"没有找到 METADATA：{metadata_path}"
                )
            metadata_lines = metadata_path.read_text(
                encoding="utf-8"
            ).splitlines()

            new_metadata_lines = []
            version_found = False
            for line in metadata_lines:
                if line.startswith("Version:"):
                    line = f"Version: {new_version}"
                    version_found = True

                new_metadata_lines.append(line)

            if not version_found:
                raise RuntimeError("METADATA 中没有找到 Version 字段")
            metadata_path.write_text(
                "\n".join(new_metadata_lines) + "\n",
                encoding="utf-8",
            )

            # ===== 写入新 wheel，并重新生成 RECORD =====
            wheel_file.write_files(temp_path)
    # ===== 验证新 wheel 中只能有一个 dist-info =====
    with zipfile.ZipFile(new_wheel_path, "r") as zip_file:
        dist_info_names = sorted(
            {
                name.split("/", 1)[0]
                for name in zip_file.namelist()
                if ".dist-info/" in name
            }
        )
    print("dist-info directories:")
    for name in dist_info_names:
        print(f"  {name}")

    if len(dist_info_names) != 1:
        raise RuntimeError(
            "修正后的 wheel 中存在多个 dist-info：\n"
            + "\n".join(dist_info_names)
        )
    # 新 wheel 完整生成并验证通过后，才删除旧 wheel
    if new_wheel_path != wheel_path:
        wheel_path.unlink()
    print(f"new wheel: {new_wheel_path}")

except Exception:
    # 处理失败时删除不完整的新 wheel，保留原始 wheel
    if new_wheel_path.exists() and new_wheel_path != wheel_path:
        new_wheel_path.unlink()

    raise
