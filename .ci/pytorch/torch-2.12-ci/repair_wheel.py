import os
import zipfile
import tempfile
from pathlib import Path
from wheel.wheelfile import WheelFile


wheel_path = list(Path("/dist/raw").glob("*.whl"))[0]

new_version = os.environ["PYTORCH_BUILD_VERSION"]

print("old wheel:", wheel_path.name)
print("new version:", new_version)


with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)

    with zipfile.ZipFile(wheel_path) as z:
        z.extractall(tmp)

    dist_info = next(tmp.glob("*.dist-info"))

    new_dist_info = tmp / (
        dist_info.name.split("-")[0]
        + "-"
        + new_version.replace("+", ".")
        + ".dist-info"
    )

    dist_info.rename(new_dist_info)

    metadata = new_dist_info / "METADATA"

    text = metadata.read_text()

    text = text.replace(
        text.split("Version: ")[1].split("\n")[0],
        new_version
    )

    metadata.write_text(text)


    parts = wheel_path.name.split("-")
    parts[1] = new_version

    new_path = wheel_path.parent / "-".join(parts)

    with WheelFile(new_path, "w") as wf:
        wf.write_files(tmp)


wheel_path.unlink()

print("new wheel:", new_path)