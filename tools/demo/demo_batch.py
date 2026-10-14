"""批量处理多视频的 GVHMR 推理脚本"""
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
from hmr4d.utils.pylogger import Log
import subprocess
import sys


def collect_videos(root_dir: Path, pattern="*.mp4"):
    """递归收集所有视频文件"""
    videos = sorted(root_dir.rglob(pattern))
    videos.extend(sorted(root_dir.rglob(pattern.upper())))
    return sorted(set(videos))


def get_unique_output_name(video_path: Path, root_dir: Path):
    """生成唯一的输出名称：parent_folder/video_stem"""
    rel_path = video_path.relative_to(root_dir)
    parent = rel_path.parent
    stem = video_path.stem
    if parent == Path("."):
        return stem
    return f"{parent}/{stem}".replace("/", "_")


def copy_json_to_source(output_dir: Path, video_path: Path):
    """将生成的 JSON 拷贝到视频源文件夹"""
    json_src = output_dir / "smpl_output.json"
    if not json_src.exists():
        Log.warn(f"JSON 不存在: {json_src}")
        return
    
    json_dst = video_path.parent / f"{video_path.stem}.json"
    shutil.copy2(json_src, json_dst)
    Log.info(f"JSON 已拷贝: {json_dst}")


def main():
    parser = argparse.ArgumentParser(description="批量处理多视频 GVHMR 推理")
    parser.add_argument("-f", "--folder", type=str, required=True, help="视频根目录")
    parser.add_argument("-d", "--output_root", type=str, default="outputs/demo_batch", help="输出根目录")
    parser.add_argument("-s", "--static_cam", action="store_true", help="静态相机模式")
    parser.add_argument("--f_mm", type=int, default=None, help="焦距(mm)")
    parser.add_argument("--pattern", type=str, default="*.mp4", help="视频文件匹配模式")
    parser.add_argument("--n_people", type=int, default=1, help="跟踪人数（默认1）")
    parser.add_argument("--no_center_priority", action="store_true", help="禁用中心优先排序")
    args = parser.parse_args()

    root_dir = Path(args.folder)
    if not root_dir.exists():
        Log.error(f"目录不存在: {root_dir}")
        sys.exit(1)

    # 收集所有视频
    videos = collect_videos(root_dir, args.pattern)
    Log.info(f"找到 {len(videos)} 个视频文件")

    # 批量处理
    for video_path in tqdm(videos, desc="批量处理"):
        # 生成唯一输出名称
        unique_name = get_unique_output_name(video_path, root_dir)
        output_dir = Path(args.output_root) / unique_name
        
        # 构建命令
        cmd = [
            "python", "tools/demo/demo.py",
            "--video", str(video_path),
            f"video_name='{unique_name}'",  # 添加引号以支持中文
            f"output_root={args.output_root}",
            f"n_people={args.n_people}",
        ]
        
        if args.static_cam:
            cmd.append("-s")
        if args.f_mm is not None:
            cmd.extend(["--f_mm", str(args.f_mm)])
        if args.no_center_priority:
            cmd.append(f"center_priority=False")
        
        # 执行推理
        Log.info(f"处理: {video_path.relative_to(root_dir)}")
        try:
            subprocess.run(cmd, check=True)
            # 拷贝 JSON 到源文件夹
            copy_json_to_source(output_dir, video_path)
        except subprocess.CalledProcessError as e:
            Log.error(f"处理失败: {video_path}, 错误: {e}")
            continue

    Log.info(f"批量处理完成，共处理 {len(videos)} 个视频")


if __name__ == "__main__":
    main()
