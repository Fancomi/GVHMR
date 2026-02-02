
# 必须使用cuda12.1, 否则预测严重bug
# python tools/demo/demo.py --video=/media/fanco/disk/Code/3DBody/HybrIK/examples/taiji.mp4 -s
# python tools/demo/demo.py --video=docs/example_video/swim-train.mp4 -s
# python tools/demo/demo.py --video=/media/fanco/disk/Videos/rowboat/20250526/20250526-160510_orb1.mp4 -s
# python tools/demo/demo.py --video=docs/example_video/20251126-101622_zcam2.mp4 -s
# python tools/demo/demo.py --video=/media/fanco/disk/Datas/muscle_wiki/wiki_videos/female/abdominals/abdominals-stretch-variation-one/side.mp4 -s
# python tools/demo/demo.py --video=/media/fanco/disk/Datas/muscle_wiki/wiki_videos/female/biceps/barbell-pronated-row/side.mp4 -s
# python tools/demo/demo.py -s --video="/media/fanco/disk/Datas/3DMocap/20251225/20251226-2/四足支撑前伸/color_m.mp4"


python tools/demo/demo.py --video=/media/fanco/disk/BaiduPan/weightlifting20251230.mp4 -s --use_dpvo --center_priority


# python tools/demo/demo.py --video=/media/fanco/disk/BaiduPan/weightlifting20251230.mp4 -s

# python tools/demo/demo.py --video=/media/fanco/disk/Datas/3DMocap/20251225/20260112/反手_cut/color_r.mp4 -s --center_priority
# python tools/demo/demo.py --video=/media/fanco/disk/Datas/3DMocap/20251225/20260112/反手_cut/color_l.mp4 -s --center_priority
# python tools/demo/demo.py --video=/media/fanco/disk/Datas/3DMocap/20251225/20260112/反手_cut/color_m.mp4 -s --center_priority

# 批处理
# python tools/demo/demo_batch.py -f /media/fanco/disk/Datas/3DMocap/20251225/20260112 -s