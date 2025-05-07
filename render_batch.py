import os
import cv2  
import random
import numpy as np 
from tqdm import tqdm   
from functools import partial
import multiprocessing as mp 
from videotool.io import export_video, load_video 
from videotool import utils as vtutils 
 

num_threads = 10
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)


def build_run_blender(fid, mesh_folder, driven):   
    cmd = f'blender -b ./resource/soap.blend -P render_4view.py -- '
    cmd += f'--mesh_folder {mesh_folder}/recon-teeth '
    cmd += f'--frame_id {fid} '
    cmd += f'--sequence {mesh_folder}/animation/{driven}-seq.npy ' 
    cmd += f'--out_dir {mesh_folder}/animation/{driven} '

    if os.path.exists(f'{mesh_folder}/animation/{driven}-seq.npy'):
        os.system(cmd)


def compose_video(fp, driven_path, out_path):   
    video_four_views = [
        load_video(fp+f'_{i}.mp4')  for i in range(4)
    ]
    driven = load_video(driven_path)
    driven = vtutils.resize_video(driven, (1024, -1)) 
    stack_video = vtutils.hstack_videos([driven] + video_four_views) 
    export_video(stack_video, out_path)


if __name__ == '__main__':
    import argparse
    import numpy as np 
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser(description="Render and compose animation videos.")
    parser.add_argument('--data_dir', type=str, required=True, help='Base data directory.')
    parser.add_argument('--exp', type=str, default='newest', help='Experiment name.')
    parser.add_argument('--driven', type=str, default='0000', help='Driven ID.')
    parser.add_argument('--subject', type=str, required=True, help='Subject UUID.') 
    parser.add_argument('--type', type=str, default='4view', help='[4view, rotation]')
    args = parser.parse_args()

    default_teeth_transl = {
        'lower': [0, -0.015, -0.003],
        'upper': [0, -0.003, -0.003],  
    }

    input_folder = f'{args.data_dir}/{args.subject}/6-views/{args.exp}' 
    teeth_lowt = default_teeth_transl['lower']
    teeth_uppt = default_teeth_transl['upper'] 

    out_path=f"{args.type}_{args.driven}_{args.subject}.mp4" 

    if not os.path.exists(out_path):   
        if not os.path.exists(f'{input_folder}/animation/{args.driven}-seq.npy'): 
            os.system(
                f'python animation.py --data_dir {args.data_dir} --exp {args.exp} --subject {args.subject} --driven {args.driven} '
                f'--teeth_lowt {",".join(map(str, teeth_lowt))} '
                f'--teeth_uppt {",".join(map(str, teeth_uppt))}'
            ) 
        
        fnum = len(np.load(f'{input_folder}/animation/{args.driven}-seq.npy'))   
        with mp.Pool(processes=10, maxtasksperchild=1) as pool:
            for _ in tqdm(
                pool.imap_unordered(
                    partial(
                        build_run_blender,
                        mesh_folder=input_folder, 
                        driven=args.driven
                    ),
                    range(fnum),
                ),
                total=fnum,
            ): 
                pass

        pool.close()
        pool.join()  

    for i in range(4):
        os.system(
            f'ffmpeg -y -framerate 30 -pattern_type glob -i "{input_folder}/animation/{args.driven}/*_{i:04d}.png" '
            f'-c:v libx264 -pix_fmt yuv420p {input_folder}/animation/{args.driven}_{i}.mp4'
        )
    
    compose_video(
        f"{input_folder}/animation/{args.driven}", 
        driven_path=f"{args.data_dir}/../data/driven/{args.driven}.mp4",
        out_path=f"{args.type}_{args.driven}_{args.subject}.mp4"
    )

