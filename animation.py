import os  
import sys 
sys.path.append('..') 
import cv2 
import torch
import random 
import numpy as np
from kiui.mesh import Mesh
from PIL import Image
import smplx 
from smplx.render import Renderer  
# from src.utils.helper import create_flame_model 



def create_flame_model(model_path="/media/mbzuai/Tingting/projects/soap/data/flame/flame2020.pkl", upsample=False, create_segms=True, add_teeth=False, device='cuda'): 
    model = smplx.create(
        model_path=os.path.expanduser(model_path), 
        model_type='flame',   
        batch_size=1,
        upsample=upsample, 
        create_segms=create_segms, 
        add_teeth=add_teeth
    ).to(device)
    return model 


def add_the_teeth(betas, teeth_lowt, teeth_uppt):
    teeth_mesh = Mesh.load_obj(f'{prj_dir}/data/teeth/teeth_tri2.obj', device=device)
    teeth_mesh.auto_normal()
    # id_mesh = Mesh.load_obj(f'{prj_dir}/data/head_template.obj', device=device)
    id_mesh = body_model.v_template
    
    body_model.set_params(flame_attributes)
    target = body_model(betas=betas).vertices[0]
    trans = (target[3797]+target[3920]-id_mesh[3797]-id_mesh[3920])/2
    scale = torch.norm(target[3797]-target[3920]) / torch.norm(id_mesh[3797]-id_mesh[3920])
    center = torch.mean(teeth_mesh.v, dim=0)
    teeth_mesh.v = (teeth_mesh.v-center)*scale+center+trans
    
    # load skinning and rigging
    with open(f'{prj_dir}/data/teeth/teeth_mask.txt') as file:
        teeth_mask = [int(line.rstrip()) for line in file]
    
    lbs_temp = []
    transl = []   
    for i in range(teeth_mesh.v.shape[0]):
        if i in teeth_mask:
            lbs_temp.append([0,0,1,0,0])
            # transl.append([0, -0.015, -0.005]) 
            transl.append(teeth_lowt)
        else:
            lbs_temp.append([0,1,0,0,0])
            # transl.append([0, -0.003, -0.005]) 
            transl.append(teeth_uppt)
    transl = torch.tensor(transl).float().to(device)
    teeth_mesh.v += transl   
    flame_attributes['lbs_weights'] = torch.cat((flame_attributes['lbs_weights'], torch.tensor(lbs_temp).to(device)), 0)
    flame_attributes['v_template'] = torch.cat((flame_attributes['v_template'], teeth_mesh.v), 0)
    flame_attributes['shapedirs'] = torch.cat((flame_attributes['shapedirs'], torch.zeros((teeth_mesh.v.shape[0],3,400)).to(device)), 0)
    flame_attributes['posedirs'] = torch.cat((flame_attributes['posedirs'], torch.zeros((36, teeth_mesh.v.shape[0]*3)).to(device)), 1)
    flame_attributes['J_regressor'] = torch.cat((flame_attributes['J_regressor'], torch.zeros((5, teeth_mesh.v.shape[0])).to(device)), 1)

    body_model.set_params(flame_attributes)

    # combine texture
    # tex = np.zeros((2048,2048,3), dtype=np.uint8)
    img = cv2.imread(texture_path) 
    tex = cv2.resize(img, (2048, 2048)) 
    img = cv2.imread(f'{prj_dir}/data/teeth/teeth_color_map.png')  
    tex[1024:, :1024,] = img
    img = cv2.imread(f'{prj_dir}/data/teeth/mouth_color_map.png')
    tex[1024:, 1024:,] = img

    # load texture and uv   
    mesh.f = torch.cat((mesh.f, teeth_mesh.f+(mesh.v.shape[0])), 0)
    mesh.v = torch.cat((mesh.v, teeth_mesh.v), 0)
    mesh.vn = torch.cat((mesh.vn, teeth_mesh.vn), 0)
    mesh.fn = torch.cat((mesh.fn, teeth_mesh.fn), 0)
    mesh.ft = torch.cat((mesh.ft, teeth_mesh.ft+(mesh.vt.shape[0])), 0)
    mesh.vt = torch.cat((mesh.vt, teeth_mesh.vt), 0)
    mesh.vt[-23084:,] = mesh.vt[-23084:,]/2.0
    mesh.vt[-23084:,1] = mesh.vt[-23084:,1] + 0.5
    mesh.vt[-27392:-23084,] = mesh.vt[-27392:-23084,]/2.0
    mesh.vt[-27392:-23084,:] = mesh.vt[-27392:-23084,:] + 0.5
    albedo = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)

    # Image.fromarray(albedo).save(f'{subject}-albedo.png')
    
    albedo = albedo.astype(np.float32) / 255
    mesh.albedo = torch.tensor(albedo, dtype=torch.float32, device=device)
    return mesh 
 

# python -m scripts.paper.pipeline
# seq 0 4 | xargs -P 5 -I {} bash -c "CUDA_VISIBLE_DEVICES={} python animation.py"
if __name__ == "__main__":   
    import argparse
    from omegaconf import OmegaConf 

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Base data directory.')
    parser.add_argument("--subject", type=str, default='', help="") 
    parser.add_argument("--driven", type=str, default='d0', help="")
    parser.add_argument("--exp", type=str, default='newest', help="") 
    parser.add_argument('--teeth_lowt', type=lambda s: [float(x) for x in s.split(',')])
    parser.add_argument('--teeth_uppt', type=lambda s: [float(x) for x in s.split(',')])
    args = parser.parse_args()

    device = 'cuda'
    exp = args.exp   
    driven = args.driven  
    subject = args.subject
    data_dir = args.data_dir
    teeth_lowt = args.teeth_lowt
    teeth_uppt = args.teeth_uppt 
    add_teeth = True    
    suffix = '-teeth' if add_teeth else '' 
     
    
    subject_dir = os.path.join(data_dir, subject, '6-views')  
    flame_path = f'{subject_dir}/deep3dface/flame.pth'
    rigging_path = f'{subject_dir}/{exp}/flame_attributes.pth'
    mesh_path = f'{subject_dir}/{exp}/recon/recon_textured.obj'
    texture_path = f'{subject_dir}/{exp}/recon/recon_textured_albedo.png'
    
    prj_dir = os.path.join(data_dir, '../..')
     
    renderer = Renderer()  
    body_model = create_flame_model(
        model_path=f"{prj_dir}/data/flame/flame2020.pkl", create_segms=True, add_teeth=False, device=device
    )  

    if not os.path.exists(texture_path):
        print(subject, 'no texture', texture_path) 
        exit() 

    # load flame params
    flame_params = torch.load(flame_path)  
    flame_params = {k: v.to(device) for k, v in flame_params.items()}
    
    # deform do you know how to teeth
    flame_attributes = torch.load(rigging_path)
    flame_attributes = {k: v.to(device) for k, v in flame_attributes.items()}
    body_model.set_params(flame_attributes) 
    
    mesh = Mesh.load_obj(mesh_path, device=device)
    if add_teeth: # TODO: mv this to the main.py  
        add_the_teeth(flame_params['betas'], teeth_lowt, teeth_uppt)             

    # mesh.v = vertices[0].detach()
    mesh.auto_normal()
    os.makedirs(f'{subject_dir}/{exp}/recon-teeth', exist_ok=True) 
    mesh.write(f'{subject_dir}/{exp}/recon-teeth/mesh.obj') 

    albedo_apha_path = f'{subject_dir}/{exp}/recon-teeth/mesh_albedo_alpha.png'
    img = np.array(Image.open(os.path.join(subject_dir, exp, 'recon-teeth/mesh_albedo.png')))
    msk = np.array(Image.open('./resource/inner_mouth_mask.png').resize((img.shape[1], img.shape[0])))
    img = np.concatenate([img, 255 - msk[..., None]], axis=2)
    Image.fromarray(img).save(albedo_apha_path)

    driven_path = f'{prj_dir}/data/driven/{driven}.npz' 
    # load driven motion sequence
    motions = np.load(driven_path)  
    focal_length = motions['focal_length'][0]
    motions = { 
        'betas': flame_params['betas'],  
        'expression': motions['expr'], 
        'jaw_pose': motions['jaw_pose'],
        'neck_pose': motions['neck_pose'], 
        'global_orient': motions['rotation'],
        'leye_pose': motions['eyes_pose'][:, :3],
        'reye_pose': motions['eyes_pose'][:, 3:], 
        # 'transl': motions['translation'],
    }   
    
    motions = {k: torch.tensor(v).float().to(device) for k, v in motions.items()}

    with torch.no_grad():
        vertices = body_model(**motions).vertices.float()    
        vertices = (vertices + flame_params['transl']) * flame_params['scale'] * 0.8
        vertices[:, : 2] -= vertices[:, : 2].mean()
        # vertices = vertices * flame_params['scale']
        
    os.makedirs(f'{subject_dir}/{exp}/animation', exist_ok=True)
    np.save(f'{subject_dir}/{exp}/animation/{driven}-seq.npy', vertices.cpu().numpy())
