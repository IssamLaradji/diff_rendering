import torch 
import math

from . import utils as ut

def get_scene(scene_dict, scale=100):
    if scene_dict['name'] == 'basic':
        scene = BasicScene()

    if scene_dict['name'] == 'moved_block':
        scene = BasicScene()
        scene.short_block_pos[0, 0] = torch.as_tensor(.2 * scale)
        scene.short_block_pos.requires_grad = True

    return scene


 
class BasicScene():
    def __init__(self, scale=100):
        self.scale = 100
        self.H = 100
        self.W = 100

        eye_pos = torch.tensor([0. , 2. , -3.5 ])* self.scale
        # you cant have look_pos depend on eye_pos, because then gradients depend on it and this doesnt make sense, check if look_pos needs grad
        look_dir = torch.tensor([0., 0., 1.])  
        up_vec = torch.tensor([0.,1.,0.]) # up axis of world

        self.ro, self.rd = ut.buildRays(eye_pos, 
                    look_dir, up_vec, H=self.H, W=self.W)

        self.left_wall_color = torch.tensor([.9165, .0833, .093])
        self.short_block_pos = torch.tensor([[.65,0.4,1.7]]) * self.scale
        self.short_box_arg_1 = torch.tensor([[0.6,.01,0.6]]) * self.scale


    def get_scene_image(self):
        scene_map_original = [(self.left_wall_color, ut.udBox, self.short_block_pos, [self.short_box_arg_1])]
        image = ut.render_fn(self.ro, self.rd, scene_map_original, num_samples=1)
        
        return image 


    