from . import scenes
import copy
import pylab as plt 
import torch
from torchvision.utils import make_grid

def get_model(model_dict, source_scene, exp_dict):
    # -- define renderer
    if model_dict['name'] == 'basic_renderer':
        model = BasicRenderer(source_scene, exp_dict)

    # -- define optimizer
    if model_dict['opt'] == 'adam':
        model.opt = torch.optim.Adam([source_scene.short_block_pos], 
                                      lr=model_dict['lr'])

    return model

class BasicRenderer:
    def __init__(self, source_scene, exp_dict):
        loss_fn = torch.nn.MSELoss(reduction='sum')
        self.source_scene = source_scene
        self.source_start_scene = copy.deepcopy(source_scene)
        self.iteration = 0

    def train_on_batch(self, target_scene):
        # get scene images
        target_image = target_scene.get_scene_image()
        pred_image = self.source_scene.get_scene_image()

        # fompute and print loss.
        loss = torch.nn.MSELoss(reduction='sum')(pred_image, target_image)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.iteration += 1

        return {'loss':float(loss)}

    def vis_on_batch(self, target_scene, fname):
        # get scene images
        H, W = 100, 100
        target_image = target_scene.get_scene_image().detach().view(H, W, 3).permute(2,0,1)
        pred_image = self.source_scene.get_scene_image().detach().view(H, W, 3).permute(2,0,1)
        start_image = self.source_start_scene.get_scene_image().detach().view(H, W, 3).permute(2,0,1)

        plt.imshow(make_grid([start_image, 
                    pred_image,
                    target_image]).permute(1,2,0))
        plt.title('left: initial scene, center: prediction, right: target, (iter: %d)' % self.iteration)
        plt.savefig(fname)
        