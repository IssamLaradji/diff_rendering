import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as AG
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import helpers
from torchvision.utils import make_grid
def plot():
    # testing cosine-weighted sphere projection
    nor = torch.tensor([[1.,1.,0.]])
    nor = torch.repeat_interleave(nor, repeats=1000, dim=0)
    rd = helpers.sampleCosineWeightedHemisphere(nor)

    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(rd[:,0],rd[:,2],rd[:,1])
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')

    ax = fig.add_subplot(122)
    ax.scatter(rd[:,0],rd[:,1])
    fig.savefig('tmp.png')



def get_target_scene():
    

    
    

    scene_map_original = [
    (left_wall_color, helpers.udBox, short_block_pos, [short_box_arg_1])
    ]

    original_img = helpers.render_fn(ro, rd, scene_map_original, num_samples=1)
    print(original_img.requires_grad) # just a test to be sure that the original image has no gradients

    return original_img



def optimize(target_scene):
    scale = 100
    short_block_pos_x = torch.tensor([0.2 * scale], requires_grad=True)
    short_block_pos_y = torch.tensor([.4 * scale])
    short_block_pos_z = torch.tensor([1.7 * scale])
    short_block_pos_new = torch.cat([short_block_pos_x, short_block_pos_y, short_block_pos_z], dim=0)
    scene_map_new = [
    (left_wall_color, helpers.udBox, short_block_pos_new, [short_box_arg_1])
    ]

    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 5e-1
    optimizer = torch.optim.Adam([short_block_pos_x], lr=learning_rate)

    target_img = target_scene.detach() # redundant call since original_img shouldnt have a gradient
    img = save_img('results/target.png', target_img)
    img_list = [img.detach().permute(2,1,0)]
    for t in range(300):
        #update scene map
        short_block_pos_new = torch.cat([short_block_pos_x, short_block_pos_y, short_block_pos_z], dim=0)
        scene_map_new = [
        (left_wall_color, helpers.udBox, short_block_pos_new, [short_box_arg_1])
        ]
        # Forward pass: compute predicted y by passing x to the model.
        predicted_img = helpers.render_fn(ro, rd, scene_map_new, num_samples=1)

        # Compute and print loss.
        loss = loss_fn(predicted_img, target_img)
        if t % 10 == 0:
            img = save_img('results/pred_%s.png' % (t), predicted_img)
            img_list += [img.detach().permute(2,1,0)]
            plt.imshow(make_grid(img_list).permute(1,2,0))
            plt.grid('off')
            plt.savefig('results/grid.png')
            print(t, loss.item())
            print(short_block_pos_x)

        #print(loss.item()) - for debugging purposes
        optimizer.zero_grad()
        loss.backward()
        #print("Autograd gradient is {0}".format(str(short_block_pos_x.grad[0]))) - for debugging purposes

        optimizer.step()
        short_block_pos_x.detach_()
        short_block_pos_x.requires_grad_()
    img_list
    make_grid()
    print(short_block_pos_x)

def save_img(fname, scene):
    img = scene
    H, W = 100, 100
    img = img.view(H, W, 3)
    img = torch.flip(img, (0,1))
    print('done')

    final_img = img.detach()
    plt.imshow(final_img, interpolation='none',vmin = 0, vmax = 1)
    plt.grid('off')
    plt.savefig(fname)
    return img

def main():
    scene = get_target_scene()
    save_img('target_scene.png', scene)

    optimize(scene)

if __name__ == "__main__":
    scale = 100.0 # THIS PARAMETER IS THE SCENE SCALE, it is arbitrary but scale might affect accuracy/precision
    left_wall_color = torch.tensor([.9165, .0833, .093])
    short_block_pos = torch.tensor([[.65,0.4,1.7]]) * scale
    short_box_arg_1 = torch.tensor([[0.6,.01,0.6]]) * scale

    # light parameters
    LIGHT_POWER = torch.tensor([25, 25, 25]) * scale * scale# Watts
    LIGHT_AREA=(1.*scale)*(1.*scale)
    emitted_radiance = LIGHT_POWER / (math.pi * LIGHT_AREA)
    nor_light = torch.tensor([[0.,-1.,0.]])

    
    

    
    H = 100
    W = 100

    eye_pos = torch.tensor([0. * scale, 2. * scale, -3.5 * scale])
    # you cant have look_pos depend on eye_pos, because then gradients depend on it and this doesnt make sense, check if look_pos needs grad
    look_dir = torch.tensor([0., 0., 1.])  
    up_vec = torch.tensor([0.,1.,0.]) # up axis of world

    ro, rd = helpers.buildRays(eye_pos, look_dir, up_vec, H=H, W=W)


    main()