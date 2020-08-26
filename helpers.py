import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as AG
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class maxVector3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2):
        ctx.save_for_backward(input1, input2)
        res = torch.max(input1, input2)
        return res

    # implement a hacked derivative of the max function
    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors

        smooth_step_wnd = 1.0
        threshold = input2[:,0,None] - input1[:,0,None] 
        tmp = torch.abs(threshold)
        condition0 = tmp <= torch.tensor([smooth_step_wnd])
        threshold = input2[:,1,None] - input1[:,1,None] 
        tmp = torch.abs(threshold)
        condition1 = tmp <= torch.tensor([smooth_step_wnd])
        threshold = input2[:,2,None] - input1[:,2,None] 
        tmp = torch.abs(threshold)
        condition2 = tmp <= torch.tensor([smooth_step_wnd])

        dout0_dx0 = torch.where(condition0, torch.tensor([1.0]), torch.tensor([0.0])) * (input1[:,0,None] - input2[:,0,None]) + \
                torch.where(input1[:,0,None] - input2[:,0,None] >= 0, torch.tensor([1.0]), torch.tensor([0.0]))

        dout1_dx1 = torch.where(condition1, torch.tensor([1.0]), torch.tensor([0.0])) * (input1[:,1,None] - input2[:,1,None]) + \
                torch.where(input1[:,1,None] - input2[:,1,None] >= 0, torch.tensor([1.0]), torch.tensor([0.0]))

        dout2_dx2 = torch.where(condition2, torch.tensor([1.0]), torch.tensor([0.0])) * (input1[:,2,None] - input2[:,2,None]) + \
                torch.where(input1[:,2,None] - input2[:,2,None] >= 0, torch.tensor([1.0]), torch.tensor([0.0]))
        
        dout0_dy0 = torch.where(condition0, torch.tensor([1.0]), torch.tensor([0.0])) * (input2[:,0,None] - input1[:,0,None]) + \
                torch.where(input2[:,0,None] - input1[:,0,None] >= 0, torch.tensor([1.0]), torch.tensor([0.0]))

        dout1_dy1 = torch.where(condition1, torch.tensor([1.0]), torch.tensor([0.0])) * (input2[:,1,None] - input1[:,1,None]) + \
                torch.where(input2[:,1,None] - input1[:,1,None] >= 0, torch.tensor([1.0]), torch.tensor([0.0]))

        dout2_dy2 = torch.where(condition2, torch.tensor([1.0]), torch.tensor([0.0])) * (input2[:,2,None] - input1[:,2,None]) + \
                torch.where(input2[:,2,None] - input1[:,2,None] >= 0, torch.tensor([1.0]), torch.tensor([0.0]))

        dout_dx = torch.cat([dout0_dx0, dout1_dx1, dout2_dx2], dim=1) * grad_output
        dout_dy = torch.cat([dout0_dy0, dout1_dy1, dout2_dy2], dim=1) * grad_output

        return dout_dx, dout_dy

#https://github.com/pytorch/pytorch/issues/2421 - interesting read on issues of gradient of norm
def length(p):
  return torch.norm(p, p=None, dim=1, keepdim=True)
  
def normalize(p):
  return p/length(p)

def sdSphere(p,radius):
  return length(p) - radius

def udBox(p, args):
  b = args[0]
  return length(maxVector3.apply(torch.abs(p)-b, torch.tensor([0., 0., 0.]).view(1,3)))

def rotateX(p,a):
  c = torch.cos(a); s = torch.sin(a);
  px,py,pz=p[:,0,None],p[:,1,None],p[:,2,None]
  return torch.cat((px,c*py-s*pz,s*py+c*pz), 1)

def rotateY(p,a):
  c = torch.cos(a); s = torch.sin(a);
  px,py,pz=p[:,0,None],p[:,1,None],p[:,2,None]
  return torch.cat((c*px+s*pz,py,-s*px+c*pz), 1)

def rotateZ(p,a):
  c = torch.cos(a); s = torch.sin(a);
  px,py,pz=p[:,0,None],p[:,1,None],p[:,2,None]
  return torch.cat((c*px-s*py,s*px+c*py,pz), 1)

# chooses the smallest vale, used for the union of two SDFs
class opUCustom(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2):
        ctx.save_for_backward(input1, input2)
        threshold = input1[:,3,None] - input2[:,3,None] # x - y
        condition = threshold <= 0.0
        res = torch.where(condition, input1, input2) 
        return res

    # we get our input in the form [x,u] and [y,z] and return 2 outputs each of which has a derivative w.r.t the 4 inputs
    # noting that we check y>x to decide which input to take
    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        # x and y are the distance values for the SDF
        x = input1[:,3,None]
        y = input2[:,3,None]
        # u and z are the color values of the respective materials of x and y
        u = torch.cat([input1[:,0,None], input1[:,1,None], input1[:,2,None]], dim=1)
        z = torch.cat([input2[:,0,None], input2[:,1,None], input2[:,2,None]], dim=1)

        threshold = x - y 
        tmp = torch.abs(threshold)
        # so it would appear that for a choose smaller function, having 2 window sizes is required
        # as the 2 different uses of delta function in the derivatives work with different windows
        condition = tmp <= torch.tensor([0.5]) # the parameter in here can be adjusted to change the smoothstep window, was 0.5
        # i think the issue is that the difference in signed distance between 2 objects isnt really a small number
        # we rarely find ourselves in a situation where the 2 distances are close and a small change in SIGNED DISTANCE will cause a difference
        # note that a small change in a positions object can cause a large change in signed distances, which would essentially allow us to jump over
        # the decision boundary/discontinuity. Basically our window is not large enough to capture the delta function and small changes in an objects
        # position can cause large changes in signed distance causing us to cross over the delta function. Without sampling the delta function
        # we have no gradient. 
        condition2 = tmp <= torch.tensor([50.0]) # the parameter in here can be adjusted to change the smoothstep window, was 1.0 for camera optimization, used 50.0 for box pos optimization
        # IT WORKED! I JUST NEEDED A REALLY LARGE WINDOW TO SAMPLE THE TRANSITION BETWEEN HEAVISIDES WHEN APPLYING THIS TO SIGNED DISTANCES

        # gradients of 1st output w.r.t inputs: 
        # https://www.wolframalpha.com/input/?i=derivative+heaviside%28x-y%29*y+%2B+heaviside%28y-x%29*x
        da_dx = torch.where(condition, torch.tensor([1.0]), torch.tensor([0.0])) * (y - x) + \
                torch.where(y >= x, torch.tensor([1.0]), torch.tensor([0.0]))
        da_du = torch.repeat_interleave(torch.tensor([0.0]).view(1,1), input1.size()[0], dim=0)
        da_du = torch.cat([da_du, da_du, da_du], dim=1)
        dout4_din1 = torch.cat([da_du, da_dx], dim=1)

        da_dy = torch.where(condition, torch.tensor([1.0]), torch.tensor([0.0])) * (x - y) + \
                torch.where(x >= y, torch.tensor([1.0]), torch.tensor([0.0]))
        da_dz = torch.repeat_interleave(torch.tensor([0.0]).view(1,1), input1.size()[0], dim=0)
        da_dz = torch.cat([da_dz, da_dz, da_dz], dim=1)
        dout4_din2 = torch.cat([da_dz, da_dy], dim=1)

        # gradients of 2nd output w.r.t inputs:
        # https://www.wolframalpha.com/input/?i=derivative+heaviside%28x-y%29*z+%2B+heaviside%28y-x%29*u
        db_dy = torch.where(condition2, torch.tensor([1.0]), torch.tensor([0.0])) * (u - z) # this has 3 columns
        db_dx = -db_dy       
        db_du = torch.where(x <= y, torch.tensor([1.0]), torch.tensor([0.0])) # this is what torch.where does
        db_dz = torch.where(x >= y, torch.tensor([1.0]), torch.tensor([0.0])) # this is what torch.where does

        dout1_din1 = torch.cat([db_du, torch.zeros_like(db_du), torch.zeros_like(db_du), db_dx[:,0,None]], dim=1)
        dout2_din1 = torch.cat([torch.zeros_like(db_du), db_du, torch.zeros_like(db_du), db_dx[:,1,None]], dim=1)
        dout3_din1 = torch.cat([torch.zeros_like(db_du), torch.zeros_like(db_du), db_du, db_dx[:,2,None]], dim=1)

        dout1_din2 = torch.cat([db_dz, torch.zeros_like(db_dz), torch.zeros_like(db_dz), db_dy[:,0,None]], dim=1)
        dout2_din2 = torch.cat([torch.zeros_like(db_dz), db_dz, torch.zeros_like(db_dz), db_dy[:,1,None]], dim=1)
        dout3_din2 = torch.cat([torch.zeros_like(db_dz), torch.zeros_like(db_dz), db_dz, db_dy[:,2,None]], dim=1)

        # chain rule jacobian-vector product we have gradients w.r.t our output and so we want gradients w.r.t our input
        grad1 = grad_output[:,0,None] * dout1_din1 + grad_output[:,1,None] * dout2_din1 + \
                grad_output[:,2,None] * dout3_din1 + grad_output[:,3,None] * dout4_din1

        grad2 = grad_output[:,0,None] * dout1_din2 + grad_output[:,1,None] * dout2_din2 + \
                grad_output[:,2,None] * dout3_din2 + grad_output[:,3,None] * dout4_din2

        return grad1, grad2



def clamp01(v):
  return torch.min(torch.max(v,torch.tensor([0.])),torch.tensor([1.]))

def relu(a):
  return torch.max(a,torch.tensor([0.]))

def dot(a,b):
  return torch.sum(a*b, dim=1, keepdim=True)

def sampleCosineWeightedHemisphere(n):
  u1 = torch.rand(n.size()[0],1)
  u2 = torch.rand(n.size()[0],1)
  tmp = torch.tensor([[0.,1.,1.]])  # if n aligns in the direction of this vector this sampling scheme will break
  tmp = torch.repeat_interleave(tmp, repeats=n.size()[0], dim=0)
  uu = normalize(torch.cross(n, tmp))
  vv = torch.cross(uu,n)
  ra = torch.sqrt(u2)
  rx = ra*torch.cos(2*math.pi*u1)
  ry = ra*torch.sin(2*math.pi*u1)
  rz = torch.sqrt(1.-u2)
  rr = rx*uu+ry*vv+rz*n
  return normalize(rr)

# takes a tensor of distances and grows them into brdf, distance pairs
# note that dist is a repeating tensor, aka a single vector is repeated n time
def df(brdf, dist):
  tmp = torch.repeat_interleave(brdf.view(1,3), dist.size()[0], dim=0)
  return torch.cat((tmp, dist), dim=1)


def sdScene(p, scene_map):
  scale = 100.0 # THIS PARAMETER IS THE SCENE SCALE, it is arbitrary but scale might affect accuracy/precision
  white_wall_color = torch.tensor([255., 239., 196.]) / 255
  left_wall_color = torch.tensor([.9165, .0833, .093])
  right_wall_color = torch.tensor([.117, .4125, .115]) * 1.5
  light_diffuse_color = torch.tensor([0.2, 0.2, 0.2])

  px,py,pz=p[:,0,None],p[:,1,None],p[:,2,None]
  # floor - is a plane at y = 0
  obj_floor = df(white_wall_color, py)
  res = obj_floor  
  # ceiling
  obj_ceil = df(white_wall_color, (4.*scale)-py)
  res = opUCustom.apply(res,obj_ceil)
  # backwall
  obj_bwall = df(white_wall_color, (4.*scale)-pz)
  res = opUCustom.apply(res,obj_bwall)
  
  # leftwall
  obj_lwall = df(left_wall_color, px-(-2.*scale))
  res = opUCustom.apply(res,obj_lwall)
  
  # rightwall
  obj_rwall = df(right_wall_color, (2.*scale)-px)
  res = opUCustom.apply(res,obj_rwall)
  
  # light
  obj_light = df(light_diffuse_color, udBox(p - torch.tensor([[0.,3.9,2.]])*scale, torch.tensor([[.5,.01,.5]])*scale))
  res = opUCustom.apply(res,obj_light)
  
  # each element in a scene map has a single diffuse brdf, a SDF, a center position for the SDF, and arguments required for the SDF
  for brdf, func, pos, args in scene_map:
    p2 = p - pos
    d = func(p2, args) # this should have a non-zero gradient w.r.t p2 and by chain rule, w.r.t pos
    tmp = df(brdf, d) # whether this brdf is chosen depends on the signed distance d
    res = opUCustom.apply(res, tmp) # opU will cause the resulting brdf to have a gradient depending on d by chain rule
  
  return res

# get nearest signed distance to point p and return only that
def dist(p, scene_map):
  res = sdScene(p, scene_map)
  return res[:,3,None] # I believe the signed distance is in element 3 not 1




# returns a tensor of brdf, distance pairs, one such pair for each ray
def raymarchScene(ray_orig, ray_dir, scene_map, max_steps=50):
  scale = 100
  black = torch.tensor([0., 0., 0.])
  FAR_CLIP=20.0*scale

  t = torch.zeros(ray_orig.size()[0], 1) # Nx1
  nearest = None
  for i in range(max_steps):
    # derivative of p w.r.t ray_orig is 1, thus doesnt depend on t in any way and inplace operation of t doesnt cause issues
    # however attempting to get the derivative of p w.r.t ray_dir would break at the moment due to modifying t inplace 
    p = ray_orig + t*ray_dir 
    nearest = sdScene(p, scene_map)
    # this is the inplace operation causing issues with gradients, however if gradient doesnt depend on t then we have no issues
    t += nearest[:,3,None]  

  condition = t[:,0,None]<FAR_CLIP

  brdf = torch.cat([nearest[:,0,None], nearest[:,1,None], nearest[:,2,None]], dim=1)
  # if hit point is beyond the far clip, we render black, this likely causes an unhandeled discontinuity that I ignore for now
  obj_brdfs = torch.where(condition, brdf, black) 
  res = torch.cat((obj_brdfs, t), dim=1)
  return res

# Fundamentally p is not a leaf node, but we need the gradients w.r.t to it
# https://discuss.pytorch.org/t/how-do-i-calculate-the-gradients-of-a-non-leaf-variable-w-r-t-to-a-loss-function/5112
def calcNormalWithAutograd(p):
  # this whole function is a mess due to indirect lighting, more thought needs to be put into how the autograd graph is being built as requires_grad is contagious
  # note, everytime this function is called, the current p tensor has no gradients, calling backward on distances will accumulate gradients all the way
  # back to the original p matrix on the first recursive iteration (depth 0) and so that tensor will have garbage gradients but we no longer need those (we already used them)
  # instead we just care about the gradients in this current p tensor which started empty and now gets accumulated (retain_grad() keeps the gradient as we propogate past this
  # back to the original p tensor that we first set requires_grad=True on). This is why this function works, we backprop a bunch of garbage beyond the current p tensor
  # but it doesnt matter because we never used those gradients again (we only use gradients of the most recent p tensor)
  p.requires_grad_(True)
  p.retain_grad()
  #p.grad = None # whether this is on or off doesnt seem to make any difference
  distances = dist(p)
  distances.backward(torch.ones_like(distances), retain_graph=True)
  normals = p.grad
  normals = normalize(normals)
  return normals #gradient of the distance function with respect to parameter p

# I USE THIS INSTEAD AT THE MOMENT
def calcNormalWithFiniteDiff(p, scene_map):
  # derivative approximation via midpoint rule
  eps = 0.001
  dx=torch.tensor([eps,0,0])
  dy=torch.tensor([0,eps,0])
  dz=torch.tensor([0,0,eps])

  # extract just the distance component - can we not just call "dist" function instead here?
  nor = sdScene(p+dx, scene_map)[:,3,None] - sdScene(p-dx, scene_map)[:,3,None]
  nor = torch.cat([nor, sdScene(p+dy, scene_map)[:,3,None] - sdScene(p-dy, scene_map)[:,3,None]], dim=1)
  nor = torch.cat([nor, sdScene(p+dz, scene_map)[:,3,None] - sdScene(p-dz, scene_map)[:,3,None]], dim=1)
  return normalize(nor)

def sampleRecursiveRay(p, nor):
  ro2 = p + 0.001 * nor # bump along normal.
  rd2 = sampleCosineWeightedHemisphere(nor)
  return ro2, rd2


def trace(ray_orig, ray_dir, scene_map, depth):
  num_pixels = ray_orig.size()[0]
  radiance = torch.zeros((num_pixels, 3)) #Nx3
  intersection = raymarchScene(ray_orig, ray_dir, scene_map)  # ray_dir is Nx3
  
  # must be a cleaner way than doing the below to build the tensor
  intersection_brdfs = torch.cat([intersection[:,0,None], intersection[:,1,None]], dim=1)
  intersection_brdfs = torch.cat([intersection_brdfs, intersection[:,2,None]], dim=1)

  intersection_points = ray_orig + ((intersection[:,3,None]-0.0001) * ray_dir) # for stable normals with autodiff, we need to make sure we are not inside the surface

  # this is for debugging optimization, if you can't optimize using JUST the brdfs with no lighting, then there is a big issue
  # note that optimization is performing better at the moment when stopping just here, I have yet to figure out exactly why
  return intersection_brdfs 

  # https://discuss.pytorch.org/t/the-logical-or-and-not-function-about-pytorch/32718/5  GRADIENTS OF BOOLEAN OPERATORS
  # torch.logical_and doesnt play well with derivatives, this has yet to cause issues but might when I optimize light position.
  # Note comparison with "black" and "lightDiffuseColor" is not at all an elegant way to check this, but rather material IDs are preferred.
  # However using IDs was not causing complications for discontinuous parameter gradients. This has a lot to do with all the boolean operators
  # the original blog post used to select a brdf based on an ID. A refactor/rework should later be done to go back to using IDs. 
  did_intersect = torch.logical_and(intersection_brdfs != black, intersection_brdfs != lightDiffuseColor)  


  #normals = calcNormalWithAutograd(intersection_points)
  normals = calcNormalWithFiniteDiff(intersection_points, scene_map)

  if depth==0: # primary visibility of the light source, can only happen at depth 0
    li_e = emitted_radiance
    radiance = (intersection_brdfs == light_diffuse_color) * li_e


  #p_light_x = (torch.rand(num_pixels, 1) - 0.5) * scale # rand between -0.5 and 0.5
  p_light_x = (torch.ones((num_pixels, 1)) * 0.0) * scale # deterministically select 0 for debugging
  #p_light_z = (torch.rand(num_pixels, 1) + 1.5) * scale # rand between 1.5 and 2.5
  p_light_z = (torch.ones((num_pixels, 1)) * 2.0) * scale # deterministically select 2 for debugging
  p_light = torch.cat([p_light_x, torch.ones_like(p_light_x) * 3.9 * scale], dim=1)
  p_light = torch.cat([p_light, p_light_z], dim = 1)
  square_distance = torch.sum(torch.square(p_light-intersection_points), dim=1).view(num_pixels, 1)
  pdf_A = 1./LIGHT_AREA
  wi_light = normalize(p_light - intersection_points)

  # occlusion factor
  res2 = raymarchScene(intersection_points + (0.001 * normals), wi_light, scene_map)
  occ_query_brdfs = torch.cat([res2[:,0,None], res2[:,1,None]], dim=1)
  occ_query_brdfs = torch.cat([occ_query_brdfs, res2[:,2,None]], dim=1)
  vis = (occ_query_brdfs == light_diffuse_color)

  # brdf * emitted radiance * clamped cosine * 1/pdf * jacobian for measure from area to solid angle
  ndotl = (normals * wi_light).sum(-1, keepdim=True)
  lightndotl = (nor_light * -wi_light).sum(-1, keepdim=True)
  li_direct =  (intersection_brdfs * 
               relu(ndotl) *
               relu(lightndotl) * 
               emitted_radiance / (square_distance * pdf_A))

  #isect_and_vis = torch.logical_and(did_intersect, vis)
  isect_and_vis = did_intersect
  radiance = torch.where(isect_and_vis, li_direct, radiance)

  # indirect lighting
  # max_depth_const = 3
  # if depth < max_depth_const: 
  #   ro2, rd2 = sampleRecursiveRay(intersection_points, normals)
  #   li_indirect = trace(ro2,rd2,brdf_map,scene_map,depth+1)
  #   # doing cosweighted sampling cancels out the geom term
  #   radiance += torch.where(did_intersect, intersection_brdfs * li_indirect, torch.zeros_like(radiance))
  return radiance
  
def render_fn(ray_orig, ray_dir, scene_map, num_samples=1):
  # the below will make consecutive calls to render_fn deterministic but
  # the initial starting state could be different vs other test attempts
  # figure out how to "Reset" the state completely and then put that here instead
  # https://discuss.pytorch.org/t/cuda-rng-state-does-not-change-when-re-seeding-why-is-that/47917/5
  torch.manual_seed(1) # not sure this actually helps
  rng_state = torch.random.get_rng_state()
  res = trace(ray_orig, ray_dir, scene_map, 0)
  for i in range(2, num_samples+1):
    sample = trace(ray_orig, ray_dir, scene_map, 0)
    res = (res + sample)

  torch.random.set_rng_state(rng_state)
  return res/num_samples

def buildRays(eye, look_dir, up, d=2.2, H=100, W=100):
  xs=torch.linspace(0, 1, steps=W)
  ys=torch.linspace(0, 1, steps=H)

  vs,us = torch.meshgrid(ys,xs) #treat the first argument as a column and tile it as many times as there are elements in the second argument, vice versa for 2nd
  uv = torch.stack([us.flatten(),vs.flatten()], dim=1)

  # normalize pixel locations to -1,1
  p = torch.cat([-1.+2.*uv, torch.zeros((W*H,1))], dim=1)

  eye_batched = torch.repeat_interleave(eye.view(1,3), p.size()[0], dim=0)
  w = torch.repeat_interleave(look_dir.view(1,3), p.size()[0], dim=0)
  w = normalize(w)

  up_batched = torch.repeat_interleave(up.view(1,3), w.size()[0], dim=0)

  u = normalize(torch.cross(w,up_batched))
  v = normalize(torch.cross(u,w))

  rd = normalize(p[:,0,None]*u + p[:,1,None]*v + d*w)
  return eye_batched, rd