import os
from models.inference_model import InferenceModel
import config
import utils.utils as utils
import torch
import PIL.Image as Image
import torchvision.transforms as transforms

import runway
from runway.data_types import *


@runway.setup(options={"style_dir" : file(is_directory=True), "network_type" : category(description="Select network architecture as per checkpoint name", choices=["normal", "dual"], default="normal")})
def setup(opts):

    ckpt = opts["style_dir"]
    model_path = ckpt + "/" + "model_dir/dynamic_net.pth"
    set_net_version = opts["network_type"]
    config_path = ckpt + "/" + "config.txt"
    print(model_path)
    opt = config.get_configurations()

    dynamic_model = InferenceModel(opt, set_net_version=set_net_version)
    dynamic_model.load_network(model_path)
    
    return {"dynamic_model" : dynamic_model,
            "network_type" : set_net_version }

command_inputs = {
"input_image" : image, 
"alpha_0_normal" : number(description="alpha values for NORMAL", min=0, max=1, step=0.1, default=0),
"alpha_0_dual" : number(description="alpha values for DUAL", min=-1, max=-0.1, step=0.1, default=-1)
}
command_outputs = {"output_image" : image}

@runway.command("stylize_image", inputs=command_inputs, outputs=command_outputs, description="Dynamically stylize the Image")
def stylize_image(model, inputs):

    to_tensor = transforms.ToTensor()
    to_pil_image = transforms.ToPILImage()
    # ------------------------ #
    im = inputs["input_image"]
    input_tensor = to_tensor(im).to(model["dynamic_model"].device)
    input_tensor = model["dynamic_model"].normalize(input_tensor)
    input_tensor = input_tensor.expand(1, -1, -1, -1)
    
    alpha0_normal = inputs["alpha_0_normal"]
    alpha0_dual = inputs["alpha_0_dual"]
    
    alpha_0 = alpha0_normal


    if model["network_type"] == "dual":
        alpha_0 = alpha0_dual + alpha0_normal

    output_tensor = model["dynamic_model"].forward_and_recover(input_tensor.requires_grad_(False), alpha_0=alpha_0, alpha_1=None, alpha_2=None)
    output_image = to_pil_image(output_tensor.clamp(min=0.0, max=1).cpu().squeeze(dim=0))

    return {"output_image" : output_image}

if __name__ == "__main__":
    runway.run(model_options={"style_dir" : "trained_nets/colors_to_waterfall_normal", "network_type" : "normal"})