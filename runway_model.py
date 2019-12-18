import os
from models.inference_model import InferenceModel
import config
import utils.utils as utils
import torch
import PIL.Image as Image
import torchvision.transforms as transforms
import ast
import runway
from runway.data_types import *


@runway.setup(options={"style_dir" : file(is_directory=True)})
def setup(opts):

    ckpt = opts["style_dir"]
    model_path = ckpt + "/" + "model_dir/dynamic_net.pth"
    config_path = ckpt + "/" + "config.json"
    
    conf = json.load(open(config_path))
    set_net_version = conf["network_type"]
    opt = config.get_configurations()

    dynamic_model = InferenceModel(opt, set_net_version=set_net_version)
    dynamic_model.load_network(model_path)
    
    return {"dynamic_model" : dynamic_model,
            "network_type" : set_net_version }

command_inputs = {
"input_image" : image, 
"alpha_normal" : number(description="Alpha values required to tune the network", min=0, max=1, step=0.1, default=0)
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
    
    alpha0 = inputs["alpha_normal"]

    output_tensor = model["dynamic_model"].forward_and_recover(input_tensor.requires_grad_(False), alpha_0=alpha0, alpha_1=None, alpha_2=None)
    output_image = to_pil_image(output_tensor.clamp(min=0.0, max=1).cpu().squeeze(dim=0))

    return {"output_image" : output_image}

if __name__ == "__main__":
    runway.run(model_options={"style_dir" : "trained_nets/normal_mosaic_rain"})
