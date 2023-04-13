# High-Fidelity Text-Driven Body and Head Avatar Generation  
![crochet candle](asset/trump.gif)
This work aims to generate high quality 3D avatar with fine geometry and realistic texture from a SMPL body and a text prompt. 
 
## Difference from related work
* Magic3D (Stage 2):
    * we directly finetune the baked texture images, instead of the implicit function (MLP for albedo).
    * we directly learn the vertices offset based on SMPL, instead of tetrahedra mesh which requires large memory and good geometry reqularizer. 
* Text2Mesh 
    * We render normal image (image space) as latent code for stable-diffusion to generate high quality geometry.  
    * We mutual optimize head and the whole body 
* TEXTure
    * We use SDS so it's slower. 
* Latent-Paint
    * We also support finetuning mesh geometry.

## Interesting observations
* Directly optimizing the texture image (uv --> color) is harder than the MLP implicit function (xyz --> color). Reasons are likely how to force smoothness in texture parameter. Tricks like progressively increasing rendering resolution, and antialiasing are required to make it work.

  
## Install

```bash
git clone https://github.com/ashawkey/stable-dreammesh.git
cd stable-dreammesh

# requirements
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/nvdiffrast/
```

## Usage

```python
### basics
# training-geometry: load an mesh and optimize its geometry given a text prompt 
python main.py --stage geo --mesh data/mesh.obj --text "Donald Trump" --workspace experiments/trump --lock_tex --iters 5000
# training-texture: load an mesh and optimize its texture given a text prompt 
python main.py --stage tex --mesh experiments/trump/mesh/mesh.obj --text "Donald Trump" --workspace experiments/trump --lock_geo --iters 10000
```