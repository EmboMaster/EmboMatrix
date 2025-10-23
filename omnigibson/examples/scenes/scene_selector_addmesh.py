import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_available_g_scenes, get_available_og_scenes
from omnigibson.utils.ui_utils import choose_from_options
import omni
from typing import List
import os

# Configure macros for maximum performance
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False

def apply_material_to_mesh_from_existing(prim, stage):
    from pxr import UsdGeom, UsdShade, Sdf, Usd
    """
    Retrieve the material and texture from an existing prim and apply them to the mesh.
    """
    # Access the material binding API of the prim
    material_binding_api = UsdShade.MaterialBindingAPI(prim)
    
    # Check if the prim has a material binding
    material_binding = material_binding_api.GetDirectBinding()
    
    if material_binding:
        material_path = material_binding.GetMaterialPath()  # Corrected method to get the material path
        print(f"Material found on {prim.GetPath()}: {material_path}")

        # Now, ensure the material_path corresponds to a valid prim in the stage
        material_prim = stage.GetPrimAtPath(material_path)
        if material_prim:
            # Now create a Shader from the material's prim
            shader = UsdShade.Shader(material_prim)
            print(f"Found shader for material: {shader.GetPath()}")

            # Look for texture (UsdUVTexture) in the shader inputs
            for input in shader.GetInputs():
                if input.GetName() == "file":
                    texture_path = input.Get()
                    if texture_path:
                        print(f"Texture found: {texture_path}")

                        # Ensure the texture path is absolute, we can use an absolute path
                        if not os.path.isabs(texture_path):
                            texture_path = os.path.join("/home/embodied/Downloads/textures", texture_path)  # Replace with your absolute path
                        input.Set(texture_path)
                        print(f"Texture path updated to absolute: {texture_path}")
        else:
            print(f"Material prim not found at path: {material_path}")
    else:
        print(f"No material found on {prim.GetPath()}")



# def apply_material_to_mesh(prim, stage, texture_folder_path):
#     from pxr import UsdGeom, UsdShade, Sdf, Usd
#     """
#     Apply a simple material to the mesh and bind it to the prim, ensuring texture path is absolute.
#     """
#     material_name = "defaultMaterial"
#     material = UsdShade.Material.Define(stage, "/Materials/" + material_name)
    
#     # Create a shader
#     shader = UsdShade.Shader.Define(stage, "/Materials/" + material_name + "/Shader")
#     shader.CreateIdAttr("UsdPreviewSurface")

#     # Setup the material properties (color, roughness, etc.)
#     shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.8, 0.8, 0.8))  # Light gray color
#     shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)


#     # Now handle texture (albedo map) as absolute path
#     texture_path = os.path.join(texture_folder_path, "albedo_map.png")
#     if os.path.exists(texture_path):
#         texture = UsdShade.Shader.Define(stage, "/Materials/" + material_name + "/Texture")
#         texture.CreateIdAttr("UsdUVTexture")
#         texture.CreateInput("file", Sdf.ValueTypeNames.String).Set(texture_path)

#         # Connect the texture to the shader's diffuseColor
#         shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((1.0, 1.0, 1.0))  # Set default color white
#         shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.String).Set(texture_path)

#     # Bind the material to the mesh
#     material.CreateSurfaceOutput().ConnectToSource(shader, "surface")
#     UsdShade.MaterialBindingAPI(prim).Bind(material)


def main(random_selection=False, headless=False, short_exec=False):
    """
    Prompts the user to select any available interactive scene and loads a turtlebot into it.
    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Choose the scene type to load
    scene_options = {
        "InteractiveTraversableScene": "Procedurally generated scene with fully interactive objects",
    }
    scene_type = choose_from_options(options=scene_options, name="scene type", random_selection=random_selection)

    # Choose the scene model to load
    scenes = get_available_og_scenes() if scene_type == "InteractiveTraversableScene" else get_available_g_scenes()
    scene_model = choose_from_options(options=scenes, name="scene model", random_selection=random_selection)

    cfg = {
        "scene": {
            "type": scene_type,
            "scene_model": scene_model,
        },
    }

    # If the scene type is interactive, also check if we want to quick load or full load the scene
    if scene_type == "InteractiveTraversableScene":
        load_options = {
            "Quick": "Only load the building assets (i.e.: the floors, walls, ceilings)",
            "Full": "Load all interactive objects in the scene",
        }
        load_mode = choose_from_options(options=load_options, name="load mode", random_selection=random_selection)
        if load_mode == "Quick":
            cfg["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]

    # Load the environment
    env = og.Environment(configs=cfg)
    stage = omni.usd.get_context().get_stage()

    # Texture folder path (absolute path)
    texture_folder_path = "/data/zxlei/embodied/embodied-bench/"  # Update with the correct absolute path

    # Iterate through all prims and apply material to meshes
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh":
            print(f"Found mesh prim: {prim.GetPath()}")
            apply_material_to_mesh_from_existing(prim, stage)

    # Export the stage to a USD file
    stage.Export("/home/embodied/Downloads/usd/test_with_materials_and_textures.usd")

    # Allow user to move camera more easily
    if not gm.HEADLESS:
        og.sim.enable_viewer_camera_teleoperation()

    # Run a simple loop and reset periodically
    max_iterations = 10 if not short_exec else 1
    for j in range(max_iterations):
        og.log.info("Resetting environment")
        env.reset()
        for i in range(10):
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)

    # Always close the environment at the end
    og.sim.save('test_with_materials_and_textures.usd')
    og.clear()

if __name__ == "__main__":
    main()
