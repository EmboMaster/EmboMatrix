import os

def update_texture_paths(usd_file_path, base_dir):
    # 加载USD文件
    from pxr import Usd, UsdShade
    stage = Usd.Stage.Open(usd_file_path)

    if not stage:
        print(f"无法打开USD文件: {usd_file_path}")
        return

    # 遍历所有Usd.Prim节点
    for prim in stage.TraverseAll():
        # 查找Shader节点，通常材质的纹理通过shader链接
        # if prim.HasAPI(UsdShade.Shader):
        if "Shader" in str(prim):
            shader = UsdShade.Shader(prim)

            # 获取shader的所有输入属性
            for input_attr in shader.GetInputs():
                # 打印每个输入的名字
                input_name = input_attr.GetBaseName()

                # 检查输入属性的值，看看它是否是纹理路径
                file_path = input_attr.GetAttr().Get()
                print(f"输入属性: {input_name}, 值: {file_path}")
                # if isinstance(file_path, str):
                # if file_path has attribute "path"
                if hasattr(file_path, "path"):
                    file_path_str = file_path.path
                    # 如果路径是相对路径，替换为绝对路径
                    if not os.path.isabs(file_path_str):
                        absolute_path = os.path.join(base_dir, file_path_str)
                        input_attr.GetAttr().Set(absolute_path)
                        print(f"已更新路径: {file_path} -> {absolute_path}")
                elif isinstance(file_path, Usd.Attribute):
                    print(f"属性 {input_name} 不是纹理路径")
                    
    stage.GetRootLayer().Save()

# 使用示例
usd_file_path = "/home/embodied/Downloads/usd/test.usd"  # 替换为你的USD文件路径
base_dir = "/data/zxlei/embodied/embodied-bench/"  # 替换为贴图文件所在的绝对路径
update_texture_paths(usd_file_path, base_dir)
