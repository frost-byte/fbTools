"""Generic/leftover utility nodes, extracted from extension.py (Plan 24).

SubdirLister (live), plus dead MultiLoraLoader and NodeInputSelect (both unregistered in
get_node_list() — carried along inertly, not fixed/deleted, per this session's established policy
for dead code discovered during a pure code-motion move).
"""
from __future__ import annotations

import folder_paths
from comfy_api.latest import io

from .shared import prefixed_node_id, get_subdirectories
from ..utils.util import get_workflow_all_nodes, listify_nodes_data, node_input_details, listify_node_inputs
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class SubdirLister(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=prefixed_node_id("SubdirLister"),
            display_name="Subdir Lister",
            category="🧊 frost-byte/File",
            inputs=[
                io.String.Input("directory_path", default="", tooltip="Path to the directory"),
            ],
            outputs=[
                io.Custom("DICT").Output("dir_dict", tooltip="Dictionary of subdirectory names to paths"),
                io.String.Output("dir_names", tooltip="List of subdirectory names"),
            ],
        )
    
    @classmethod
    def execute(cls, directory_path):

        subdir_dict = get_subdirectories(directory_path)

        return io.NodeOutput({
            "dir_dict": subdir_dict,
            "dir_names": list(subdir_dict.keys()) if subdir_dict else []
        })

class MultiLoraLoader(io.ComfyNode):
    """
    MultiLoraLoader: Load and apply multiple LoRAs to a diffusion model.
    All LoRA slots are always visible (up to 10), but only the number specified
    in 'num_loras_to_apply' will be processed. This preserves user selections
    when changing the count.
    """
    
    def __init__(self):
        super().__init__()
        self.loaded_loras = {}
    
    @classmethod
    def define_schema(cls):
        lora_files = folder_paths.get_filename_list("loras")
        
        # Create all 10 LoRA slots upfront (they'll always be visible)
        lora_inputs = [
            io.Model.Input("model", tooltip="The diffusion model that the LoRAs will be applied to"),
            io.Int.Input(
                "num_loras_to_apply",
                default=1,
                min=0,
                max=10,
                tooltip="How many of the LoRA slots below to actually apply (0-10)"
            ),
        ]
        
        # Add 10 LoRA slots
        for i in range(1, 11):
            lora_inputs.extend([
                io.Combo.Input(
                    f"lora_name_{i}",
                    options=lora_files,
                    optional=True,
                    tooltip=f"LoRA file #{i} (optional - leave empty to skip)"
                ),
                io.Float.Input(
                    f"strength_{i}",
                    default=1.0,
                    min=-100.0,
                    max=100.0,
                    step=0.01,
                    tooltip=f"Strength for LoRA #{i} (can be negative)"
                ),
            ])
        
        return io.Schema(
            node_id=prefixed_node_id("MultiLoraLoader"),
            display_name="Multi LoRA Loader",
            category="🧊 frost-byte/Loaders",
            inputs=lora_inputs,
            outputs=[
                io.Model.Output("model", tooltip="The modified diffusion model with LoRAs applied"),
            ],
        )
    
    @classmethod
    def execute(cls, model, num_loras_to_apply=1, **kwargs):
        """
        Load and apply multiple LoRAs to the model sequentially.
        
        Args:
            model: The input diffusion model
            num_loras_to_apply: How many LoRA slots to process (0-10)
            **kwargs: Contains lora_name_N and strength_N parameters
        
        Returns:
            NodeOutput with the modified model
        """
        import comfy.utils
        import comfy.sd
        
        # Ensure num_loras_to_apply is within bounds
        num_to_apply = max(0, min(10, int(num_loras_to_apply)))
        
        # Apply each LoRA sequentially up to the specified count
        current_model = model
        applied_count = 0
        
        for i in range(1, num_to_apply + 1):
            lora_name = kwargs.get(f"lora_name_{i}")
            strength = kwargs.get(f"strength_{i}", 1.0)
            
            # Skip if no LoRA name provided or strength is zero
            if not lora_name or strength == 0:
                logger.debug(f"Skipping LoRA slot {i}: lora_name='{lora_name}', strength={strength}")
                continue
            
            try:
                # Get the full path to the LoRA file
                lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                
                # Load the LoRA
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                
                # Apply the LoRA to the model (model only, no clip)
                current_model, _ = comfy.sd.load_lora_for_models(
                    current_model, None, lora, strength, 0
                )
                
                applied_count += 1
                logger.info(f"Applied LoRA {applied_count}/{num_to_apply}: {lora_name} (strength: {strength})")
            except Exception as e:
                logger.error(f"Failed to apply LoRA {i} ({lora_name}): {e}")
        
        if applied_count == 0:
            logger.warning("No LoRAs were applied")
        
        return io.NodeOutput(model=current_model)

class NodeInputSelect(io.ComfyNode):
    """
    NodeInputSelect:
      - The user is presented with a dropdown list of available nodes - a string containing the node id and type, separated using an _.
      - The user is presented with a dropdown list of available names for the inputs in the selected node.
      - A node that allows selection of a input from a list of available inputs.
      - Outputs the selected input name
      - Outputs the selected input id as a string.
      - Outputs the selected input value as a string.
    """

    @classmethod
    def define_schema(cls):
        node_data = None
        input_name = "unknown_input"        
        default_inputs = ["unknown_input"]
        # All nodes for the workflow
        nodes_data = get_workflow_all_nodes(cls.__name__)
        
        # List of node names for the dropdown
        nodes = listify_nodes_data(nodes_data)
        nodes = nodes if nodes is not None else []
        # The selected node, default to the first node if available
        first_node_key = list(nodes_data.keys())[0] if nodes_data and isinstance(nodes_data, dict) and len(nodes_data) > 0 else None

        if isinstance(nodes_data, dict) and first_node_key:
            node_data = nodes_data.get(first_node_key, None)

        default_node_name = nodes[0] if nodes and len(nodes) > 0 else "1_Unknown_Node"
        node_inputs = node_input_details(cls.__name__, node_data) if node_data else []
        
        if isinstance(node_inputs, dict):
            default_inputs = listify_node_inputs(node_inputs)
        
        if not default_inputs:
            default_inputs = ["unknown_input"]

        if node_inputs and isinstance(node_inputs, dict) and len(node_inputs) > 0:
            input_name = list(node_inputs.keys())[0]
    
        return io.Schema(
            node_id=prefixed_node_id("NodeInputSelect"),
            display_name="NodeInputSelect",
            category="🧊 frost-byte/Nodes",
            inputs=[
                io.Combo.Input(
                    id="node_name",
                    display_name="node_name",
                    options=nodes,
                    default=default_node_name,
                    tooltip="Select a node from the available nodes"
                ),
                io.Combo.Input(
                    id="input_name_in",
                    display_name="input_name",
                    options=default_inputs,
                    default=input_name,
                    tooltip="Select a widget from the available options"
                ),
            ],
            outputs=[
                io.String.Output(id="input_name_out", display_name="input_name", tooltip="Name of the selected input"),
                io.String.Output(id="input_value", display_name="input_value", tooltip="Value of the selected input"),
            ],
        )

    @classmethod
    def execute(
        cls,
        node_name: str = "1_Unknown_Node",
        input_name_in: str = "unknown_input",
    ):
        class_name = cls.__name__
        input_name_out= "No Inputs"
        input_value = ""

        logger.debug("%s: node='%s'; input_name_in='%s'", class_name, node_name, input_name_in)

        # All nodes for the workflow
        nodes_data = get_workflow_all_nodes(cls.__name__)

        if nodes_data is None or not isinstance(nodes_data, dict) or len(nodes_data) == 0:
            logger.warning("%s: No nodes available.", class_name)
            return io.NodeOutput(
                input_name_in,
                ""
            )

        logger.debug("%s: nodes_data keys=%s", class_name, list(nodes_data.keys()) if nodes_data else "None")

        # List of node names for the dropdown
        nodes = listify_nodes_data(nodes_data)
        logger.debug("%s: available nodes=%s", class_name, nodes)

        # The default is the first node, if available
        node_id = list(nodes_data.keys())[0] if nodes_data and len(nodes_data) > 0 else None

        # If a node name is provided, extract the node id
        if node_name != "1_Unknown_Node":
            node_id = node_name.split("_", 1)[0] if "_" in node_name else None

        if node_id is None:
            logger.warning("%s: Could not determine node_id from node_name='%s'", class_name, node_name)
            return io.NodeOutput(
                input_name_in,
                ""
            )

        logger.debug("%s: selected node_id=%s", class_name, node_id)

        if isinstance(nodes_data, dict):
            node_data = nodes_data.get(str(node_id), None)
            
            if node_data is None:
                logger.warning("%s: No data found for node_id=%s", class_name, node_id)
                return io.NodeOutput(
                    input_name_in,
                    ""
                )

            node_inputs = node_input_details(cls.__name__, node_data)

            if node_inputs and isinstance(node_inputs, dict):
                logger.debug("%s: node_inputs keys=%s", class_name, list(node_inputs.keys()))
                input_name_out = input_name_in if input_name_in and input_name_in in node_inputs.keys() else None
                
                # If the specified input name is not found, default to the first input
                if input_name_out == "No Inputs" or input_name_out is None:
                    input_name_out = list(node_inputs.keys())[0]

                input_value = node_inputs.get(input_name_out, "")

        logger.info(
            "%s: selected input_name='%s'; input_value='%s'",
            class_name,
            input_name_out,
            input_value,
        )

        return io.NodeOutput(
            input_name_out,
            input_value,
        )
