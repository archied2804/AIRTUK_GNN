import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Dict, List, Tuple
import re

from .model import GNN
from .blocks import MLP, MP, DownMP, UpMP
from ..graph import Graph

class UniversalGNN(GNN):
    """
    A Universal Graph Neural Network class that dynamically constructs 
    Single-Scale or Multi-Scale (U-Net) architectures based on a configuration dictionary.
    
    This replaces specific implementations like OneScale_Cardiac.py.
    
    It supports:
    1. Dynamic number of scales.
    2. Dynamic number of MP blocks per scale.
    3. Automatic detection of U-Net structures (DownMP/UpMP).
    4. Skip connections.
    
    Args:
        arch (dict, optional): The architecture dictionary (output from GNNArchBuilder).
        weights (str, optional): Path to pretrained weights.
        checkpoint (str, optional): Path to a checkpoint.
    """

    def __init__(self, arch: Dict = None, *args, **kwargs) -> None:
        super().__init__(arch=arch, *args, **kwargs)

    def load_arch(self, arch: Dict):
        """
        Parses the architecture dictionary and initializes the PyTorch modules.
        """
        self.arch = arch
        self.mp_blocks = nn.ModuleDict()
        self.down_blocks = nn.ModuleDict()
        self.up_blocks = nn.ModuleDict()
        self.encoders = nn.ModuleDict()
        
        # 1. Load Encoders
        # ----------------
        # Standard encoders
        if "edge_encoder" in arch:
            self.encoders["edge_encoder"] = MLP(*arch["edge_encoder"])
        if "node_encoder" in arch:
            self.encoders["node_encoder"] = MLP(*arch["node_encoder"])
            
        # Additional edge encoders for multi-scale (e.g., edge_encoder2, edge_encoder3)
        for key in arch:
            if key.startswith("edge_encoder") and key not in ["edge_encoder"]:
                self.encoders[key] = MLP(*arch[key])

        # 2. Load Message Passing Blocks (mp{scale}{block})
        # ------------------------------------------------
        # We store them in a ModuleDict, but we need to identify max scale to organize flow later
        self.scales = set()
        
        mp_pattern = re.compile(r"^mp(\d+)(\d+)$") # Matches mp101, mp201, etc.
        
        for key, args in arch.items():
            match = mp_pattern.match(key)
            if match:
                scale_lvl = int(match.group(1))
                self.scales.add(scale_lvl)
                self.mp_blocks[key] = MP(*args)

        # 3. Load Downsampling Blocks (down_mp{s}{s+1})
        # ---------------------------------------------
        down_pattern = re.compile(r"^down_mp(\d+)(\d+)$")
        for key, args in arch.items():
            if down_pattern.match(key):
                self.down_blocks[key] = DownMP(*args)

        # 4. Load Upsampling Blocks (up_mp{s+1}{s})
        # -----------------------------------------
        up_pattern = re.compile(r"^up_mp(\d+)(\d+)$")
        for key, args in arch.items():
            if up_pattern.match(key):
                self.up_blocks[key] = UpMP(*args)

        # 5. Load Decoder
        # ---------------
        if "decoder" in arch:
            self.node_decoder = MLP(*arch["decoder"])
        
        self.num_scales = max(self.scales) if self.scales else 1
        self.to(self.device)

    def _get_sorted_mp_keys(self, scale: int) -> List[str]:
        """
        Helper to get MP block keys for a specific scale in the correct execution order.
        e.g., returns ['mp101', 'mp102', 'mp103']
        """
        # Filter keys belonging to this scale
        keys = [k for k in self.mp_blocks.keys() if k.startswith(f"mp{scale}")]
        
        # Sort by the numeric value of the full string to ensure 101 comes before 102
        # Assuming the builder ensures zero-padding or consistent lengths, string sort works.
        # If not, we parse the block number.
        def block_sorter(k):
            # Extract the block part. Assuming format mp{scale}{block}
            # Remove the 'mp{scale}' prefix
            prefix = f"mp{scale}"
            block_num = int(k[len(prefix):])
            return block_num
            
        return sorted(keys, key=block_sorter)

    def forward(self, graph: Graph, t: Optional[int] = None) -> torch.Tensor:
        """
        Dynamic forward pass handling single-scale or U-Net flow.
        """
        original_field = graph.field
        original_edge_attr = graph.edge_attr
        
        # -----------------------------------------------------------
        # 1. Preprocessing & Encoding (Scale 1)
        # -----------------------------------------------------------
        # Concatenate available fields (field + omega + etc)
        # Allows usage with different datasets (Cardiac vs NS)
        features = [graph.field]
        if hasattr(graph, 'omega'): features.append(graph.omega)
        # Add other fields if necessary (loc, glob) based on dataset
        
        graph.field = torch.cat(features, dim=1)

        # Apply Encoders
        if "edge_encoder" in self.encoders:
            graph.edge_attr = F.selu(self.encoders["edge_encoder"](graph.edge_attr))
        
        if "node_encoder" in self.encoders:
            graph.field = F.selu(self.encoders["node_encoder"](graph.field))

        # -----------------------------------------------------------
        # 2. Downward Sweep (Encoder Path)
        # -----------------------------------------------------------
        skip_connections = {} # Store fields for skip connections: {scale: field_tensor}

        for scale in range(1, self.num_scales + 1):
            
            # A. Prepare Edge Attributes for this scale (if multi-scale)
            if scale > 1:
                # For scale > 1, we might have specific edge encoders (edge_encoder2, etc.)
                encoder_key = f"edge_encoder{scale}"
                if encoder_key in self.encoders:
                    # Retrieve the edge attribute for this scale (graph.e_12, graph.e_23 etc usually handle relative pos)
                    # However, usually the graph already has edge_attr, edge_attr2, etc. pre-loaded or computed
                    # checking specific graphs4cfd conventions:
                    current_edge_attr_name = "edge_attr" if scale == 1 else f"edge_attr{scale}"
                    if hasattr(graph, current_edge_attr_name):
                         # Encode it
                         e_attr = getattr(graph, current_edge_attr_name)
                         e_encoded = F.selu(self.encoders[encoder_key](e_attr))
                         setattr(graph, current_edge_attr_name, e_encoded)

            # B. Execute MP blocks for this scale (Pre-Downsampling)
            mp_keys = self._get_sorted_mp_keys(scale)
            
            # In U-Net, usually we process MP blocks, then save skip, then downsample.
            # However, the builder generates blocks like mp101...mp104. 
            # In the down-path, we run all defined blocks for this scale.
            
            # Note: If this is a U-Net, the 'mp_keys' might include blocks meant for the 
            # UP-path (after upsampling). 
            # The builder logic:
            # Scale 1: mp101, mp102 (Down) ... [Up] ... mp103, mp104 (Up)
            # We need to split these based on the builder's logic.
            # Assuming standard builder: mp_blocks_per_scale is split half down, half up?
            # Actually, looking at the builder code provided:
            # - Down path adds: mp{scale}{1..N}
            # - Up path adds: mp{scale}{N+1..2N}
            # We need to detect which blocks are "Down" blocks and which are "Up" blocks.
            
            # HEURISTIC: If up_mp exists for this scale, we split blocks.
            # If no up_mp exists (Single Scale), we run all.
            
            num_blocks_total = len(mp_keys)
            
            # If we are going to downsample from this scale, we only run the first half of blocks?
            # Or does the builder define them explicitly?
            # In the builder:
            # Down loop: adds mp{scale}{1..N}
            # Up loop: adds mp{scale}{N+1..2N}
            # So simply sorting them puts them in 1,2,3,4 order.
            
            # Execution Strategy:
            # 1. Run "Down" blocks.
            # 2. Store Skip.
            # 3. Downsample.
            
            # We determine split index. If scale == num_scales (Bottleneck), run ALL.
            # If scale < num_scales, run first half (Down), second half comes later (Up).
            
            if scale < self.num_scales:
                # Identify split point. Assuming equal blocks down/up if not explicitly tagged.
                # Based on builder: num_blocks down = num_blocks up.
                split_idx = num_blocks_total // 2
                
                # Special case: If builder didn't create Up blocks for scale 1 (e.g. pure down), 
                # but valid U-Nets always have symmetry.
                down_mp_keys = mp_keys[:split_idx]
                up_mp_keys_for_later = mp_keys[split_idx:]
                
                # Execute Down Blocks
                for key in down_mp_keys:
                    graph.field, graph.edge_attr = self.mp_blocks[key](
                        graph.field, graph.edge_attr, graph.edge_index
                    )
                    graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
                
                # Store Skip Connection
                skip_connections[scale] = (graph.field.clone(), graph.pos)

                # Execute Downsampling
                down_key = f"down_mp{scale}{scale+1}"
                if down_key in self.down_blocks:
                    graph = self.down_blocks[down_key](graph, activation=F.selu)
            
            else:
                # BOTTLENECK (Deepest Scale)
                # Run ALL blocks here
                for key in mp_keys:
                    # Note: At deeper scales, edge_index might be edge_index2, edge_attr2, etc.
                    # The MP block expects standard names (x, edge_index, edge_attr).
                    # Graphs4CFD DownMP updates graph.edge_index/graph.edge_attr to the current resolution.
                    # So we can simply pass graph.field, graph.edge_attr, graph.edge_index.
                    
                    graph.field, graph.edge_attr = self.mp_blocks[key](
                        graph.field, graph.edge_attr, graph.edge_index
                    )
                    graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)

        # -----------------------------------------------------------
        # 3. Upward Sweep (Decoder Path)
        # -----------------------------------------------------------
        for scale in range(self.num_scales - 1, 0, -1):
            
            # A. Upsample
            up_key = f"up_mp{scale+1}{scale}"
            if up_key in self.up_blocks:
                field_skip, pos_skip = skip_connections[scale]
                graph = self.up_blocks[up_key](
                    graph, field_skip, pos_skip, activation=F.selu
                )

            # B. Execute Up Blocks
            # These are the second half of the blocks for this scale
            mp_keys = self._get_sorted_mp_keys(scale)
            num_blocks_total = len(mp_keys)
            split_idx = num_blocks_total // 2
            up_mp_keys = mp_keys[split_idx:]
            
            for key in up_mp_keys:
                graph.field, graph.edge_attr = self.mp_blocks[key](
                    graph.field, graph.edge_attr, graph.edge_index
                )
                graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)

        # -----------------------------------------------------------
        # 4. Decode & Restore
        # -----------------------------------------------------------
        graph.field = F.selu(graph.field)
        output = self.node_decoder(graph.field)
        
        # Restore original data
        graph.field = original_field
        graph.edge_attr = original_edge_attr
        
        # Return prediction (Residual connection: previous field + delta)
        # Use only the last N fields where N is output size
        return graph.field[:,-self.num_fields:] + output