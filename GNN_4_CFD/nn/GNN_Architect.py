"""
GNN Architecture Builder
====================================
A flexible system to programmatically generate multi-scale GNN architectures
with arbitrary numbers of layers, hidden dimensions, and message passing blocks.

Usage:
    from GNN_Architect import GNNArchBuilder, MultiScaleGNNBuilde
    
    # Simple single-scale GNN
    arch = GNNArchBuilder(
        num_mp_blocks=8,
        hidden_dim=128,
        edge_encoder_in=2,
        node_encoder_in=5,
        decoder_out=3
    ).build_single_scale()
    
    # Multi-scale GNN (U-Net style with skip connections)
    arch = MultiScaleGNNBuilder(
        num_scales=3,
        mp_blocks_per_scale=[4, 4, 4],
        hidden_dims=[64, 128, 256],
        edge_encoder_in=2,
        node_encoder_in=5,
        decoder_out=3,
        skip_connections=True
    ).build()
"""

from typing import Dict, List, Tuple, Optional, Union
import warnings


class GNNArchBuilder:
    """
    Builder for single-scale GNN architectures.
    Generates message passing block specifications following graphs4cfd convention.
    """
    
    def __init__(self,
                 arch_parameters: Dict = {
                    "num_mp_blocks": 4,
                    "hidden_dim": 128,
                    "edge_encoder_in": 2,
                    "node_encoder_in": 4,
                    "decoder_out": 3,
                    "mlp_depth": 3},
                 use_activation: bool = True                 
                ):
        """
        Args:
            num_mp_blocks: Number of message passing blocks to create
            hidden_dim: Hidden dimension(s). If int, use uniform; if list, specify per-block
            edge_encoder_in: Input dimension for edge encoder (edge features)
            node_encoder_in: Input dimension for node encoder (concatenated node features)
            decoder_out: Output dimension from decoder (e.g., 3 for u,v,p)
            mlp_depth: Depth of MLP layers (number of hidden layers in each MLP)
            use_activation: Whether to use activation in encoder/decoder
        """


        num_mp_blocks = arch_parameters.get("num_mp_blocks")
        hidden_dim=arch_parameters.get("hidden_dim")
        edge_encoder_in=arch_parameters.get("edge_encoder_in")
        node_encoder_in=arch_parameters.get("node_encoder_in")
        decoder_out=arch_parameters.get("decoder_out")
        mlp_depth=arch_parameters.get("mlp_depth")

        self.num_mp_blocks = num_mp_blocks
        self.hidden_dim = hidden_dim
        self.edge_encoder_in = edge_encoder_in
        self.node_encoder_in = node_encoder_in
        self.decoder_out = decoder_out
        self.mlp_depth = mlp_depth
        self.use_activation = use_activation
        
        # Validate and expand hidden_dim
        if isinstance(hidden_dim, int):
            self.hidden_dims = [hidden_dim] * num_mp_blocks
        elif isinstance(hidden_dim, (list, tuple)):
            if len(hidden_dim) != num_mp_blocks:
                raise ValueError(
                    f"Length of hidden_dim list ({len(hidden_dim)}) must match "
                    f"num_mp_blocks ({num_mp_blocks})"
                )
            self.hidden_dims = list(hidden_dim)
        else:
            raise TypeError("hidden_dim must be int or list of ints")
    
    def _make_mlp_hidden_dims(self, dim: int) -> Tuple[int, ...]:
        """Create tuple of hidden dimensions for MLP with specified depth."""
        return tuple([dim] * self.mlp_depth)
    
    def _make_mp_block_spec(self, 
                           block_idx: int,
                           edge_attr_in: int,
                           node_attr_in: int) -> Tuple[Tuple, Tuple]:
        """
        Create edge and node function specifications for a single MP block.
        
        Returns:
            (edge_spec, node_spec) where each is (input_dim, hidden_dims, use_activation)
        """
        hidden = self.hidden_dims[block_idx]
        
        # Edge function: takes edge features + 2x node features (source & target)
        edge_in = edge_attr_in + 2 * hidden
        edge_spec = (edge_in, self._make_mlp_hidden_dims(hidden), self.use_activation)
        
        # Node function: takes node features + edge features (aggregated from neighbors)
        node_in = node_attr_in + hidden
        node_spec = (node_in, self._make_mlp_hidden_dims(hidden), self.use_activation)
        
        return edge_spec, node_spec
    
    def build_single_scale(self) -> Dict:
        """
        Build single-scale GNN architecture.
        
        Returns:
            dict: Architecture specification compatible with graphs4cfd.nn.GNN
        """
        arch = {}
        
        # Encoder
        arch["edge_encoder"] = (
            self.edge_encoder_in,
            self._make_mlp_hidden_dims(self.hidden_dims[0]),
            False  # No activation in encoder
        )
        arch["node_encoder"] = (
            self.node_encoder_in,
            self._make_mlp_hidden_dims(self.hidden_dims[0]),
            False  # No activation in encoder
        )
        
        # Message passing blocks
        current_edge_dim = self.hidden_dims[0]
        current_node_dim = self.hidden_dims[0]
        
        for i in range(self.num_mp_blocks):
            block_name = f"mp{i+1:02d}"
            edge_spec, node_spec = self._make_mp_block_spec(
                i, 
                current_edge_dim,
                current_node_dim
            )
            arch[block_name] = (edge_spec, node_spec)
            
            # Update dimensions for next block (assuming residual connections preserve dim)
            current_edge_dim = self.hidden_dims[i]
            current_node_dim = self.hidden_dims[i]
        
        # Decoder
        arch["decoder"] = (
            self.hidden_dims[-1],
            self._make_mlp_hidden_dims(self.hidden_dims[-1]) + (self.decoder_out,),
            False  # No activation in decoder
        )
        
        return arch
    
    def summary(self) -> str:
        """Print architecture summary."""
        lines = [
            "Single-Scale GNN Architecture",
            f"  MP blocks: {self.num_mp_blocks}",
            f"  Hidden dims: {self.hidden_dims}",
            f"  Edge encoder input: {self.edge_encoder_in}",
            f"  Node encoder input: {self.node_encoder_in}",
            f"  Decoder output: {self.decoder_out}",
            f"  MLP depth: {self.mlp_depth}"
        ]
        return "\n".join(lines)


class MultiScaleGNNBuilder:
    """
    Builder for multi-scale (U-Net style) GNN architectures.
    Supports arbitrary numbers of scales with skip connections.
    """
    
    def __init__(self,
                 num_scales: int,
                 mp_blocks_per_scale: Union[int, List[int]],
                 hidden_dims: Union[int, List[int]],
                 edge_encoder_in: int = 2,
                 node_encoder_in: int = 5,
                 decoder_out: int = 3,
                 mlp_depth: int = 3,
                 skip_connections: bool = True,
                 num_edge_encoders: Optional[int] = None):
        """
        Args:
            num_scales: Number of resolution scales (pyramid levels)
            mp_blocks_per_scale: MP blocks per scale. If int, use same for all; if list, specify per-scale
            hidden_dims: Hidden dimensions per scale. If int, use same; if list, specify per-scale
            edge_encoder_in: Input dimension for edge encoder
            node_encoder_in: Input dimension for node encoder
            decoder_out: Output dimension
            mlp_depth: Depth of MLPs
            skip_connections: Whether to use skip connections (for upsampling)
            num_edge_encoders: Number of separate edge encoders (one per scale). 
                              If None, defaults to num_scales (multi-scale needs separate encoders)
        """
        self.num_scales = num_scales
        self.edge_encoder_in = edge_encoder_in
        self.node_encoder_in = node_encoder_in
        self.decoder_out = decoder_out
        self.mlp_depth = mlp_depth
        self.skip_connections = skip_connections
        self.num_edge_encoders = num_edge_encoders or num_scales
        
        # Validate and expand mp_blocks_per_scale
        if isinstance(mp_blocks_per_scale, int):
            self.mp_blocks_per_scale = [mp_blocks_per_scale] * num_scales
        elif isinstance(mp_blocks_per_scale, (list, tuple)):
            if len(mp_blocks_per_scale) != num_scales:
                raise ValueError(
                    f"Length of mp_blocks_per_scale ({len(mp_blocks_per_scale)}) "
                    f"must match num_scales ({num_scales})"
                )
            self.mp_blocks_per_scale = list(mp_blocks_per_scale)
        else:
            raise TypeError("mp_blocks_per_scale must be int or list")
        
        # Validate and expand hidden_dims
        if isinstance(hidden_dims, int):
            self.hidden_dims = [hidden_dims] * num_scales
        elif isinstance(hidden_dims, (list, tuple)):
            if len(hidden_dims) != num_scales:
                raise ValueError(
                    f"Length of hidden_dims ({len(hidden_dims)}) must match num_scales ({num_scales})"
                )
            self.hidden_dims = list(hidden_dims)
        else:
            raise TypeError("hidden_dims must be int or list")
    
    def _make_mlp_hidden_dims(self, dim: int) -> Tuple[int, ...]:
        """Create tuple of hidden dimensions for MLP."""
        return tuple([dim] * self.mlp_depth)
    
    def _make_mp_block_spec(self,
                           hidden_dim: int,
                           edge_attr_in: int,
                           node_attr_in: int) -> Tuple[Tuple, Tuple]:
        """Create edge and node function specs for a single MP block."""
        edge_in = edge_attr_in + 2 * hidden_dim
        node_in = node_attr_in + hidden_dim
        
        edge_spec = (edge_in, self._make_mlp_hidden_dims(hidden_dim), True)
        node_spec = (node_in, self._make_mlp_hidden_dims(hidden_dim), True)
        
        return edge_spec, node_spec
    
    def build(self) -> Dict:
        """
        Build multi-scale GNN architecture.
        
        Naming convention:
        - Encoders: edge_encoder, edge_encoder2, edge_encoder3, ...
        - Scale L blocks: mp<L><i> (e.g., mp111, mp112 for scale 1, blocks 1-2)
        - Downsampling: down_mp<L><L+1>
        - Upsampling: up_mp<L+1><L>
        
        Returns:
            dict: Architecture specification
        """
        arch = {}
        
        # Encoders
        arch["edge_encoder"] = (
            self.edge_encoder_in,
            self._make_mlp_hidden_dims(self.hidden_dims[0]),
            False
        )
        arch["node_encoder"] = (
            self.node_encoder_in,
            self._make_mlp_hidden_dims(self.hidden_dims[0]),
            False
        )
        
        # Additional edge encoders for multi-scale
        for scale_idx in range(1, self.num_scales):
            encoder_name = f"edge_encoder{scale_idx+1}"
            arch[encoder_name] = (
                self.edge_encoder_in,
                self._make_mlp_hidden_dims(self.hidden_dims[scale_idx]),
                False
            )
        
        # Encoder path (downsampling)
        for scale_idx in range(self.num_scales):
            scale_num = scale_idx + 1
            hidden = self.hidden_dims[scale_idx]
            num_blocks = self.mp_blocks_per_scale[scale_idx]
            
            # MP blocks at this scale
            for block_idx in range(num_blocks):
                block_num = block_idx + 1
                block_name = f"mp{scale_num}{block_num:02d}" if scale_num <= 2 else f"mp{scale_num}{block_num}"
                
                # Determine input edge dimension
                edge_attr_in = hidden  # From encoder or previous block
                node_attr_in = hidden
                
                edge_spec, node_spec = self._make_mp_block_spec(
                    hidden, edge_attr_in, node_attr_in
                )
                arch[block_name] = (edge_spec, node_spec)
            
            # Downsampling layer (except at deepest scale)
            if scale_idx < self.num_scales - 1:
                down_mp_name = f"down_mp{scale_num}{scale_num+1}"
                edge_attr_in = 2 + hidden  # Edge attr from current scale
                arch[down_mp_name] = (edge_attr_in, self._make_mlp_hidden_dims(hidden), True)
        
        # Decoder path (upsampling)
        for scale_idx in range(self.num_scales - 2, -1, -1):
            scale_num = scale_idx + 1
            hidden = self.hidden_dims[scale_idx]
            num_blocks = self.mp_blocks_per_scale[scale_idx]
            
            # Upsampling layer
            up_mp_name = f"up_mp{scale_num+1}{scale_num}"
            if self.skip_connections:
                # Skip connection concatenates features from encoder path
                upsample_in = 2 * hidden + hidden  # from coarse + skip connection
            else:
                upsample_in = 2 * hidden
            arch[up_mp_name] = (upsample_in, self._make_mlp_hidden_dims(hidden), True)
            
            # MP blocks at this scale (after upsampling)
            for block_idx in range(num_blocks):
                block_num = block_idx + 1
                block_name = f"mp{scale_num}{block_num+num_blocks:02d}"
                
                edge_attr_in = hidden
                node_attr_in = 2 * hidden if block_idx == 0 else hidden
                
                edge_spec, node_spec = self._make_mp_block_spec(
                    hidden, edge_attr_in, node_attr_in
                )
                arch[block_name] = (edge_spec, node_spec)
        
        # Decoder
        arch["decoder"] = (
            self.hidden_dims[0],
            self._make_mlp_hidden_dims(self.hidden_dims[0]) + (self.decoder_out,),
            False
        )
        
        return arch
    
    def summary(self) -> str:
        """Print architecture summary."""
        lines = [
            "Multi-Scale GNN Architecture",
            f"  Scales: {self.num_scales}",
            f"  MP blocks per scale: {self.mp_blocks_per_scale}",
            f"  Hidden dims: {self.hidden_dims}",
            f"  Skip connections: {self.skip_connections}",
            f"  Edge encoders: {self.num_edge_encoders}",
            f"  Edge encoder input: {self.edge_encoder_in}",
            f"  Node encoder input: {self.node_encoder_in}",
            f"  Decoder output: {self.decoder_out}",
            f"  MLP depth: {self.mlp_depth}"
        ]
        return "\n".join(lines)
