from GNN_Architect import GNNArchBuilder, MultiScaleGNNBuilder


def main():    
    print("="*80)
    print("SINGLE-SCALE GNN EXAMPLE")
    print("="*80)
    
    # Build single-scale GNN with 8 MP blocks
    builder = GNNArchBuilder(
        num_mp_blocks=8,
        hidden_dim=128,
        edge_encoder_in=2,
        node_encoder_in=5,
        decoder_out=3,
        mlp_depth=3
    )
    
    print(builder.summary())
    arch = builder.build_single_scale()
    
    print("\nGenerated architecture keys:")
    for key in arch:
        print(f"  {key}: {arch[key]}")
    
    print("\n" + "="*80)
    print("MULTI-SCALE GNN EXAMPLE (3 scales)")
    print("="*80)
    
    # Build 3-scale GNN with varying MP blocks and hidden dims
    multi_builder = MultiScaleGNNBuilder(
        num_scales=3,
        mp_blocks_per_scale=[4, 4, 4],
        hidden_dims=[128, 128, 128],
        edge_encoder_in=2,
        node_encoder_in=5,
        decoder_out=3,
        mlp_depth=3,
        skip_connections=True
    )
    
    print(multi_builder.summary())
    multi_arch = multi_builder.build()
    
    print("\nGenerated architecture keys:")
    for key in sorted(multi_arch.keys()):
        print(f"  {key}")


if __name__ == "__main__":
    main()
