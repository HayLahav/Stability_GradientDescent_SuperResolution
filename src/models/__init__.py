"""
Model implementations for stability analysis in super-resolution
"""

# Import individual components to avoid circular imports
from .srcnn import SimpleSRCNN
from .adafm_layers import AdaFMLayer
from .correction_filter import CorrectionFilter

# Import utility function that doesn't create circular dependencies
def create_srcnn_variant(use_correction=False, use_adafm=False, **kwargs):
    """
    Factory function to create SRCNN variants
    
    Args:
        use_correction: Whether to use correction filter
        use_adafm: Whether to use AdaFM layers
        **kwargs: Additional arguments for SimpleSRCNN
        
    Returns:
        SimpleSRCNN model instance
    """
    return SimpleSRCNN(
        use_correction=use_correction,
        use_adafm=use_adafm,
        **kwargs
    )

# Conditional import to avoid issues when siamese module doesn't exist
try:
    from .siamese_network import SiameseNetwork
    __all__ = [
        'SimpleSRCNN',
        'create_srcnn_variant',
        'AdaFMLayer', 
        'CorrectionFilter',
        'SiameseNetwork'
    ]
except ImportError:
    # Fallback if siamese module is missing
    SiameseNetwork = None
    __all__ = [
        'SimpleSRCNN',
        'create_srcnn_variant',
        'AdaFMLayer', 
        'CorrectionFilter'
    ]
