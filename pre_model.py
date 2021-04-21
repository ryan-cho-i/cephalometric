import segmentation_models_pytorch as smp
import os 

def build_model(Architecture, encoder, weights, CLASSES, activation):
        
    model_list = { 
    
    "DeepLabV3" : smp.DeepLabV3,
    "DeepLabV3Plus" : smp.DeepLabV3Plus,
    "FPN" : smp.FPN,
    "Linknet" : smp.Linknet,
    "PSPnet" : smp.PSPNet,
    "Unet" : smp.Unet,
    "UnetPlusPlus" : smp.UnetPlusPlus
    
    }

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, weights)

    model = model_list[Architecture](
            encoder_name=encoder,
            encoder_weights=weights,
            classes=len(CLASSES), 
            activation=activation,
        )

    return model, preprocessing_fn