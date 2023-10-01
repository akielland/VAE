import litdata
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader


class ToRGBTensor:
    # converting gray scale images to 3 times the image to imitating RGB image dimesion
    def __call__(self, x):
        return T.functional.to_tensor(x).expand(3, -1, -1)


def load_car_data(batch_size=128):
    datapath = '/projects/ec232/data/'
    in_mean = [0.485, 0.456, 0.406]
    in_std = [0.229, 0.224, 0.225]
    
    postprocess = (
        T.Compose([
            T.Resize((96, 128), antialias=True),  # Resize with antialiasing.
            ToRGBTensor(),        # Convert from PIL image to RGB torch.Tensor.
            T.Normalize(in_mean, in_std),
        ]),
        T.ToTensor(),
    )

    data = litdata.LITDataset(
        'CarRecs',
        datapath,
        override_extensions=[
            'jpg',
            'scores.npy'
        ]
    ).map_tuple(*postprocess)
    
    train_loader = DataLoader(data, shuffle=True, batch_size=batch_size)
    return train_loader
