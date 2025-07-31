from models.alexnet import AlexNet
from models.light_autoencoder import LightAutoencoder
from utils.dataset import construct_random_wm_position

model = AlexNet(3, 10)
encoder =  LightAutoencoder().encoder
position_dict = construct_random_wm_position(model, 10)

print('position_dict length: %d.' % len(position_dict))

for i, client_positions in position_dict.items():
    print('client_dict %d length: %d.' % (i, len(client_positions)))

print(position_dict[0][:5])
