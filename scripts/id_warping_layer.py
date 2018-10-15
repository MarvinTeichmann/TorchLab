import numpy as np
from matplotlib import pyplot as plt


def fake_warp(img):
    result = img.copy()
    result[50:100, 50:100] = 0
    plt.imshow(np.abs(img - result))
    plt.show()
    return result


W = 200
H = 100
id_array_list = np.arange(W * H)
feature_image = np.random.rand(H, W, 3)
feature_image_list = np.reshape(feature_image, (H * W, 3))

id_array_image = np.reshape(id_array_list, (H, W))

warped_id_array_image = fake_warp(id_array_image)
plt.imshow(warped_id_array_image)
plt.show()
warped_id_array_list = np.reshape(warped_id_array_image, (W * H))

wfil_array = np.zeros((W * H, 3))

for c in range(feature_image.shape[-1]):
    wfil_array[:, c] = np.take(
        feature_image_list[:, c], np.int32(warped_id_array_list))

print(wfil_array.shape)
warped_feature_image = np.reshape(wfil_array, (H, W, 3))

plt.imshow(np.concatenate((warped_feature_image, np.abs(
    warped_feature_image - feature_image)), axis=1))
plt.show()
