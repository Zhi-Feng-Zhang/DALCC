import numpy as np
import torch
import torch.utils
import torch.utils.data
import math
import cv2
import random
import matplotlib.pyplot as plt
import dataset.data_read as datasets
import  scipy.io as sio

#papremeter of data augment
augment_scale = [0.5, 1.0]
fcn_input_size = (256, 256)
augment_angle = 60


class dataset(torch.utils.data.Dataset):
    def __init__(self, Traing=False, file_path=None,file_path_list=None):
        self.data = []
        self.file_path = file_path  
        self.Traing = Traing
        self.file_path_list = file_path_list

        if self.file_path_list is not None:
            for file_path_fold in self.file_path_list:
                with open(file_path_fold) as f:
                    for line in f:
                        fields = line.split()[0]
                        fields = fields.split("/")[-1]
                        self.data.append(fields)
        else:
            with open(self.file_path) as f:
                for line in f:
                    fields = line.split()[0]
                    fields=fields.split("/")[-1]
                    self.data.append(fields)

    def __len__(self):
        return len(self.data)

    def _ranom_flip(self, image):
        return image[:, ::-1, :] if random.random() > 0.5 else image

    @staticmethod
    def rotate_image(image, angle):
        """
          Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
          (in degrees). The returned image will be large enough to hold the entire
          new image, with a black background
          """

        # Get the image size
        # No that's not an error - NumPy stores image matricies backwards
        image_size = (image.shape[1], image.shape[0])
        image_center = tuple(np.array(image_size) / 2)

        # Convert the OpenCV 3x2 rotation matrix to 3x3
        rot_mat = np.vstack(
            [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])

        rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

        # Shorthand for below calcs
        image_w2 = image_size[0] * 0.5
        image_h2 = image_size[1] * 0.5

        # Obtain the rotated coordinates of the image corners
        rotated_coords = [
            (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
        ]

        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        x_pos = [x for x in x_coords if x > 0]
        x_neg = [x for x in x_coords if x < 0]

        y_coords = [pt[1] for pt in rotated_coords]
        y_pos = [y for y in y_coords if y > 0]
        y_neg = [y for y in y_coords if y < 0]

        right_bound = max(x_pos)
        left_bound = min(x_neg)
        top_bound = max(y_pos)
        bot_bound = min(y_neg)

        new_w = int(abs(right_bound - left_bound))
        new_h = int(abs(top_bound - bot_bound))

        # We require a translation matrix to keep the image centred
        trans_mat = np.matrix([[1, 0, int(new_w * 0.5 - image_w2)],
                               [0, 1, int(new_h * 0.5 - image_h2)], [0, 0, 1]])

        # Compute the tranform for the combined rotation and translation
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

        # Apply the transform
        result = cv2.warpAffine(
            image, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)

        return result

    @staticmethod
    def largest_rotated_rect(w, h, angle):
        """
          Given a rectangle of size wxh that has been rotated by 'angle' (in
          radians), computes the width and height of the largest possible
          axis-aligned rectangle within the rotated rectangle.

          Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

          Converted to Python by Aaron Snoswell
          """

        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

        delta = math.pi - alpha - gamma

        length = h if (w < h) else w

        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        return (bb_w - 2 * x, bb_h - 2 * y)

   
    @staticmethod
    def crop_around_center(image, width, height):
        image_size = (image.shape[1], image.shape[0])
        image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

        if (width > image_size[0]):
            width = image_size[0]

        if (height > image_size[1]):
            height = image_size[1]

        x1 = int(image_center[0] - width * 0.5)
        x2 = int(image_center[0] + width * 0.5)
        y1 = int(image_center[1] - height * 0.5)
        y2 = int(image_center[1] + height * 0.5)

        return image[y1:y2, x1:x2]

    def rotate_and_crop(self, image, angle):
        image_width, image_height = image.shape[:2]
        image_rotated = self.rotate_image(image, angle)
        image_rotated_cropped = self.crop_around_center(image_rotated,
                                                        *self.largest_rotated_rect(
                                                            image_width, image_height,
                                                            math.radians(angle)))
        return image_rotated_cropped


    def augment_train(self, image, illum):
        scale = math.exp(random.random() * math.log(augment_scale[1] / augment_scale[0])) * augment_scale[0]
        s = min(max(int(round(min(image.shape[:2]) * scale)), 10), min(image.shape[:2]))

        start_x = random.randrange(0, image.shape[0] - s + 1)
        start_y = random.randrange(0, image.shape[1] - s + 1)

        image = image[start_x:start_x + s, start_y:start_y + s]
        image = self.rotate_and_crop(image, angle=(random.random() - 0.5) * augment_angle)

        image = cv2.resize(image, fcn_input_size)

        # Perform random left/right flip with probability 0.5
        if random.randint(0, 1):
            image = image[:, ::-1]
        image = image.astype(np.float32)
        return image, illum

    def crop_test(self, image, illum):
        image = cv2.resize(image, fcn_input_size)
        return image, illum

    def __getitem__(self, index):
        filename = self.data[index]
        image,illum,C_index=datasets.load_image(filename)

        if (self.Traing):
            image, illum = self.augment_train(image, illum)  
        else:
            image, illum = self.crop_test(image, illum)

        image=image/image.max()
        illum = illum / np.linalg.norm(illum)
        image = np.power(image, 1 / 2.2)  
        image = image.transpose(2, 0, 1)  # h*w*c to c*h*w
        image = torch.from_numpy(image.copy())  
        illum = torch.from_numpy(illum.copy())

        ret = (image, illum, C_index)
        return ret

if __name__ == "__main__":
    file=""  ### The dataset fold file for each dataset
    Dataset = dataset(Traing=True,file_path_list=[file,file])
    loader = torch.utils.data.DataLoader(Dataset, batch_size=1, shuffle=True, num_workers=12)
    for img, illum,C_index in loader:  #B 5 3
        print(img.size(), illum.size())
        img = img[0].cpu().numpy()
        illum = illum[0].cpu().numpy()
        print(illum)
        img = np.transpose(img, (1, 2, 0))
        plt.imshow(img)
        plt.show()
        break

