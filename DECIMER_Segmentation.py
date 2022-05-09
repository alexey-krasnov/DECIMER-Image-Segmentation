'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by ©Kohulan Rajan 2020
'''
import os
import numpy as np
from copy import deepcopy
from itertools import cycle
import skimage.io
import cv2
from typing import List, Tuple
from PIL import Image
import argparse
import warnings
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn import moldetect
from Scripts.complete_structure import complete_structure_mask

warnings.filterwarnings("ignore")

# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))


class InferenceConfig(moldetect.MolDetectConfig):
    """
    Inference configuration class for MRCNN
    """
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class DecimerSegmentation:
    """
    This class contains the main functionalities of DECIMER Segmentation
    """
    def __init__(self):
        self.model = self.load_model()
        
    def segment_chemical_structures_from_file(self, file_path):
        pass

    def segment_chemical_structures(self, image: np.array,
                                    expand: bool = True
                                    ) -> List[np.array]:
        """
        This function runs the segmentation model as well as the mask expansion
        -> returns a List of segmented chemical structure depictions (np.array)

        Args:
            image (np.array): image of a page from a scientific publication
            expand (bool): indicates whether or not to use mask expansion

        Returns:
            List[np.array]: expanded masks (shape: (h, w, num_masks))
        """
        if not expand:
            masks, _, _ = self.get_mrcnn_results(image)
        else:
            masks = self.get_expanded_masks(image)
        segments = self.apply_masks(image, masks)
        return segments

    def load_model(self) -> modellib.MaskRCNN:
        """
        This function loads the segmentation model and returns it. The weights
        are downloaded if necessary.

        Returns:
            modellib.MaskRCNN: MRCNN model with trained weights
        """
        # Define directory with trained model weights
        root_dir = os.path.split(__file__)[0]
        model_path = os.path.join(root_dir, "model_trained/mask_rcnn_molecule.h5")
        # Download trained weights if needed
        if not os.path.exists(model_path):
            utils.download_trained_weights(model_path)
        # Create model object in inference mode.
        model = modellib.MaskRCNN(mode="inference",
                                  model_dir=".",
                                  config=InferenceConfig())
        # Load weights
        model.load_weights(model_path, by_name=True)
        return model

    def get_expanded_masks(self, image: np.array) -> np.array:
        """
        This function runs the segmentation model and returns an
        array with the masks (shape: height, width, num_masks).
        Slicing along the third axis of the output of this function
        yields a binary array of shape (h, w) for a single structure.

        Args:
            image (np.array): image of a page from a scientific publication

        Returns:
            np.array: expanded masks (shape: (h, w, num_masks))
        """
        # Structure detection with MRCNN
        masks, _, _ = self.get_mrcnn_results(image)
        # Mask expansion
        expanded_masks = complete_structure_mask(image_array=image,
                                                 mask_array=masks)
        return expanded_masks

    def get_mrcnn_results(self, image: np.array) -> Tuple[np.array,
                                                          List[Tuple[int]],
                                                          List[float]]:
        """
        This function runs the segmentation model as well as the mask
        expansion mechanism and returns an array with the masks (shape:
        height, width, num_masks), a list of bounding boxes and a list
        of confidence scores.
        Slicing along the third axis of the mask output of this function
        yields a binary array of shape (h, w) for a single structure.

        Args:
            image (np.array): image of a page from a scientific publication
            List[Tuple[int]]: bounding boxes [(y0, x0, y1, x1), ...]
            List[float]: confidence scores
        Returns:
            np.array: expanded masks (shape: (h, w, num_masks))
        """
        results = self.model.detect([image], verbose=1)
        scores = results[0]['scores']
        bboxes = results[0]['rois']
        masks = results[0]['masks']
        return masks, bboxes, scores
    
    def apply_masks(self, image: np.array, masks: np.array) -> List[np.array]:
        """
        This function takes an image and the masks for this image
        (shape: (h, w, num_structures)) and returns a list of segmented
        chemical structure depictions (np.array)

        Args:
            image (np.array): image of a page from a scientific publication
            masks (np.array): masks (shape: (h, w, num_masks))

        Returns:
            List[np.array]: segmented chemical structure depictions
        """
        masks = [masks[:, :, i] for i in range(masks.shape[2])]
        segmented_images = map(self.apply_mask, cycle([image]), masks)
        return list(segmented_images)

    def apply_mask(self, image: np.array, mask: np.array) -> np.array:
        """
        This function takes an image and a mask for this image (shape: (h, w))
        and returns a segmented chemical structure depictions (np.array)

        Args:
            image (np.array): image of a page from a scientific publication
            masks (np.array): binary mask (shape: (h, w))

        Returns:
            np.array: segmented chemical structure depiction
        """
        # TODO: Further cleanup
        im = deepcopy(image)
        for channel in range(image.shape[2]):
            im[:, :, channel] = im[:, :, channel] * mask
        masked_image, bbox = self.get_masked_image(deepcopy(image),
                                                   mask)
        x, y, w, h = bbox
        im_gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
        _, im_bw = cv2.threshold(im_gray, 128, 255,
                                 cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # Removal of transparent layer and generation of segment
        _, alpha = cv2.threshold(im_bw, 0, 255, cv2.THRESH_BINARY)
        b, g, r = cv2.split(image)
        rgba = [b, g, r, alpha]
        dst = cv2.merge(rgba, 4)
        background = dst[y:y + h, x:x + w]
        trans_mask = background[:, :, 3] == 0
        background[trans_mask] = [255, 255, 255, 255]
        segmented_image = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)
        return segmented_image

    def get_masked_image(self, image: np.array, mask: np.array) -> np.array:
        """
            This function takes an image and a masks for this image
            (shape: (h, w)) and returns the masked image where only the
            masked area is not completely white and a bounding box of the
            segmented object

            Args:
                image (np.array): image of a page from a scientific publication
                mask (np.array): masks (shape: (h, w, num_masks))

            Returns:
                List[np.array]: segmented chemical structure depictions
                List[int]: bounding box of segmented object
        """
        for channel in range(image.shape[2]):
            image[:, :, channel] = image[:, :, channel] * mask[:, :]
        # Remove unwanted background
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(grayscale, 0, 255, cv2.THRESH_OTSU)
        bbox = cv2.boundingRect(thresholded)

        masked_image = np.zeros(image.shape).astype(np.uint8)
        masked_image = visualize.apply_mask(masked_image,
                                            mask,
                                            [1, 1, 1])
        masked_image = Image.fromarray(masked_image)
        masked_image = masked_image.convert('RGB')
        return np.array(masked_image), bbox


def main():
    # Handle input arguments
    parser = argparse.ArgumentParser(description="Select the chemical structures from a scanned literature and save them")
    parser.add_argument(
        '--input',
        help='Enter the input filename',
        required=True
    )
    args = parser.parse_args()

    # Define image path and output path
    IMAGE_PATH = os.path.normpath(args.input)
    output_directory = str(IMAGE_PATH) + '_output'
    if os.path.exists(output_directory):
        pass
    else:
        os.system("mkdir " + output_directory)

    # Segment chemical structure depictions
    #Save segments
        #Making directory for saving the segments
        if os.path.exists(output_directory+"/segments"):
            pass
        else:
            os.system("mkdir "+str(os.path.normpath(output_directory+"/segments")))

        #Define the correct path to save the segments
        segment_dirname = os.path.normpath(output_directory+"/segments/")
        filename = str(IMAGE_PATH).replace("\\", "/").split("/")[-1][:-4]+"_%d.png"%i
        file_path = os.path.normpath(segment_dirname + "/" +filename)

        print(file_path)
        cv2.imwrite(file_path, new_img)
    print("Segmented Images can be found in: ", str(os.path.normpath(zipper)))

if __name__ == '__main__':
    main()
