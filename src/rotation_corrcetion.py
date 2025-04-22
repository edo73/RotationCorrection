
import cv2
import numpy as np
from math import degrees, atan2
import matplotlib.pyplot as plt


class RotationCorrection():

    def load_image(self,image_path: str):
        """
        This function loads an image from a fime

        Args:
            filepath: string

        Returns:
            image: image loaded
        """
        # Load image from file
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Show image and let user select ROI
        roi = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=True)

        # Close the ROI selection window
        cv2.destroyWindow("Select ROI")

        # Crop the selected region
        x, y, w, h = roi
        cropped_image = image[y:y + h, x:x + w]

        # Show cropped image
        cv2.imshow("Cropped ROI", cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return cropped_image, image

    def correct_image(self,image, x: float = 0, y: float = 0, angle: float = 0, shift_first: bool = True, y_bottom_up: bool =True):
        """
        This function copies the input image in variable image_out and applies a x and y shift and a rotation to the image_out.

        Args:
            image: image input
            x: shift in pixel. positive left to right
            y: shift in pixel. positive bottom to up
            angle: rotation angle in degrees. Positive CCW
            shift_first: boolean which determines whether to do the shift or the rotation first
            y_bottom_up: boolean which determines the direction of the y shift. If "False", then positive direction is up to bottom

        Returns:
            image_out: image corrected by shift and rotation
        """


    def _find_template_location(self, corrected_img, template_img):
        """
        Uses template matching to locate the region in the corrected image that best matches the template.
        """
        result = cv2.matchTemplate(corrected_img, template_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < 0.5:
            raise ValueError("Template matching confidence too low.")

        return max_loc  # top-left corner of best match


    def calculate_offset_and_rotation(self, template_img, corrected_img, visualize=True):
        """
        Matches the template image against the corrected image, using template matching for localization
        and ORB feature matching for rotation + offset calculation.
        """

        # Step 1: Use template matching to find where the template lies in the larger image
        top_left = self._find_template_location(corrected_img, template_img)
        h, w = template_img.shape[:2]

        # Ensure the crop fits inside corrected image
        if (top_left[1] + h > corrected_img.shape[0]) or (top_left[0] + w > corrected_img.shape[1]):
            raise ValueError("Cropped region exceeds corrected image boundaries.")

        roi = corrected_img[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]

        # Step 2: ORB keypoint detection
        """ 
        ORB stands for Oriented FAST and Rotated BRIEF. It’s a fast and efficient algorithm for:
            -	Detecting keypoints (important spots in the image like corners or edges).
            -	Computing descriptors (vectors that describe how that keypoint looks).
            -	kp1 and kp2: lists of keypoints found in each image.
            -	des1 and des2: the descriptors (numeric features) for each keypoint.
        """
        orb = cv2.ORB_create(5000)
        kp1, des1 = orb.detectAndCompute(template_img, None)
        kp2, des2 = orb.detectAndCompute(roi, None)

        if des1 is None or des2 is None:
            raise ValueError("Keypoints could not be detected in one or both images.")

        # Step 3: Feature matching
        """ 
        BFMatcher = Brute-Force Matcher:
            -	Takes each descriptor from des1 (template) and compares it with every descriptor in des2 (corrected).
            -	Finds the closest match using a distance metric.
            -	cv2.NORM_HAMMING: This is used for binary descriptors like ORB.
            -	crossCheck=True: Ensures that a match is mutual. If A matches B, B must also match A.
            -	Result: matches is a list of correspondences between keypoints in both images.
        """
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 4:
            raise ValueError("Not enough matches for transformation.")

        # Step 4: Extract coordinates of matching keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Step 5: Estimate affine transformation
        """
            M = [ cosθ  -sinθ   dx ]
                [ sinθ   cosθ   dy ]
        """
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

        if M is None:
            raise ValueError("Could not estimate affine transformation.")

        dx = M[0, 2] + top_left[0]  # adjust with offset of ROI
        dy = M[1, 2] + top_left[1]
        angle = degrees(atan2(M[1, 0], M[0, 0]))

        # Step 6: Visualization
        if visualize:
            # Draw top matches
            match_img = cv2.drawMatches(template_img, kp1, roi, kp2, matches[:20], None, flags=2)
            plt.figure(figsize=(12, 6))
            plt.title("Keypoint Matches (Template vs ROI)")
            plt.imshow(match_img, cmap='gray')
            plt.axis("off")
            plt.show()

            # Overlay warped template on original corrected image
            warped = cv2.warpAffine(template_img, M, (w, h))
            full_overlay = corrected_img.copy()
            full_overlay[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w] = \
                cv2.addWeighted(roi, 0.5, warped, 0.5, 0)

            plt.figure(figsize=(6, 6))
            plt.title("Overlay: Corrected Image + Transformed Template")
            plt.imshow(full_overlay, cmap='gray')
            plt.axis("off")
            plt.show()

        return dx, dy, angle



rc = RotationCorrection()

template, dut = rc.load_image("image_1.png")
cv2.imshow("Cropped ROI", template)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Assuming you already have the dataclass setup and images loaded
"""template_item = rc.get_template_item(name, item_id)
correction_item = rc.get_correction_item(name, temp, item_id)

# Make sure images are grayscale
template_gray = cv2.cvtColor(template_item.image, cv2.COLOR_BGR2GRAY)
correction_gray = cv2.cvtColor(correction_item.image, cv2.COLOR_BGR2GRAY)

dx, dy, angle = calculate_offset_and_rotation(template_gray, correction_gray)
print(f"Offset: dx={dx:.2f}, dy={dy:.2f}, angle={angle:.2f}°")
"""