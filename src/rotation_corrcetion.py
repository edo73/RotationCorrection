
import cv2
import numpy as np
from math import degrees, atan2
import matplotlib.pyplot as plt
import logging
from src.rotation_correction_status import RotationCorrectionData, TemplateItem, CorrectionItem


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


    def shift_and_rotate_image(self,
            image, dx: float =0,
            dy: float =0,
            angle: float =0,
            rotate_first: bool = True,
            invert_y: bool = False
    ):
        """
        Applies translation and rotation to the image with options.

        Args:
            image: Input image (numpy array).
            dx: Shift along X-axis in pixels.
            dy: Shift along Y-axis in pixels.
            angle: Rotation angle in degrees (counter-clockwise).
            rotate_first: If True, rotate then shift. If False, shift then rotate.
            invert_y: If True, reverse direction of dy (positive goes up).

        Returns:
            Transformed image.
        """
        rows, cols = image.shape[:2]
        center = (cols / 2, rows / 2)

        # Adjust Y-direction if requested
        if invert_y:
            dy = -dy

        # Get rotation matrix
        R = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Get translation matrix as an affine matrix
        T = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)

        # Combine transforms
        if rotate_first:
            # First rotate, then shift: T * R
            combined = T @ np.vstack([R, [0, 0, 1]])  # Add 3rd row for matrix mult
        else:
            # First shift, then rotate: R * T
            combined = R @ np.vstack([T, [0, 0, 1]])

        # Extract the top 2 rows for warpAffine
        M = combined[:2, :]

        # Apply the transformation
        transformed = cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_LINEAR)

        return transformed

    def _find_template_location(self, corrected_img, template_img):
        """
        Uses template matching to locate the region in the corrected image that best matches the template.
        It does not handle rotation at all, and it’s not designed for sub-pixel accuracy or complex misalignments.
        """
        result = cv2.matchTemplate(corrected_img, template_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < 0.5:
            logging.warning("Template matching confidence too low")

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
            plt.show(block=True)

            # Warp the template image with the estimated transformation
            warped = cv2.warpAffine(template_img, M, (w, h))
            plt.figure(figsize=(12, 6))
            plt.title("Warp the template image with the estimated transformation")
            plt.imshow(warped, cmap='gray')
            plt.axis("off")
            plt.show(block=True)

            # Convert to color for overlay visualization
            if len(corrected_img.shape) == 2:
                corrected_color = cv2.cvtColor(corrected_img, cv2.COLOR_GRAY2BGR)
            else:
                corrected_color = corrected_img.copy()

            if len(warped.shape) == 2:
                warped_color = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
            else:
                warped_color = warped

            # Create overlay
            overlay = corrected_color.copy()

            # Define ROI bounds
            y1, y2 = top_left[1], top_left[1] + h
            x1, x2 = top_left[0], top_left[0] + w

            # Clip if ROI exceeds image boundaries
            y2 = min(y2, overlay.shape[0])
            x2 = min(x2, overlay.shape[1])
            warped_color = warped_color[:y2 - y1, :x2 - x1]

            # Blend only in the ROI
            roi_overlay = overlay[y1:y2, x1:x2]
            blended = cv2.addWeighted(roi_overlay, 0.5, warped_color, 0.5, 0)
            overlay[y1:y2, x1:x2] = blended

            # Draw green rectangle around ROI
            template_corners = np.array([
                [[0, 0]],
                [[template_img.shape[1], 0]],
                [[template_img.shape[1], template_img.shape[0]]],
                [[0, template_img.shape[0]]]
            ], dtype=np.float32)  # shape: (4, 1, 2)

            # Apply the affine transformation to the corners
            transformed_corners = cv2.transform(template_corners, M).astype(int)

            # Offset by top-left corner to match position in full corrected image
            transformed_corners += np.array(top_left, dtype=int).reshape(1, 1, 2)

            # Draw the polygon on the overlay
            cv2.polylines(
                overlay,
                [transformed_corners],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2
            )

            # Step 1: Calculate the top-left corner of the rotated bounding box
            # Ensure that the result is a tuple of integers (x, y)
            top_left_rot = tuple(transformed_corners[0][0].astype(int))  # Ensure it's a tuple (x, y)

            # Step 2: Print the type and value to verify it's correct
            print(f"top_left_rot: {top_left_rot}, type: {type(top_left_rot)}")

            # Check if top_left_rot is indeed a tuple with two integer values
            if not isinstance(top_left_rot, tuple) or len(top_left_rot) != 2:
                print(f"Error: top_left_rot is not a tuple with two values. Actual value: {top_left_rot}")
            else:
                print(f"top_left_rot is correctly a tuple: {top_left_rot}")

            # Step 3: Calculate angle of rotation for text (using two consecutive corners)
            dx = transformed_corners[1][0][0] - transformed_corners[0][0][0]  # X diff
            dy = transformed_corners[1][0][1] - transformed_corners[0][0][1]  # Y diff
            angle = np.arctan2(dy, dx) * 180 / np.pi  # Convert angle to degrees

            # Step 4: Rotate the text and place it above the top-left corner
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            text = "Template"

            # Get the text size to center it properly above the corner
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

            # Define the position above the top-left corner (offset by 10 pixels)
            text_pos = (top_left_rot[0] - text_width // 2, top_left_rot[1] - text_height - 10)

            # Step 5: Ensure rotation center is an integer tuple (recheck)
            rotation_center = (int(top_left_rot[0]), int(top_left_rot[1]))  # Ensure both x and y are integers

            # Step 6: Apply the rotation matrix to the text
            M_text = cv2.getRotationMatrix2D(rotation_center, angle, 1)  # Rotation matrix

            rotated_text_overlay = overlay.copy()  # Create a copy for overlaying the text

            # Put the text in the rotated position
            cv2.putText(
                rotated_text_overlay,
                text,
                text_pos,
                font,
                font_scale,
                (0, 255, 0),
                font_thickness,
                cv2.LINE_AA
            )

            # Step 7: Apply the rotation matrix only to the text (not the entire image)
            final_overlay = cv2.warpAffine(rotated_text_overlay, M_text,
                                           (rotated_text_overlay.shape[1],
                                            rotated_text_overlay.shape[0]))  # Only rotate text

            # Show result
            plt.figure(figsize=(6, 6))
            plt.title("Overlay: Corrected Image + Transformed Template + ROI Box with Rotated Text")
            plt.imshow(cv2.cvtColor(final_overlay, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show(block=True)

        return dx, dy, angle



rc = RotationCorrection()
rc_data = RotationCorrectionData()

template, dut = rc.load_image("image_1.png")
rc_data.add_template_item("DUT_1","small",TemplateItem(22, template, 1.0, 2.0, 3.0))
rc_data.add_correction_item("DUT1","small",50.0, CorrectionItem(50.0, dut, 0, 0, 0))
shifted = rc.shift_and_rotate_image(image= dut, dx= 5,dy= 7, angle= 0, rotate_first= False, invert_y= False)
rotated = rc.shift_and_rotate_image(image= dut, dx= 5,dy= 7, angle= 10, rotate_first= False, invert_y= False)
dx1,dy1,angle1 = rc.calculate_offset_and_rotation(template_img= template, corrected_img= dut, visualize= True)
dx2,dy2,angle2 = rc.calculate_offset_and_rotation(template_img= template, corrected_img= shifted, visualize= True)
dx3,dy3,angle3 = rc.calculate_offset_and_rotation(template_img= template, corrected_img= rotated, visualize= True)


print(rc)
# Assuming you already have the dataclass setup and images loaded
"""template_item = rc.get_template_item(name, item_id)
correction_item = rc.get_correction_item(name, temp, item_id)

# Make sure images are grayscale
template_gray = cv2.cvtColor(template_item.image, cv2.COLOR_BGR2GRAY)
correction_gray = cv2.cvtColor(correction_item.image, cv2.COLOR_BGR2GRAY)

dx, dy, angle = calculate_offset_and_rotation(template_gray, correction_gray)
print(f"Offset: dx={dx:.2f}, dy={dy:.2f}, angle={angle:.2f}°")
"""