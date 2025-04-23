
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


    def apply_offset_and_rotation(self,
            image, x: float =0,
            y: float =0,
            angle: float =0,
            rotate_first: bool = True,
            invert_y: bool = False
    ):
        """
        Applies translation and rotation to the image with options.

        Args:
            image: Input image (numpy array).
            x: Shift along X-axis in pixels.
            y: Shift along Y-axis in pixels.
            angle: Rotation angle in degrees (counter-clockwise).
            rotate_first: If True, rotate then shift. If False, shift then rotate.
            invert_y: If True, reverse direction of y (positive goes up).

        Returns:
            Transformed image.
        """
        rows, cols = image.shape[:2]
        center = (cols / 2, rows / 2)

        # Adjust Y-direction if requested
        if invert_y:
            y = -y

        # Get rotation matrix
        R = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Get translation matrix as an affine matrix
        T = np.array([[1, 0, x], [0, 1, y]], dtype=np.float32)

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

    def _display_results(self, template_img, corrected_img, kp1, kp2, roi, M, matches, top_left, h, w):
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

        # Text: Apply the affine transformation to the corners (rotate the template)
        transformed_corners = cv2.transform(template_corners, M).astype(int)

        # Text: Offset by top-left corner to match position in full corrected image
        transformed_corners += np.array(top_left, dtype=int).reshape(1, 1, 2)

        # Text: Calculate the top-left corner of the rotated bounding box
        top_left_rot = transformed_corners[0][0]  # First corner of the rotated polygon

        # Text: Calculate angle of rotation for text (using two consecutive corners)
        x = transformed_corners[1][0][0] - transformed_corners[0][0][0]  # X diff
        y = transformed_corners[1][0][1] - transformed_corners[0][0][1]  # Y diff
        angle = np.arctan2(y, x) * 180 / np.pi  # Convert angle to degrees

        # Text: Rotate the text and place it above the top-left corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text = "Template"

        # Get the text size to center it properly above the corner
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

        # Define the position above the top-left corner (offset by 10 pixels)
        text_pos = (top_left_rot[0] - text_width // 2, top_left_rot[1] - text_height - 10)

        # Text: Add the rotated text to the overlay without rotating the entire image
        # Rotate just the text, not the whole overlay
        M_text = cv2.getRotationMatrix2D((float(top_left_rot[0]), float(top_left_rot[1])), angle,
                                         1)  # Rotation matrix
        rotated_text_overlay = overlay.copy()  # Copy overlay to avoid modifying it during rotation

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

        # Finally, you can visualize both: the template rotated and the rotated text
        plt.figure(figsize=(6, 6))
        plt.title("Overlay: Corrected Image + Transformed Template + ROI Box")
        plt.imshow(cv2.cvtColor(rotated_text_overlay, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show(block=True)

        return


    def calculate_offset_and_rotation(self, template_img, corrected_img, visualize: bool =True):
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
            M = [ cosθ  -sinθ   x ]
                [ sinθ   cosθ   y ]
        """
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

        if M is None:
            raise ValueError("Could not estimate affine transformation.")

        x = M[0, 2] + top_left[0]  # adjust with offset of ROI
        y = M[1, 2] + top_left[1]
        #dx = top_left[0]
        #dy = top_left[1]
        angle = degrees(atan2(M[1, 0], M[0, 0]))

        # Step 6: Visualization
        if visualize:
            self._display_results(template_img, corrected_img, kp1, kp2, roi, M, matches, top_left, h, w)


        return x, y, angle


visualize = False
rc = RotationCorrection()
rc_data = RotationCorrectionData()
# Load image and teach template
template, dut = rc.load_image("image_1.png")
# Store template in data structure
rc_data.add_template_item("DUT_1","small",TemplateItem(22, template, 1.0, 2.0, 3.0))
# Store image in data structure
rc_data.add_correction_item("DUT1","small",50.0, CorrectionItem(50.0, dut, 0, 0, 0))
# Apply shift and rotation
shifted = rc.apply_offset_and_rotation(image= dut, x= 5, y= 7, angle= 0, rotate_first= False, invert_y= False)
rotated = rc.apply_offset_and_rotation(image= dut,x= 19, y= -50, angle= 1, rotate_first= False, invert_y= False)

# Calculate the shift and rotation original dut image
x0, y0,angle0 = rc.calculate_offset_and_rotation(template_img= template, corrected_img= dut, visualize= visualize)

# Calculate the shift and rotation of shifted dut image
x_s, y_s,angle_s = rc.calculate_offset_and_rotation(template_img= template, corrected_img= shifted, visualize= visualize)
corrected_shift = rc.apply_offset_and_rotation(image= shifted,x=x0 - x_s, y= y0 - y_s, angle= angle_s, rotate_first= False, invert_y= False)
cx_s, cy_s,c_angle_s = rc.calculate_offset_and_rotation(template_img= template, corrected_img= corrected_shift, visualize= visualize)
print(f'Shifted corrected image: delta x:{cx_s -x0}, delta y: {cy_s - y0}, angolo: {angle_s}')

# Calculate the shift and rotation rotated dut image
x_r1, y_r1,angle_r1 = rc.calculate_offset_and_rotation(template_img= template, corrected_img= rotated, visualize= visualize)
print(f'Rotated image: delta x:{x_r1 -x0}, delta y: {y_r1 - y0}, angolo: {angle_r1}')
corrected_rotation = rc.apply_offset_and_rotation(image= rotated,x= 0, y= 0, angle= angle_r1, rotate_first= False, invert_y= False)
x_r1, y_r1,angle_r1 = rc.calculate_offset_and_rotation(template_img= template, corrected_img= corrected_rotation, visualize= visualize)
print(f'Rotated corrected image: delta x:{x_r1 -x0}, delta y: {y_r1 - y0}, angolo: {angle_r1}')
corrected_shift = rc.apply_offset_and_rotation(image= corrected_rotation,x= -(x_r1 - x0), y= -(y_r1 - y0), angle= 0, rotate_first= False, invert_y= False)
x_r1, y_r1,angle_r1 = rc.calculate_offset_and_rotation(template_img= template, corrected_img= corrected_shift, visualize= visualize)
print(f'Rotated and shifted corrected image: delta x:{x_r1 - x0}, delta y: {y_r1 - y0}, angolo: {angle_r1}')




print(rc)
