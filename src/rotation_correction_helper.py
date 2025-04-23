from src.rotation_correction import RotationCorrection
from src.rotation_correction_status import RotationCorrectionData, TemplateItem, CorrectionItem


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
