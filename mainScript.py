from PIL import Image
import numpy

# read the image in
image = Image.open("earth_color.png")
image_grey = image.convert("L")
image_blackwhite = image_grey.point(lambda x: 0 if x < 128 else 255, "1")

processed_image = image_blackwhite.resize((200, 200))
processed_image.show()


image_matrix = numpy.array(processed_image)
image_array = image_matrix.flatten()

print(image_array)