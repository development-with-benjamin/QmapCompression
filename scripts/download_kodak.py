import urllib.request
import sys
import os

if len(sys.argv) != 2:
    print("Usage python download_kodak.py <path_to_store_images>")
    exit()

base_path = sys.argv[1]
if not os.path.isdir(base_path):
    print("Usage python download_kodak.py <path_to_store_images>")
    exit()

print("start scraping.")
for im in range(1,25):
	im_path = base_path + str(im)+'.png'
	im_url = 'http://r0k.us/graphics/kodak/kodak/kodim' + str(im).zfill(2) + '.png'
	try:
		urllib.request.urlretrieve(im_url, im_path)
		print ("Image Nr." + str(im) + " is saved.") 
	except:
		print ("Image Nr." + str(im) + " isn't saved.")
		pass
	
print("end scraping.")
saved_images = len([file for file in os.listdir(base_path) if os.path.isfile(file)])	
print(f"saved {saved_images} images")