from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import cv2
import numpy

RESIZE_FACTOR = 1

model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")


# define a video capture object 
vid = cv2.VideoCapture(0) 
  
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
  
    # Convert image to PIL
    img = frame
    image = Image.fromarray(img)
    image.resize((image.width // RESIZE_FACTOR, image.height // RESIZE_FACTOR))



    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # model predicts bounding boxes and corresponding COCO classes
    logits = outputs.logits
    bboxes = outputs.pred_boxes


    # print results
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.8, target_sizes=target_sizes)[0]


    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        label = model.config.id2label[label.item()] 
        x = max(0, round(box[0]))*RESIZE_FACTOR
        y = max(0, round(box[1]))*RESIZE_FACTOR
        x_plus_w = max(0, round(box[2]))*RESIZE_FACTOR 
        y_plus_h = max(0, round(box[3]))*RESIZE_FACTOR
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), (0, 0, 255), 2)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("image", img)
            
        
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 