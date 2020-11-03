import tflite_runtime.interpreter as tflite
import time
import cv2
import numpy as np
import platform


# Select the TPU shared library
EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


interpreter_coral = tflite.Interpreter(
    model_path="model.tflite",
    experimental_delegates=[
        tflite.load_delegate(EDGETPU_SHARED_LIB, {})
        ])



labels_dict={0:'MASK', 1:'NO MASK'}
color_dict={0:(50,200,50), 1:(50,50,200)}

size = 1
webcam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # FIX: Not very accurate in poor light conditions

input_details = interpreter_coral.get_input_details()
output_details = interpreter_coral.get_output_details()
interpreter_coral.allocate_tensors()



# Preprocess image for tensorflow lite
def preprocess_img(img, IMG_SIZE=160):
    resized_img = cv2.resize(face_img,(IMG_SIZE,IMG_SIZE))
    normalized = resized_img / 255.0
    tensor_img = np.reshape(normalized,(1,IMG_SIZE,IMG_SIZE,3)) #Add axes 0 for tf
    tensor_img = np.vstack([tensor_img])
    return tensor_img


while True:
    t0 = time.time()
    (rval, img) = webcam.read()
    img = cv2.flip(img,1,1)

    faces = face_detector.detectMultiScale(img) # Reduce size for speed

    for f in faces:
        (x, y, w, h) = [v * size for v in f]
        face_img = img[y:y+h, x:x+w]
        
        # Invoke tf-lite classifier
        tensor_img = preprocess_img(img, IMG_SIZE=160)
        interpreter_coral.set_tensor(input_details[0]['index'], tensor_img.astype('float32'))
        interpreter_coral.invoke()
        result = interpreter_coral.get_tensor(output_details[0]['index'])
        label = np.argmax(result,axis=1)[0]
 
        # Draw rectangles and output
        cv2.rectangle(img, (x,y),(x+w,y+h), color_dict[label],2)
        cv2.rectangle(img, (x,y-50),(x+w,y), color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-27),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(img, f'MASK PROB: {result[0][0]*100:.1f}%', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX   ,0.6,(255,255,255),2)
    cv2.putText(img, f'{1/(time.time()-t0):.3f} Hz', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
    cv2.imshow('Mask detection - press q to exit', img)

    # Quit using 'q'
    key = cv2.waitKey(10)
    if key == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()

