import numpy as np

import os

import errno

import argparse

import cv2
import tensorflow as tf


def batch_run(fun, dict_data_in, ot, sz_bh): # done ----------------------------------------------

    batches = int(len(ot)/ sz_bh)

    s1=0
    e1=0

    for i in range(batches):
        s1=i * sz_bh
        e1=(i + 1) * sz_bh

        dict_batchdata={}

        for k1, v1 in dict_data_in.items():
        	dict_batchdata[k1]=v1[s1:e1]

        ot[s1:e1] = fun(dict_batchdata)


    if e1 < len(ot):
      dict_batchdata={}
      for k1, v1 in dict_data_in.items():
        dict_batchdata[k1]=v1[s1:e1]
      ot[e1:] = fun(dict_batchdata)


def get_img(img,bounding_box):  # done----------------------------------------------------------------------------
    sx1, sy1, ex1, ey1 = bounding_box
    img = img[sy1:ey1, sx1:ex1]
    return img

def fetch_imagepatch(img, box_in, seg_shape): # done------------------------------------------------------------

    bounding_box = np.array(box_in)
    #print(bounding_box)
    #print(seg_shape)
    if seg_shape is not None:
        n_width = (float(seg_shape[1]) / seg_shape[0]) * bounding_box[3]
        bounding_box[0] =bounding_box[0]- (n_width - bounding_box[2]) / 2
        bounding_box[2] = n_width


    bounding_box[2:]=bounding_box[2:]+ bounding_box[:2]
    bounding_box = bounding_box.astype(np.int)


    bounding_box[:2] = np.maximum( bounding_box[:2],0)
    bounding_box[2:] = np.minimum(bounding_box[2:], np.asarray(img.shape[:2][::-1]) - 1)

    if np.any(bounding_box[:2] >= bounding_box[2:]):
        return None


    img=get_img(img,bounding_box)

    img = cv2.resize(img, tuple(seg_shape[::-1]))
    return img


class Encoder(object):  # done----------------------------------------------------------------------------------

    def __init__(self, check_file, in_name="images",name_otput="features"):

        self.session = tf.Session()
        #self.session_encoder456 = tf.session_encoder()

        with tf.gfile.GFile(check_file, "rb") as file_handle:

            graph_def = tf.GraphDef()

            graph_def.ParseFromString(file_handle.read())


        tf.import_graph_def(graph_def, name="net")


        self.in_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % in_name)

        self.in_var123 = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % in_name)


        self.out_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % name_otput)
        self.out_var123 = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % name_otput)

        assert len(self.out_var.get_shape()) == 2
        assert len(self.in_var.get_shape()) == 4

        self.img_sh = self.in_var.get_shape().as_list()[1:]
        #self.img_sh1 = self.in_var.get_shape().as_list()[1:]

        self.dimension_feature = self.out_var.get_shape().as_list()[-1]
        #self.dimension_feature1 = self.out_var.get_shape().as_list()[-1]

    def class_log(self,msg):
    	print("class log to be printed ",msg)

    def __call__(self, x_data, sz_bh=32):
        out1 = np.zeros((len(x_data), self.dimension_feature), np.float32)
        #class_log("class instance ")

        batch_run(
            lambda x: self.session.run(self.out_var, feed_dict=x),
            {self.in_var: x_data}, out1, sz_bh)

        return out1


def box_encoder_fetch(file_model, in_name="images",name_otput="features", sz_bh=32):  # done ---------------------------------------------------
    encoderImage = Encoder(file_model, in_name, name_otput)

    img_sh = encoderImage.img_sh

    def encoder_log():
    	print("encoder log")

    def get_patches(imagesPatches,image, total_boxes):

        for bx in total_boxes:
            patch = fetch_imagepatch(image, bx, img_sh[:2])

            if patch is None:

                patch = np.random.uniform(0., 255., img_sh).astype(np.uint8)



            imagesPatches.append(patch)

        #encoder_log() # to be done-------------------------------------
        imagesPatches = np.asarray(imagesPatches)
        return imagesPatches


    def encoder(image, total_boxes):
      imagesPatches1=[]
      imagesPatches =get_patches(imagesPatches1,image, total_boxes)
      return encoderImage(imagesPatches, sz_bh)

    return encoder


def get_out_detection(in_detection,file_img):  # done----------------------------------------------------------------------------------------------------------

  out_detections = []

  inxices_frame1 = in_detection[:, 0]
  inxices_frame=inxices_frame1.astype(np.int)
  
  frame_indMax1 = inxices_frame.astype(np.int)
  frame_indMax= frame_indMax1.max()

  frame_indMin1 = inxices_frame.astype(np.int)
  frame_indMin = frame_indMin1.min()


  for idxFrame in range(frame_indMin, frame_indMax + 1):

      mask = inxices_frame == idxFrame
      rows1 = in_detection[mask]

      if idxFrame not in file_img:
          continue

      img_bgr = cv2.imread(file_img[idxFrame], cv2.IMREAD_COLOR)

      features = encoder(img_bgr, rows1[:, 2:6].copy())

      out_detections = out_detections+ [np.r_[(row, feature)] for row, feature in zip(rows1, features)]

  return out_detections


def produce_detection(encoder, dir_mot, dir_out, dir_detection=None):  # done-------------------------------------------------------------------

    if dir_detection is None:
        dir_detection = dir_mot

    print("exception not handled")


    print("processing from directory")

    for sequence in os.listdir(dir_mot):



        dir_img=  os.path.join(dir_mot, sequence,"img1")



        file_detection = os.path.join(dir_detection, sequence, "det/det.txt")

        exception_handling()

        in_detection = np.loadtxt(file_detection, delimiter=',')

        file_img = {
            int(os.path.splitext(f)[0]): os.path.join(dir_img, f)
            for f in os.listdir(dir_img)}



        out_detections=get_out_detection(in_detection,file_img)

        output_filename = os.path.join(dir_out, "%s.npy" % sequence)

        np.save(output_filename, np.asarray(out_detections), allow_pickle=False)


def parse_args(): # worked---------------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")

    parser.add_argument(
        "--dir_out", help=" directory for output creation if not created"
        " exist.", default="detections")
    parser.add_argument(
        "--model",
        default="resources/networks/mars-small128.pb",
        help="Path for FIGP .")   # FIGP =freezed inference Graph protobuf

    ''' path=Path to custom detections. Defaults to "
        "standard MOT detections Directory structure should be the default "
        "MOTChallenge structure: [sequence]/det/det.txt'''

    parser.add_argument(
        "--dir_detection", help="path", default=None)
    parser.add_argument(
        "--dir_mot", help="Path to  MD",  #MD= MOTChallenge directory (train or test)
        required=True)



    return parser.parse_args()

def exception_handling():

	print("exception handling to be done ")




def main():  # done---------------------------------------------------------------------------------------------------------------------
    args_bundle = parse_args()

    encoder_fetched = box_encoder_fetch(args_bundle.model, sz_bh=32)


    print("detection started -----------------------------------------")

    exception_handling()


    produce_detection(encoder_fetched, args_bundle.dir_mot, args_bundle.dir_out,args_bundle.dir_detection)

    print("detection ended -----------------------------------------")



if __name__ == "__main__":
    main()
