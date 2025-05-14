import os
from PIL import Image
import cv2
import numpy as np
import sys
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

WIDTH = 310 // 2
HEIGHT = 568 // 2
size = (WIDTH, HEIGHT)

# Ensure necessary directories exist
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created missing directory: {directory}")

ensure_directory_exists("images/resized/train")
ensure_directory_exists("images/resized/test")

def resize_and_save(img_path, save_path):
    try:
        main_image = Image.open(img_path)
        resized_image = main_image.resize(size, Image.NEAREST)
        resized_image.save(save_path)
    except IOError:
        sys.stderr.write(f"Warning: Could not open file {img_path}\n")

def _reshape_img(arr):
    if arr is None:
        return None
    return arr.flatten()

def _create_data(img_list, label_list):
    inp_arr = []
    for img in img_list:
        img_data = cv2.imread(img)
        if img_data is None:
            sys.stderr.write(f"Warning: Could not read image {img}\n")
            continue
        reshaped_img = _reshape_img(img_data)
        if reshaped_img is not None:
            inp_arr.append(reshaped_img)
    
    if len(inp_arr) == 0:
        sys.stderr.write("Error: No valid images found!\n")
        sys.exit(1)
    
    return np.array(inp_arr), np.array(label_list[:len(inp_arr)])

def train_and_save(train_img_list, label_list, model_name, degree=2):
    try:
        with open(model_name, "rb") as file:
            model, poly = pickle.load(file)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        in_arr, out_arr = _create_data(train_img_list, label_list)
        poly = PolynomialFeatures(degree)
        in_arr_poly = poly.fit_transform(in_arr)
        model = LinearRegression().fit(in_arr_poly, out_arr)
        with open(model_name, "wb") as file:
            pickle.dump((model, poly), file)
    return model, poly

def get_model(model_name):
    try:
        with open(model_name, "rb") as file:
            model, poly = pickle.load(file)
            return model, poly
    except FileNotFoundError:
        sys.stderr.write(f"Error: {model_name} doesn't exist. Train and save a model first.\n")
        sys.exit(1)

if __name__ == "__main__":
    from train_label import train_label, test_label
    
    train_img_list = []
    test_img_list = []
    
    for key in train_label.keys():
        original_path = f"images/Fractured Bone/{key}.jpg"
        resized_path = f"images/resized/train/{key}.jpg"
        if os.path.exists(original_path):
            resize_and_save(original_path, resized_path)
            train_img_list.append(resized_path)
        else:
            sys.stderr.write(f"Warning: {original_path} not found! Skipping...\n")
    
    for key in test_label.keys():
        original_path = f"images/Fractured Bone/{key}.jpg"
        resized_path = f"images/resized/test/{key}.jpg"
        if os.path.exists(original_path):
            resize_and_save(original_path, resized_path)
            test_img_list.append(resized_path)
        else:
            sys.stderr.write(f"Warning: {original_path} not found! Skipping...\n")
    
    train_label_list = list(train_label.values())
    test_label_list = list(test_label.values())
    
    if not train_img_list or not test_img_list:
        sys.stderr.write("Error: Training or testing images not found!\n")
        sys.exit(1)
    
    print("Training started...")
    model, poly = train_and_save(train_img_list, train_label_list, "poly_model")
    print("Training finished...")
    
    train_in_arr, train_out_arr = _create_data(train_img_list, train_label_list)
    test_in_arr, test_out_arr = _create_data(test_img_list, test_label_list)
    
    train_in_poly = poly.transform(train_in_arr)
    test_in_poly = poly.transform(test_in_arr)
    
    print("Training set score: {:.2f}".format(model.score(train_in_poly, train_out_arr)))
    print("Test set score: {:.2f}".format(model.score(test_in_poly, test_out_arr)))
