import torch
from torchvision import models, transforms
from PIL import Image
import os
import argparse
import json
import torch.nn as nn
import time

# Argument parser
parser = argparse.ArgumentParser(description='Language Classification Inference')
parser.add_argument('--main_folder', metavar='main_folder', type=str,
                    help='Path to the main image folder containing images')
parser.add_argument('--model_path', metavar='model_path', type=str,
                    help='Path to the trained model file')
parser.add_argument('--single_image', metavar='single_image', type=str, default=None,
                    help='Path to a single image for prediction')
parser.add_argument('--batch_size', type=int, default=1000,
                    help='Batch size for processing images in batches')
parser.add_argument('--accuracy', action='store_true', help='Calculate accuracy')
args = parser.parse_args()

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')



print(args.model_path)



# Define the data transforms
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path,model):
    
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    # print(checkpoint.keys())
    # print(checkpoint)
    # s = checkpoint["odict_keys"]
    s = checkpoint
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = data_transforms(image)
    return image.to(device)

def predict_image(image_path, model_infer):
    image = preprocess_image(image_path).unsqueeze(0)
    with torch.no_grad():
        output = model_infer(image)
        _, pred = torch.max(output, 1)
    return pred.item()


def predict_images(image_paths, batch_size, model_infer, accuracy_file_path='batch_accuracies.txt'):

    
    predictions = {}
    print(f'Total number of images: {len(image_paths)}')
    total_batches = (len(image_paths) + batch_size - 1) // batch_size  # Calculate total number of batches
    print(f'Total number of batches: {total_batches}')

    with open(accuracy_file_path, 'w') as f:
        for i in range(total_batches):
            batch_paths = image_paths[i * batch_size:(i + 1) * batch_size]
            images = [preprocess_image(image_path) for image_path in batch_paths]
            images = torch.stack(images, dim=0)  # Stack images into a batch
            with torch.no_grad():
                outputs = model_infer(images)
                _, preds = torch.max(outputs, 1)
            
            batch_predictions = {image_path: pred.item() for image_path, pred in zip(batch_paths, preds)}
            predictions.update(batch_predictions)

            # # Calculate accuracy for the current batch
            # batch_accuracy = calculate_accuracy(batch_predictions)
            # print(f'Batch {i + 1}/{total_batches} - Batch Accuracy: {batch_accuracy * 100:.2f}%')

            # # Write batch accuracy to file
            # f.write(f'Batch {i + 1}/{total_batches} - Batch Accuracy: {batch_accuracy * 100:.2f}%\n')

    return predictions


def calculate_accuracy(predictions):
    correct_predictions = 0
    total_images = len(predictions)
    for image_path, pred in predictions.items():
        true_label = 0 if 'beng_' in image_path else 1  # Assuming 'beng_' and 'eng_' prefix for labels
        if pred == true_label:
            correct_predictions += 1
    accuracy = correct_predictions / total_images if total_images else 0
    return accuracy


    
def main():
    main_folder = args.main_folder
    model_path = args.model_path

    # Initialize the model
    model = models.resnet101(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    start_time = time.time()

    model = load_model(model_path, model)

    print(f'Model loaded in {time.time() - start_time:.2f} seconds')





    if args.single_image is not None:
        single_image_path = args.single_image
        print(f'Single Image Prediction: {single_image_path}')
        prediction = predict_image(single_image_path, model)
        output = "bengali" if prediction == 0 else "english"
        print(f'Prediction: {output}')

    else:
        # Perform inference on multiple images from main_folder
        image_paths = [os.path.join(main_folder, image_name) for image_name in os.listdir(main_folder)
                       if os.path.isfile(os.path.join(main_folder, image_name))
                       and image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
        
        # Batch processing
        batch_size = len(image_paths)
        predictions = predict_images(image_paths, batch_size, model)
        print('Multiple Image Predictions:')
        for image_path, prediction in predictions.items():
            output = "bengali" if prediction == 0 else "english"
            print(f'{os.path.basename(image_path)}: {output}')

        # Save predictions to a JSON file
        output_file = 'temp/predictions.json'
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=4)
        print(f'Predictions saved to {output_file}')

        # # Calculate accuracy if specified
        # if args.accuracy:
        #     accuracy = calculate_accuracy(predictions)
        #     print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()
