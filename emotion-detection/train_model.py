import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
from collections import Counter
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import sys
import time

# Constants
IMG_SIZE = 224  # MobileNetV3 input size (upscaled from FER2013's 48x48)
BATCH_SIZE = 32  # Good balance for grayscale emotion data
EPOCHS = 25     # Sufficient for emotion recognition
NUM_CLASSES = 8  # FER+ has 8 emotion classes
MODEL_PATH = "/kaggle/working/fer_mobilenet_model.pth"
DATA_DIR = r"/kaggle/input/fer2013new/FER/FER+"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class MobileNetV3Model(nn.Module):
    """
    MobileNetV3Large model adapted for FER+ dataset
    """
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super(MobileNetV3Model, self).__init__()

        # Load MobileNetV3Large with pre-trained weights
        self.backbone = models.mobilenet_v3_large(pretrained=pretrained)

        # Get the number of features from the backbone
        num_features = self.backbone.classifier[0].in_features

        # Replace the classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the custom classifier weights"""
        for m in self.backbone.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def freeze_backbone(self):
        """Freeze backbone layers for initial training"""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone layers for fine-tuning"""
        for param in self.backbone.features.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)

def load_data_from_csv():
    """
    Load image paths and labels from the fer2013new.csv file
    """
    train_images = []
    train_labels = []
    valid_images = []
    valid_labels = []
    test_images = []
    test_labels = []

    # Load the CSV file with labels
    csv_path = r"/kaggle/input/fer2013new/fer2013new.csv"

    print("Loading data from CSV...")
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip header
        print(f"CSV header: {header}")

        for row_idx, row in enumerate(reader):
            if len(row) < 10:  # Ensure we have enough columns
                continue

            usage = row[0]
            image_name = row[1]

            # Skip rows with empty image names
            if not image_name:
                continue

            try:
                # Extract the emotion with the highest vote count
                emotion_votes = [int(vote) for vote in row[2:10]]
                emotion_label = np.argmax(emotion_votes)

                # Skip samples with very low confidence (sum of votes < 3)
                if sum(emotion_votes) < 3:
                    continue

                if usage == 'Training':
                    img_path = os.path.join(DATA_DIR, 'FER2013Train', image_name)
                    if os.path.exists(img_path):
                        train_images.append(img_path)
                        train_labels.append(emotion_label)
                elif usage == 'PublicTest':
                    img_path = os.path.join(DATA_DIR, 'FER2013Valid', image_name)
                    if os.path.exists(img_path):
                        valid_images.append(img_path)
                        valid_labels.append(emotion_label)
                elif usage == 'PrivateTest':
                    img_path = os.path.join(DATA_DIR, 'FER2013Test', image_name)
                    if os.path.exists(img_path):
                        test_images.append(img_path)
                        test_labels.append(emotion_label)

            except (ValueError, IndexError) as e:
                print(f"Error processing row {row_idx}: {e}")
                continue

    return (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels)

def calculate_class_weights(labels):
    """
    Calculate class weights for imbalanced dataset
    """
    # Count samples per class
    class_counts = Counter(labels)
    print("Class distribution:", class_counts)

    # Calculate class weights
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )

    # Convert to tensor for PyTorch
    weight_tensor = torch.FloatTensor(weights)
    print("Class weights:", {i: weights[i] for i in range(len(weights))})

    return weight_tensor

class FERDataset(Dataset):
    """
    Custom dataset for the FER+ dataset with proper preprocessing for grayscale images
    """
    def __init__(self, image_paths, labels, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment

        # For grayscale FER2013 images converted to RGB, use modified normalization
        # We'll normalize each channel (R=G=B) with grayscale statistics
        self.base_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            # Use grayscale mean/std replicated across 3 channels for pretrained models
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Augmentation transforms (lighter for grayscale emotion data)
        self.augment_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),  # Smaller crop range for 48x48 source
            transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),  # Smaller rotation for faces
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Small translation
            # Don't use ColorJitter on grayscale - use brightness/contrast instead
            transforms.Lambda(lambda x: transforms.functional.adjust_brightness(x, 1 + np.random.uniform(-0.1, 0.1))),
            transforms.Lambda(lambda x: transforms.functional.adjust_contrast(x, 1 + np.random.uniform(-0.1, 0.1))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # Load grayscale image
            img = Image.open(image_path).convert('L')

            # Convert grayscale to RGB by duplicating channels
            # This is necessary for pretrained models that expect 3 channels
            img = img.convert('RGB')

            # Apply transforms
            if self.augment:
                img = self.augment_transform(img)
            else:
                img = self.base_transform(img)

            return img, label

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a zero tensor with proper normalization and correct label
            zero_img = torch.zeros((3, IMG_SIZE, IMG_SIZE))
            # Apply same normalization to zero tensor
            zero_img = (zero_img - 0.5) / 0.5  # Normalize to [-1, 1] range
            return zero_img, label

def create_data_loaders():
    """
    Create training and validation data loaders
    """
    # Load data from CSV
    (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels) = load_data_from_csv()

    print(f"Loaded {len(train_images)} training images")
    print(f"Loaded {len(valid_images)} validation images")
    print(f"Loaded {len(test_images)} test images")

    # Verify we have data
    if len(train_images) == 0 or len(valid_images) == 0:
        raise ValueError("No training or validation data found! Check your file paths and CSV format.")

    # Calculate class weights for the training data
    class_weights = calculate_class_weights(train_labels)

    # Create datasets
    train_dataset = FERDataset(train_images, train_labels, augment=True)
    valid_dataset = FERDataset(valid_images, valid_labels, augment=False)
    test_dataset = FERDataset(test_images, test_labels, augment=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, valid_loader, test_loader, class_weights

def print_progress_bar(current, total, start_time, metrics=None, bar_length=30):
    """
    Print a progress bar
    """
    percent = current / total
    filled_length = int(bar_length * percent)
    bar = '=' * filled_length + '>' + '.' * (bar_length - filled_length - 1)

    # Calculate ETA
    elapsed_time = time.time() - start_time
    if current > 0:
        eta = elapsed_time * (total - current) / current
        eta_str = f"{int(eta)}s"
    else:
        eta_str = "?"

    # Format metrics
    metrics_str = ""
    if metrics:
        metrics_str = " - ".join([f"{k}: {v}" for k, v in metrics.items()])
        metrics_str = " - " + metrics_str

    # Print progress bar
    sys.stdout.write(f'\r{current}/{total} [{bar}] - ETA: {eta_str}{metrics_str}')
    sys.stdout.flush()

    if current == total:
        elapsed_str = f"{int(elapsed_time)}s"
        sys.stdout.write(f'\r{current}/{total} [{bar}] - {elapsed_str}{metrics_str}\n')
        sys.stdout.flush()

def train_model():
    """
    Train the MobileNetV3 model on the FER+ dataset
    """
    # Setup data loaders
    train_loader, valid_loader, test_loader, class_weights = create_data_loaders()

    # Create model
    try:
        model = MobileNetV3Model(num_classes=NUM_CLASSES, pretrained=True).to(device)
        print("✓ MobileNetV3 loaded successfully")
    except Exception as e:
        print(f"Failed to load MobileNetV3: {e}")
        print("Training without pretrained weights...")
        model = MobileNetV3Model(num_classes=NUM_CLASSES, pretrained=False).to(device)
        print("Model created with MobileNetV3Large backbone")

    # Setup loss function with class weights (reduced weight effect)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Setup optimizer with lower learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Early stopping parameters
    best_val_acc = 0.0
    patience_counter = 0
    early_stopping_patience = 8

    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Phase 1: Train only classifier layers
    print("Phase 1: Training only classifier layers...")
    model.freeze_backbone()

    for epoch in range(10):  # First 10 epochs with frozen backbone
        print(f'Epoch {epoch+1}/10 (Classifier only)')

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

            # Update progress bar
            current_loss = train_loss / (batch_idx + 1)
            current_acc = train_correct / train_total
            metrics = {
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.4f}'
            }
            print_progress_bar(batch_idx + 1, len(train_loader), start_time, metrics)

        train_acc = 100. * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        val_acc = 100. * val_correct / val_total

        # Store metrics
        train_losses.append(train_loss/len(train_loader))
        val_losses.append(val_loss/len(valid_loader))
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Print epoch results
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(valid_loader):.4f}, Val Acc: {val_acc:.2f}%')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            patience_counter = 0
            print(f'New best validation accuracy: {val_acc:.2f}%')
        else:
            patience_counter += 1

    # Phase 2: Fine-tune entire network
    print("\nPhase 2: Fine-tuning entire network...")
    model.unfreeze_backbone()

    # Lower learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    remaining_epochs = EPOCHS - 10
    for epoch in range(remaining_epochs):
        print(f'Epoch {epoch+11}/{EPOCHS} (Full network)')

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

            # Update progress bar
            current_loss = train_loss / (batch_idx + 1)
            current_acc = train_correct / train_total
            metrics = {
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.4f}'
            }
            print_progress_bar(batch_idx + 1, len(train_loader), start_time, metrics)

        train_acc = 100. * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        val_acc = 100. * val_correct / val_total

        # Store metrics
        train_losses.append(train_loss/len(train_loader))
        val_losses.append(val_loss/len(valid_loader))
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Print epoch results
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(valid_loader):.4f}, Val Acc: {val_acc:.2f}%')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            patience_counter = 0
            print(f'New best validation accuracy: {val_acc:.2f}%')
        else:
            patience_counter += 1

        scheduler.step()

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after epoch {epoch+11}")
            break

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
    return model, test_loader

def evaluate_model(model, test_loader):
    """
    Evaluate the trained model
    """
    print("\nEvaluating model...")

    # Define emotion class names
    emotion_classes = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']

    # Load best model
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Predict on test data
    predictions = []
    true_labels = []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

            _, predicted = torch.max(output, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(target.cpu().numpy())

    # Convert to numpy arrays
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Calculate metrics
    accuracy = np.mean(predictions == true_labels)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")

    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=emotion_classes))

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_classes, yticklabels=emotion_classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    return accuracy, predictions, true_labels

def validate_dataset():
    """
    Validate dataset structure and sample loading for FER2013 grayscale images
    """
    print("Validating FER2013 dataset structure...")

    try:
        (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels) = load_data_from_csv()

        print(f"✓ Training set: {len(train_images)} images")
        print(f"✓ Validation set: {len(valid_images)} images")
        print(f"✓ Test set: {len(test_images)} images")

        if len(train_images) == 0 or len(valid_images) == 0:
            print("✗ No training or validation images found!")
            return False

        # Check label distribution
        print(f"Training label distribution: {Counter(train_labels)}")
        print(f"Validation label distribution: {Counter(valid_labels)}")

        # Test loading a few images and check if they're grayscale FER2013 format
        test_images_sample = train_images[:3] + valid_images[:3]
        failed_count = 0

        for img_path in test_images_sample:
            try:
                img = Image.open(img_path)
                print(f"Sample image {os.path.basename(img_path)}: size={img.size}, mode={img.mode}")

                # Convert to grayscale then RGB as the model expects
                img = img.convert('L').convert('RGB')

                if img.size[0] == 0 or img.size[1] == 0:
                    failed_count += 1
                    print(f"  ✗ Invalid dimensions: {img.size}")
                else:
                    print(f"  ✓ Loaded successfully: {img.size}")

            except Exception as e:
                print(f"  ✗ Failed to load {img_path}: {e}")
                failed_count += 1

        if failed_count == 0:
            print("✓ All sample images loaded successfully")
            return True
        else:
            print(f"✗ {failed_count} sample images failed to load")
            return False

    except Exception as e:
        print(f"✗ Dataset validation failed: {e}")
        return False

if __name__ == "__main__":
    # Check data directory
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
        exit(1)

    # Validate dataset
    if not validate_dataset():
        print("Dataset validation failed. Please check your data.")
        exit(1)

    # Train the model
    trained_model, test_loader = train_model()

    # Evaluate the model
    accuracy, predictions, true_labels = evaluate_model(trained_model, test_loader)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")