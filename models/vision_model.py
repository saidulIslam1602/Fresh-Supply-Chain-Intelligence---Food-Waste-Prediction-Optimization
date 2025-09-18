"""
Computer vision model for fresh produce quality assessment
Using EfficientNet-B4 with transfer learning for BAMA's quality control
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FreshProduceVisionModel:
    """
    Computer vision model for fresh produce quality assessment
    Using EfficientNet-B4 with transfer learning
    """
    
    def __init__(self, num_classes: int = 5, device: str = None):
        """
        Initialize vision model
        
        Args:
            num_classes: Number of quality classes (Fresh, Good, Fair, Poor, Spoiled)
            device: CUDA device or CPU
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Load pre-trained EfficientNet-B4
        self.model = models.efficientnet_b4(pretrained=True)
        
        # Modify the classifier for our use case
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        
        self.model = self.model.to(self.device)
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.CenterCrop(380),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Quality labels
        self.quality_labels = ['Fresh', 'Good', 'Fair', 'Poor', 'Spoiled']
        
    def predict_quality(self, image_path: str) -> Tuple[str, float, np.ndarray]:
        """
        Predict quality of produce from image
        
        Returns:
            Tuple of (quality_label, confidence, all_probabilities)
        """
        self.model.eval()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        quality_label = self.quality_labels[predicted.item()]
        confidence_score = confidence.item()
        all_probs = probabilities.cpu().numpy()[0]
        
        return quality_label, confidence_score, all_probs
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract feature vector from image for downstream tasks"""
        self.model.eval()
        
        # Remove the final classification layer
        feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        feature_extractor = feature_extractor.to(self.device)
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = feature_extractor(image_tensor)
            features = features.squeeze().cpu().numpy()
        
        return features
    
    def train_model(self, train_loader, val_loader, epochs: int = 10):
        """Training loop for the vision model"""
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            train_acc = 100 * train_correct / len(train_loader.dataset)
            val_acc = 100 * val_correct / len(val_loader.dataset)
            
            logger.info(f"Epoch [{epoch+1}/{epochs}] "
                       f"Train Loss: {train_loss/len(train_loader):.4f}, "
                       f"Train Acc: {train_acc:.2f}%, "
                       f"Val Loss: {val_loss/len(val_loader):.4f}, "
                       f"Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_vision_model.pth')
            
            scheduler.step()
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")