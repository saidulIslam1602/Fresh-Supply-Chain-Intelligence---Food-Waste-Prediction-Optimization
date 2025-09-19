        """
        Enhanced quality prediction with ensemble, TTA, and uncertainty
        
        Args:
            image_path: Path to image file
            use_tta: Use test-time augmentation
            return_uncertainty: Return uncertainty estimates
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)
            
            predictions = []
            uncertainties = []
            
            # Test-time augmentation
            tta_iterations = 8 if use_tta else 1
            
            for model in self.models:
                model.eval()
                model_predictions = []
                
                with torch.no_grad():
                    for _ in range(tta_iterations):
                        # Apply augmentation
                        if use_tta:
                            augmented = self.tta_transform(image=image_np)['image']
                        else:
                            augmented = self.val_transform(image=image_np)['image']
                        
                        input_tensor = augmented.unsqueeze(0).to(self.device)
                        
                        # Forward pass
                        outputs = model(input_tensor, mc_samples=10 if return_uncertainty else 1)
                        logits = outputs['logits']
                        probs = F.softmax(logits, dim=1)
                        
                        model_predictions.append(probs.cpu().numpy())
                        
                        if return_uncertainty and 'uncertainty' in outputs:
                            uncertainties.append(outputs['uncertainty'].cpu().numpy())
                
                # Average TTA predictions
                avg_prediction = np.mean(model_predictions, axis=0)
                predictions.append(avg_prediction)
            
            # Ensemble averaging
            final_probs = np.mean(predictions, axis=0)[0]
            predicted_class = np.argmax(final_probs)
            confidence = float(final_probs[predicted_class])
            quality_label = self.quality_labels[predicted_class]
            
            # Calculate prediction uncertainty
            prediction_uncertainty = 0.0
            if uncertainties:
                prediction_uncertainty = float(np.mean(uncertainties))
            
            # Ensemble uncertainty (disagreement between models)
            ensemble_uncertainty = float(np.std([pred[0][predicted_class] for pred in predictions]))
            
            # Quality assessment
            quality_assessment = self._assess_quality(quality_label, confidence, prediction_uncertainty)
            
            results = {
                'quality_label': quality_label,
                'confidence': confidence,
                'probabilities': final_probs.tolist(),
                'predicted_class': int(predicted_class),
                'quality_assessment': quality_assessment,
                'all_class_probs': {
                    label: float(prob) for label, prob in zip(self.quality_labels, final_probs)
                }
            }
            
            if return_uncertainty:
                results.update({
                    'prediction_uncertainty': prediction_uncertainty,
                    'ensemble_uncertainty': ensemble_uncertainty,
                    'total_uncertainty': prediction_uncertainty + ensemble_uncertainty
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in quality prediction: {e}")
            return {
                'quality_label': 'Unknown',
                'confidence': 0.0,
                'probabilities': [0.2] * 5,
                'error': str(e)
            }
    
    def _assess_quality(self, quality_label: str, confidence: float, uncertainty: float) -> Dict[str, any]:
        """Assess overall quality with recommendations"""
        
        threshold = self.quality_thresholds.get(quality_label, 0.5)
        is_reliable = confidence >= threshold and uncertainty < 0.3
        
        # Generate recommendations
        recommendations = []
        if quality_label in ['Poor', 'Spoiled']:
            recommendations.append("Immediate action required - remove from inventory")
            recommendations.append("Investigate storage conditions")
        elif quality_label == 'Fair':
            recommendations.append("Monitor closely - consider early distribution")
            recommendations.append("Check temperature and humidity levels")
        elif quality_label in ['Good', 'Fresh']:
            recommendations.append("Product quality acceptable")
            if quality_label == 'Fresh':
                recommendations.append("Optimal quality - prioritize for premium sales")
        
        if not is_reliable:
            recommendations.append("Low confidence prediction - manual inspection recommended")
        
        return {
            'is_reliable': is_reliable,
            'risk_level': self._calculate_risk_level(quality_label, confidence),
            'recommendations': recommendations,
            'shelf_life_estimate': self._estimate_remaining_shelf_life(quality_label),
            'action_priority': self._get_action_priority(quality_label, confidence)
        }
    
    def _calculate_risk_level(self, quality_label: str, confidence: float) -> str:
        """Calculate risk level for decision making"""
        risk_scores = {
            'Fresh': 0.1, 'Good': 0.3, 'Fair': 0.6, 'Poor': 0.8, 'Spoiled': 1.0
        }
        
        base_risk = risk_scores.get(quality_label, 0.5)
        confidence_factor = 1 - confidence
        total_risk = base_risk + confidence_factor * 0.3
        
        if total_risk < 0.3:
            return 'Low'
        elif total_risk < 0.6:
            return 'Medium'
        else:
            return 'High'
    
    def _estimate_remaining_shelf_life(self, quality_label: str) -> Dict[str, int]:
        """Estimate remaining shelf life based on quality"""
        estimates = {
            'Fresh': {'min_days': 5, 'max_days': 14},
            'Good': {'min_days': 3, 'max_days': 7},
            'Fair': {'min_days': 1, 'max_days': 3},
            'Poor': {'min_days': 0, 'max_days': 1},
            'Spoiled': {'min_days': 0, 'max_days': 0}
        }
        return estimates.get(quality_label, {'min_days': 0, 'max_days': 0})
    
    def _get_action_priority(self, quality_label: str, confidence: float) -> str:
        """Get action priority for operations team"""
        if quality_label in ['Poor', 'Spoiled']:
            return 'URGENT'
        elif quality_label == 'Fair':
            return 'HIGH'
        elif confidence < 0.7:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def train_epoch(self, train_loader, val_loader=None, epoch: int = 0) -> Dict[str, float]:
        """Train for one epoch with advanced techniques"""
        
        if not self.optimizer:
            raise ValueError("Training not setup. Call setup_training() first.")
        
        # Training phase
        for model in self.models:
            model.train()
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            total_loss = 0.0
            ensemble_predictions = []
            
            # Forward pass through ensemble
            for model in self.models:
                if self.scaler:  # Mixed precision training
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        logits = outputs['logits']
                        loss = self.criterion(logits, targets)
                        
                        # Add uncertainty loss if available
                        if 'uncertainty_logits' in outputs:
                            uncertainty_loss = self._uncertainty_loss(outputs['uncertainty_logits'], targets)
                            loss += 0.1 * uncertainty_loss
                        
                        total_loss += loss / len(self.models)
                else:
                    outputs = model(images)
                    logits = outputs['logits']
                    loss = self.criterion(logits, targets)
                    total_loss += loss / len(self.models)
                
                ensemble_predictions.append(F.softmax(logits, dim=1))
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()
            
            # Calculate ensemble accuracy
            ensemble_probs = torch.mean(torch.stack(ensemble_predictions), dim=0)
            _, predicted = torch.max(ensemble_probs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            train_loss += total_loss.item()
        
        # Validation phase
        val_metrics = {}
        if val_loader:
            val_metrics = self.validate(val_loader)
        
        # Update scheduler
        if self.scheduler:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics.get('val_loss', train_loss))
            else:
                self.scheduler.step()
        
        # Record metrics
        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        self.training_history['train_loss'].append(train_loss)
        self.training_history['train_acc'].append(train_acc)
        self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
        
        if val_metrics:
            self.training_history['val_loss'].append(val_metrics['val_loss'])
            self.training_history['val_acc'].append(val_metrics['val_acc'])
        
        metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        metrics.update(val_metrics)
        
        return metrics
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validation with ensemble"""
        
        for model in self.models:
            model.eval()
        
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                ensemble_predictions = []
                ensemble_loss = 0.0
                
                for model in self.models:
                    outputs = model(images)
                    logits = outputs['logits']
                    loss = self.criterion(logits, targets)
                    ensemble_loss += loss / len(self.models)
                    ensemble_predictions.append(F.softmax(logits, dim=1))
                
                # Ensemble prediction
                ensemble_probs = torch.mean(torch.stack(ensemble_predictions), dim=0)
                _, predicted = torch.max(ensemble_probs.data, 1)
                
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                val_loss += ensemble_loss.item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        return {
            'val_loss': val_loss,
            'val_acc': val_acc,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def _uncertainty_loss(self, uncertainty_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate uncertainty loss for better calibration"""
        # uncertainty_logits: (mc_samples, batch_size, num_classes)
        # targets: (batch_size,)
        
        mean_logits = torch.mean(uncertainty_logits, dim=0)
        variance = torch.var(uncertainty_logits, dim=0)
        
        # Entropy-based uncertainty loss
        mean_probs = F.softmax(mean_logits, dim=1)
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)
        
        # Encourage high entropy for incorrect predictions
        correct_mask = (torch.argmax(mean_logits, dim=1) == targets).float()
        uncertainty_loss = torch.mean((1 - correct_mask) * (1 - entropy))
        
        return uncertainty_loss
    
    def generate_gradcam(self, image_path: str, target_class: int = None) -> np.ndarray:
        """Generate Grad-CAM visualization for explainability"""
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)
            input_tensor = self.val_transform(image=image_np)['image'].unsqueeze(0).to(self.device)
            
            # Use first model for Grad-CAM
            model = self.models[0]
            model.eval()
            
            # Define target layer (last conv layer of backbone)
            target_layers = [model.backbone.features[-1]]
            
            # Create Grad-CAM
            cam = GradCAM(model=model, target_layers=target_layers)
            
            # Generate CAM
            if target_class is None:
                # Use predicted class
                with torch.no_grad():
                    outputs = model(input_tensor)
                    target_class = torch.argmax(outputs['logits'], dim=1).item()
            
            grayscale_cam = cam(input_tensor=input_tensor, targets=[target_class])
            grayscale_cam = grayscale_cam[0, :]
            
            # Overlay on original image
            rgb_img = np.float32(image_np) / 255
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            return visualization
            
        except ImportError:
            logger.warning("pytorch-grad-cam not available. Install with: pip install grad-cam")
            return np.array(image)
        except Exception as e:
            logger.error(f"Error generating Grad-CAM: {e}")
            return np.array(image)
    
    def save_model(self, path: str, include_optimizer: bool = True):
        """Save ensemble model"""
        save_dict = {
            'config': self.config.__dict__,
            'models': [model.state_dict() for model in self.models],
            'training_history': self.training_history,
            'quality_labels': self.quality_labels,
            'quality_thresholds': self.quality_thresholds
        }
        
        if include_optimizer and self.optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()
            if self.scheduler:
                save_dict['scheduler'] = self.scheduler.state_dict()
        
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str, load_optimizer: bool = False):
        """Load ensemble model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load config
        self.config = ModelConfig(**checkpoint['config'])
        
        # Recreate models
        self.models = nn.ModuleList([
            EnhancedVisionModel(self.config) for _ in range(len(checkpoint['models']))
        ])
        
        # Load model weights
        for i, model_state in enumerate(checkpoint['models']):
            self.models[i].load_state_dict(model_state)
            self.models[i].to(self.device)
        
        # Load training history
        self.training_history = checkpoint.get('training_history', {})
        self.quality_labels = checkpoint.get('quality_labels', self.quality_labels)
        self.quality_thresholds = checkpoint.get('quality_thresholds', self.quality_thresholds)
        
        # Load optimizer if requested
        if load_optimizer and 'optimizer' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        logger.info(f"Model loaded from {path}")
    
    def get_model_summary(self) -> Dict[str, any]:
        """Get comprehensive model summary"""
        total_params = sum(
            sum(p.numel() for p in model.parameters()) 
            for model in self.models
        )
        trainable_params = sum(
            sum(p.numel() for p in model.parameters() if p.requires_grad) 
            for model in self.models
        )
        
        return {
            'model_architecture': self.config.model_name,
            'ensemble_size': len(self.models),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'config': self.config.__dict__,
            'device': str(self.device),
            'training_history_length': len(self.training_history.get('train_loss', [])),
            'features': {
                'attention_mechanism': self.config.use_attention,
                'uncertainty_quantification': self.config.use_uncertainty,
                'test_time_augmentation': True,
                'ensemble_prediction': True,
                'grad_cam_visualization': self.config.use_gradcam
            }
        }

# Utility functions for model evaluation and analysis
def evaluate_model_performance(model: FreshProduceVisionModel, 
                             test_loader, 
                             save_plots: bool = True,
                             plot_dir: str = "./model_analysis") -> Dict[str, any]:
    """Comprehensive model evaluation"""
    
    # Get predictions
    val_results = model.validate(test_loader)
    predictions = val_results['predictions']
    targets = val_results['targets']
    
    # Classification report
    class_report = classification_report(
        targets, predictions, 
        target_names=model.quality_labels,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    if save_plots:
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=model.quality_labels,
                   yticklabels=model.quality_labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot training history
        if model.training_history['train_loss']:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss curves
            axes[0, 0].plot(model.training_history['train_loss'], label='Train Loss')
            if model.training_history['val_loss']:
                axes[0, 0].plot(model.training_history['val_loss'], label='Val Loss')
            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Accuracy curves
            axes[0, 1].plot(model.training_history['train_acc'], label='Train Acc')
            if model.training_history['val_acc']:
                axes[0, 1].plot(model.training_history['val_acc'], label='Val Acc')
            axes[0, 1].set_title('Accuracy Curves')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Learning rate schedule
            axes[1, 0].plot(model.training_history['learning_rates'])
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
            
            # Class distribution
            unique, counts = np.unique(targets, return_counts=True)
            axes[1, 1].bar([model.quality_labels[i] for i in unique], counts)
            axes[1, 1].set_title('Test Set Class Distribution')
            axes[1, 1].set_xlabel('Quality Class')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/training_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    return {
        'accuracy': val_results['val_acc'],
        'loss': val_results['val_loss'],
        'classification_report': class_report,
        'confusion_matrix': cm.tolist(),
        'per_class_accuracy': {
            model.quality_labels[i]: class_report[model.quality_labels[i]]['f1-score']
            for i in range(len(model.quality_labels))
        }
    }