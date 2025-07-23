import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from datetime import datetime
import json


class ValidationDebugger:
    def __init__(self, output_dir="debug_outputs"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_dir = os.path.join(output_dir, f"debug_{self.timestamp}")
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Create separate directories for logs and visualizations
        self.log_dir = os.path.join(self.debug_dir, "logs")
        self.viz_dir = os.path.join(self.debug_dir, "visualizations")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        
        self.sample_data = []
        
    def log_validation_sample(self, batch_idx, video_id, 
                            pred_offsets, gt_offsets, 
                            cls_logits, gt_labels,
                            cls_loss, reg_loss, 
                            masks=None):
        """Log detailed information about a validation sample"""
        sample_info = {
            'batch_idx': batch_idx,
            'video_id': video_id,
            'pred_offsets': pred_offsets.cpu().numpy().tolist() if torch.is_tensor(pred_offsets) else pred_offsets,
            'gt_offsets': gt_offsets.cpu().numpy().tolist() if torch.is_tensor(gt_offsets) else gt_offsets,
            'cls_logits': cls_logits.cpu().numpy().tolist() if torch.is_tensor(cls_logits) else cls_logits,
            'gt_labels': gt_labels.cpu().numpy().tolist() if torch.is_tensor(gt_labels) else gt_labels,
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'masks': masks.cpu().numpy().tolist() if masks is not None and torch.is_tensor(masks) else masks
        }
        self.sample_data.append(sample_info)
        
    def visualize_predictions(self, epoch, num_samples=5):
        """Create visualizations comparing predictions vs ground truth"""
        if not self.sample_data:
            return []
            
        visualization_paths = []
        
        # Select samples to visualize
        samples_to_viz = self.sample_data[:num_samples]
        
        for idx, sample in enumerate(samples_to_viz):
            fig, axes = plt.subplots(3, 1, figsize=(15, 10))
            
            # Get data
            pred_offsets = np.array(sample['pred_offsets'])
            gt_offsets = np.array(sample['gt_offsets'])
            cls_logits = np.array(sample['cls_logits'])
            gt_labels = np.array(sample['gt_labels'])
            
            # Flatten if needed
            if pred_offsets.ndim == 3:
                pred_offsets = pred_offsets.squeeze(0)
            if gt_offsets.ndim == 3:
                gt_offsets = gt_offsets.squeeze(0)
            if cls_logits.ndim == 3:
                cls_logits = cls_logits.squeeze()
            if gt_labels.ndim == 3:
                gt_labels = gt_labels.squeeze()
                
            seq_len = pred_offsets.shape[0]
            time_points = np.arange(seq_len)
            
            # Plot 1: Classification scores
            ax1 = axes[0]
            cls_probs = 1 / (1 + np.exp(-cls_logits))  # sigmoid
            ax1.plot(time_points, cls_probs, 'b-', label='Predicted Prob', alpha=0.7)
            ax1.scatter(time_points[gt_labels > 0], gt_labels[gt_labels > 0], 
                       color='red', s=50, label='GT Positive', zorder=5)
            ax1.set_ylabel('Classification Score')
            ax1.set_title(f'Video {sample["video_id"]} - Classification (Loss: {sample["cls_loss"]:.4f})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(-0.1, 1.1)
            
            # Plot 2: Offset predictions
            ax2 = axes[1]
            # Plot predicted offsets
            ax2.plot(time_points, pred_offsets[:, 0], 'b-', label='Pred Left Offset', alpha=0.7)
            ax2.plot(time_points, pred_offsets[:, 1], 'b--', label='Pred Right Offset', alpha=0.7)
            # Plot GT offsets only at positive positions
            positive_idx = gt_labels > 0
            if np.any(positive_idx):
                ax2.scatter(time_points[positive_idx], gt_offsets[positive_idx, 0], 
                           color='red', s=30, label='GT Left Offset', marker='o')
                ax2.scatter(time_points[positive_idx], gt_offsets[positive_idx, 1], 
                           color='darkred', s=30, label='GT Right Offset', marker='s')
            ax2.set_ylabel('Offset Value')
            ax2.set_title(f'Offset Predictions (Reg Loss: {sample["reg_loss"]:.4f})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Segments visualization
            ax3 = axes[2]
            # Draw predicted segments at high confidence points
            high_conf_idx = cls_probs > 0.5
            for t_idx in np.where(high_conf_idx)[0]:
                left_offset = pred_offsets[t_idx, 0]
                right_offset = pred_offsets[t_idx, 1]
                start = t_idx - left_offset
                end = t_idx + right_offset
                ax3.add_patch(patches.Rectangle((start, 0.6), end - start, 0.3,
                                              facecolor='blue', alpha=0.5))
                
            # Draw GT segments
            for t_idx in np.where(positive_idx)[0]:
                left_offset = gt_offsets[t_idx, 0]
                right_offset = gt_offsets[t_idx, 1]
                start = t_idx - left_offset
                end = t_idx + right_offset
                ax3.add_patch(patches.Rectangle((start, 0.1), end - start, 0.3,
                                              facecolor='red', alpha=0.5))
                
            ax3.set_xlim(0, seq_len)
            ax3.set_ylim(0, 1)
            ax3.set_xlabel('Time Steps')
            ax3.set_title('Segment Visualization (Blue: Predicted, Red: Ground Truth)')
            ax3.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = os.path.join(self.viz_dir, f'epoch_{epoch}_sample_{idx}_video_{sample["video_id"]}.png')
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            visualization_paths.append(viz_path)
        
        return visualization_paths
            
    def save_debug_logs(self, epoch):
        """Save detailed logs to JSON file"""
        log_path = os.path.join(self.log_dir, f'epoch_{epoch}_debug_log.json')
        log_paths = []
        
        # Calculate statistics
        if self.sample_data:
            reg_losses = [s['reg_loss'] for s in self.sample_data]
            cls_losses = [s['cls_loss'] for s in self.sample_data]
            
            stats = {
                'epoch': epoch,
                'num_samples': len(self.sample_data),
                'reg_loss_stats': {
                    'mean': np.mean(reg_losses),
                    'std': np.std(reg_losses),
                    'min': np.min(reg_losses),
                    'max': np.max(reg_losses),
                    'median': np.median(reg_losses)
                },
                'cls_loss_stats': {
                    'mean': np.mean(cls_losses),
                    'std': np.std(cls_losses),
                    'min': np.min(cls_losses),
                    'max': np.max(cls_losses),
                    'median': np.median(cls_losses)
                }
            }
            
            # Check for collapsed predictions
            all_pred_offsets = []
            for s in self.sample_data:
                pred_offsets = np.array(s['pred_offsets'])
                if pred_offsets.ndim == 3:
                    pred_offsets = pred_offsets.reshape(-1, 2)
                all_pred_offsets.append(pred_offsets)
            
            all_pred_offsets = np.vstack(all_pred_offsets)
            
            stats['prediction_analysis'] = {
                'left_offset_mean': float(np.mean(all_pred_offsets[:, 0])),
                'left_offset_std': float(np.std(all_pred_offsets[:, 0])),
                'right_offset_mean': float(np.mean(all_pred_offsets[:, 1])),
                'right_offset_std': float(np.std(all_pred_offsets[:, 1])),
                'offset_variance_warning': float(np.std(all_pred_offsets)) < 0.1
            }
            
            # Save full log
            full_log = {
                'stats': stats,
                'samples': self.sample_data[:20]  # Save first 20 samples
            }
            
            with open(log_path, 'w') as f:
                json.dump(full_log, f, indent=2)
            log_paths.append(log_path)
                
            # Create summary log
            summary_path = os.path.join(self.log_dir, f'epoch_{epoch}_summary.txt')
            with open(summary_path, 'w') as f:
                f.write(f"Epoch {epoch} Validation Debug Summary\n")
                f.write("="*50 + "\n\n")
                f.write(f"Regression Loss - Mean: {stats['reg_loss_stats']['mean']:.6f}, "
                       f"Std: {stats['reg_loss_stats']['std']:.6f}\n")
                f.write(f"Classification Loss - Mean: {stats['cls_loss_stats']['mean']:.6f}, "
                       f"Std: {stats['cls_loss_stats']['std']:.6f}\n\n")
                f.write(f"Prediction Analysis:\n")
                f.write(f"  Left Offset - Mean: {stats['prediction_analysis']['left_offset_mean']:.4f}, "
                       f"Std: {stats['prediction_analysis']['left_offset_std']:.4f}\n")
                f.write(f"  Right Offset - Mean: {stats['prediction_analysis']['right_offset_mean']:.4f}, "
                       f"Std: {stats['prediction_analysis']['right_offset_std']:.4f}\n")
                
                if stats['prediction_analysis']['offset_variance_warning']:
                    f.write("\n⚠️  WARNING: Low variance in predictions - possible model collapse!\n")
            
            log_paths.append(summary_path)
                    
        # Clear sample data for next epoch
        self.sample_data = []
        
        return log_paths
        
    def get_debug_summary(self):
        """Get a summary of the current debugging session"""
        return {
            'debug_dir': self.debug_dir,
            'num_samples_logged': len(self.sample_data),
            'timestamp': self.timestamp
        }